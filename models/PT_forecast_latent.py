"""
PT_forecast_latent: Encoder-Decoder MFVI with Latent Consistency
for Time Series Forecasting.

Based on ST-PT (PT_forecast_v15), extended with:
  - Decoder MFVI phase for future latent state inference
  - Oracle MFVI + latent consistency loss (self-supervised)
  - Parallel future-state inference (no accumulative error)

Key idea: treat forecasting as posterior inference on a partially-observed CRF.
Observed patches have observation-driven unary potentials; future patches have
a learnable prior.  An Encoder produces causal belief states Z_enc from the
observed region, then a Decoder propagates MFVI into the future region while
keeping Z_enc fixed.  During training an oracle path (full-sequence MFVI with
ground-truth future) provides a stop-gradient latent consistency target.
"""
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_pt import PtConfig
from .PT_forecast_v15 import (
    Config,
    config,
    POTENTIAL2ACT,
    PtEncoderIterator,
)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.bert.modeling_bert import MaskedLMOutput


def _build_dep_mask(batch_dim: int, seq_len: int, device, dtype):
    """4-D dependency mask with masked self-loops (diagonal)."""
    mask2d = torch.ones(batch_dim, seq_len, device=device)
    converter = AttentionMaskConverter(is_causal=False)
    mask4d = converter.to_4d(mask2d, seq_len, dtype=dtype)
    diag = torch.eye(seq_len, dtype=mask4d.dtype, device=mask4d.device)[None, None]
    return mask4d.masked_fill(diag.bool(), torch.finfo(dtype).min)


class PtLatentModel(nn.Module):
    """Core model: Encoder-Decoder MFVI with latent consistency."""

    def __init__(self, args):
        super().__init__()
        self.dim_z = args.d_model
        self.patch_len = args.patch_len
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.num_enc_iter = args.e_layers
        self.num_dec_iter = max(args.d_layers, 1)

        self.patch_num = self.seq_len // self.patch_len
        self.future_patch_num = math.ceil(self.pred_len / self.patch_len)
        self.padded_pred_len = self.future_patch_num * self.patch_len
        self.total_patches = self.patch_num + self.future_patch_num

        # Shared MFVI iterator — encoder & decoder use the same CRF factors
        self.iterator = PtEncoderIterator(args)
        self.z_norm = POTENTIAL2ACT[config.potential_func_z](
            dim=-1, eps=config.potential_eps
        )

        # RoPE (separate for time / channel axes)
        cfg = PtConfig.from_dict(config.to_dict())
        cfg.hidden_size = self.dim_z
        cfg.num_attention_heads = args.n_heads
        cfg.head_dim = self.dim_z // args.n_heads
        self.rotary_emb_time = LlamaRotaryEmbedding(config=cfg)
        self.rotary_emb_channel = LlamaRotaryEmbedding(config=cfg)

        # Patch embedding (unary factor MLP)
        self.unary_factors = nn.Sequential(
            nn.Linear(self.patch_len, self.dim_z),
            nn.GELU(),
            nn.Linear(self.dim_z, self.dim_z),
        )

        # Learnable prior for future Z nodes (replaces observation-driven unary)
        self.z_prior = nn.Parameter(torch.empty(self.dim_z))
        nn.init.normal_(self.z_prior, std=0.02)

        # Prediction head: future Z → pred_len per channel
        self.prediction = nn.Linear(
            self.future_patch_num * self.dim_z, self.pred_len, bias=True
        )

        # Loss hyper-parameters (tunable)
        self.lambda_latent = 1.0
        self.lambda_kl = 0.1

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _mfvi_ctx(self, B, N, L_time, device, dtype):
        """Build masks, position IDs, and RoPE for a temporal length *L_time*."""
        mask_time = _build_dep_mask(B * N, L_time, device, dtype)
        mask_chan = _build_dep_mask(B * L_time, N, device, dtype)

        pid_t = torch.arange(L_time, device=device, dtype=torch.long)[None]
        pid_c = torch.arange(N, device=device, dtype=torch.long)[None]

        dummy_t = torch.zeros(B * N, L_time, self.dim_z, device=device, dtype=dtype)
        dummy_c = torch.zeros(B * L_time, N, self.dim_z, device=device, dtype=dtype)
        rope_t = self.rotary_emb_time(dummy_t, pid_t)
        rope_c = self.rotary_emb_channel(dummy_c, pid_c)

        return dict(
            dependency_mask_time=mask_time,
            dependency_mask_channel=mask_chan,
            position_ids_time=pid_t,
            position_ids_channel=pid_c,
            position_embeddings_time=rope_t,
            position_embeddings_channel=rope_c,
        )

    def _mfvi_loop(self, unary, qz, n_iter, ctx):
        """Standard MFVI: *n_iter* iterations of the shared iterator."""
        for _ in range(n_iter):
            qz = self.iterator(unary, qz, **ctx)
        return qz

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, time_series, y_true=None):
        """
        Args:
            time_series: [B, T, N]  observed multivariate series
            y_true:      [B, S, N]  ground-truth future (training only)
        Returns:
            predictions   [B, pred_len, N]
            aux_loss      scalar  (only when y_true is given)
        """
        device = time_series.device
        B = time_series.size(0)
        N = self.enc_in
        P = self.patch_num
        Sf = self.future_patch_num

        # ---- instance normalization ----
        means = time_series.mean(1, keepdim=True).detach()
        x = time_series - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x = x / stdev

        # ---- patching: [B, T, N] → [B, N, P, patch_len] ----
        x = x.transpose(1, 2).reshape(B, N, P, self.patch_len)
        unary_obs = self.unary_factors(x)          # [B, N, P, d]
        dtype = unary_obs.dtype

        # ============================================================
        # Phase 1  —  Encoder MFVI  (observed region only)
        # ============================================================
        enc_ctx = self._mfvi_ctx(B, N, P, device, dtype)
        z_enc = self._mfvi_loop(
            unary_obs, unary_obs.clone(), self.num_enc_iter, enc_ctx
        )                                          # [B, N, P, d]

        # ============================================================
        # Phase 2  —  Decoder MFVI  (extend to P+Sf, fix z_enc)
        # ============================================================
        full_ctx = self._mfvi_ctx(B, N, P + Sf, device, dtype)

        prior_exp = self.z_prior.view(1, 1, 1, -1).expand(B, N, Sf, -1)
        unary_full = torch.cat([unary_obs, prior_exp], dim=2)   # [B, N, P+Sf, d]

        z_full = torch.cat([z_enc, prior_exp], dim=2)
        for _ in range(self.num_dec_iter):
            z_full = self.iterator(unary_full, z_full, **full_ctx)
            z_full = torch.cat([z_enc, z_full[:, :, P:, :]], dim=2)

        z_dec = z_full[:, :, P:, :]                # [B, N, Sf, d]

        # ============================================================
        # Prediction head
        # ============================================================
        dec_out = self.prediction(
            z_dec.reshape(B, N, -1)
        ).permute(0, 2, 1)                         # [B, pred_len, N]

        # de-normalise
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand_as(dec_out)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand_as(dec_out)

        # ============================================================
        # Oracle path — latent consistency target  (training only)
        # ============================================================
        if y_true is not None:
            with torch.no_grad():
                # normalise GT with the same observed statistics
                y_n = (y_true - means) / stdev              # [B, S, N]
                y_n = y_n.transpose(1, 2)                   # [B, N, S]
                if self.padded_pred_len > self.pred_len:
                    y_n = F.pad(y_n, (0, self.padded_pred_len - self.pred_len))
                y_n = y_n.reshape(B, N, Sf, self.patch_len)

                unary_gt = self.unary_factors(y_n)
                unary_oracle = torch.cat(
                    [unary_obs.detach(), unary_gt], dim=2
                )
                z_target = self._mfvi_loop(
                    unary_oracle, unary_oracle.clone(),
                    self.num_enc_iter, full_ctx,
                )[:, :, P:, :]                              # [B, N, Sf, d]

            # SmoothL1 on raw Z scores
            loss_latent = F.smooth_l1_loss(z_dec, z_target)

            # KL on SquaredSoftmax-normalised distributions
            p = self.z_norm(z_target)
            q = self.z_norm(z_dec)
            loss_kl = F.kl_div((q + 1e-8).log(), p, reduction="batchmean")

            aux_loss = self.lambda_latent * loss_latent + self.lambda_kl * loss_kl
            return dec_out, aux_loss

        return dec_out


# ======================================================================
# Thin wrapper expected by the Time-Series-Library registry
# ======================================================================
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = PtLatentModel(args)

    def forward(
        self,
        x_enc: Optional[torch.Tensor] = None,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        y_true: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if y_true is not None:
            pred, aux_loss = self.model(time_series=x_enc, y_true=y_true)
            return pred, aux_loss
        return self.model(time_series=x_enc)
