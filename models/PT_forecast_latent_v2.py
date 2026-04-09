"""
PT_forecast_latent_v2: Encoder-Decoder MFVI with pure MSE end-to-end training.

Key design:
  - Z_enc is pure evidence: fixed, never updated in the decoder
  - Future Z has ZERO unary — driven entirely by ternary messages from Z_enc
  - DecoderIterator: computes ternary messages on the full [P+Sf] graph
    but only updates future Z; Z_enc participates as a fixed source
  - Separate decoder temporal ternary; shared channel ternary + binary
  - Dynamic prior: future Z initialized from mean(Z_enc)
  - Trained end-to-end with MSE only — no oracle, no auxiliary loss
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_pt import PtConfig
from .PT_forecast_v15 import (
    config,
    POTENTIAL2ACT,
    PtHeadSelection,
    PtTopicModeling,
    PtEncoderIterator,
)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def _build_dep_mask(batch_dim: int, seq_len: int, device, dtype):
    """4-D dependency mask with masked self-loops (diagonal)."""
    mask2d = torch.ones(batch_dim, seq_len, device=device)
    converter = AttentionMaskConverter(is_causal=False)
    mask4d = converter.to_4d(mask2d, seq_len, dtype=dtype)
    diag = torch.eye(seq_len, dtype=mask4d.dtype, device=mask4d.device)[None, None]
    return mask4d.masked_fill(diag.bool(), torch.finfo(dtype).min)


class DecoderIterator(nn.Module):
    """
    CRF decoder iterator where Z_enc is fixed evidence.

    Unlike PtEncoderIterator which updates ALL positions, this iterator:
    1. Concatenates [Z_enc, Z_future] for ternary message computation
    2. Runs head_selection on the full [P+Sf] graph so Z_enc can send
       messages to Z_future through ternary factors
    3. Extracts only future-position messages
    4. Runs binary (FFN) only on Z_future
    5. Updates only Z_future (no unary, no update to Z_enc)

    Z_enc participates purely as a fixed message source in the ternary factors.
    """

    def __init__(self, args, encoder_iterator: PtEncoderIterator):
        super().__init__()
        self.config = config

        # Own temporal ternary factors (decoder learns predictive dynamics)
        self.head_selection = PtHeadSelection(args)
        # Share channel ternary with encoder (cross-variable structure is universal)
        enc_hs = encoder_iterator.head_selection
        self.head_selection.ternary_factor_u_channel = enc_hs.ternary_factor_u_channel
        self.head_selection.ternary_factor_v_channel = enc_hs.ternary_factor_v_channel

        # Share binary (FFN / topic modeling) with encoder
        self.topic_modeling = encoder_iterator.topic_modeling

        self.norm = POTENTIAL2ACT[config.potential_func_z](
            dim=-1, eps=config.potential_eps
        )

    def forward(
        self,
        z_enc: torch.Tensor,
        z_future: torch.Tensor,
        dependency_mask_channel: Optional[torch.Tensor] = None,
        dependency_mask_time: Optional[torch.Tensor] = None,
        position_ids_time: Optional[torch.LongTensor] = None,
        position_ids_channel: Optional[torch.LongTensor] = None,
        position_embeddings_time: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_channel: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_enc:    [B, N, P, d]  — fixed evidence, NOT updated
            z_future: [B, N, Sf, d] — to be updated
            ctx kwargs: masks/RoPE for the full [P+Sf] graph

        Returns:
            z_future_updated: [B, N, Sf, d]
        """
        P = z_enc.size(2)
        old_z_future = z_future

        # --- Concatenate and normalize all Z for message computation ---
        z_all = torch.cat([z_enc, z_future], dim=2)  # [B, N, P+Sf, d]
        z_all_normed = self.norm(z_all)

        # --- Head selection on the FULL [P+Sf] graph ---
        m_t_all, m_c_all, _, _ = self.head_selection(
            qz=z_all_normed,
            dependency_mask_channel=dependency_mask_channel,
            dependency_mask_time=dependency_mask_time,
            position_ids_time=position_ids_time,
            position_ids_channel=position_ids_channel,
            position_embeddings_time=position_embeddings_time,
            position_embeddings_channel=position_embeddings_channel,
        )

        # --- Extract messages for future positions ONLY ---
        m_t_future = m_t_all[:, :, P:, :]
        m_c_future = m_c_all[:, :, P:, :]

        # --- Binary (FFN) on future Z only ---
        z_future_normed = z_all_normed[:, :, P:, :]
        m_g_future = self.topic_modeling(z_future_normed)

        # --- Update: no unary for future positions ---
        z_future_new = (
            (m_t_future + m_c_future + m_g_future) / self.config.regularize_z
        )

        # --- Damping ---
        z_future = (z_future_new + old_z_future) * 0.5

        return z_future


class PtLatentModelV2(nn.Module):
    """Encoder-Decoder MFVI v2 trained end-to-end with MSE only."""

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

        # --- Encoder iterator (standard ST-PT) ---
        self.encoder_iterator = PtEncoderIterator(args)

        # --- Decoder iterator (Z_enc as fixed evidence) ---
        self.decoder_iterator = DecoderIterator(args, self.encoder_iterator)

        # RoPE
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

        # Per-patch prediction head
        self.patch_predictor = nn.Sequential(
            nn.Linear(self.dim_z, self.dim_z),
            nn.GELU(),
            nn.Linear(self.dim_z, self.patch_len),
        )

        # MFVI context cache
        self._ctx_cache = {}

    # ------------------------------------------------------------------
    def _mfvi_ctx(self, B, N, L_time, device, dtype):
        key = (B, N, L_time, device, dtype)
        if key in self._ctx_cache:
            return self._ctx_cache[key]

        mask_time = _build_dep_mask(B * N, L_time, device, dtype)
        mask_chan = _build_dep_mask(B * L_time, N, device, dtype)
        pid_t = torch.arange(L_time, device=device, dtype=torch.long)[None]
        pid_c = torch.arange(N, device=device, dtype=torch.long)[None]
        dummy_t = torch.zeros(B * N, L_time, self.dim_z, device=device, dtype=dtype)
        dummy_c = torch.zeros(B * L_time, N, self.dim_z, device=device, dtype=dtype)
        rope_t = self.rotary_emb_time(dummy_t, pid_t)
        rope_c = self.rotary_emb_channel(dummy_c, pid_c)

        ctx = dict(
            dependency_mask_time=mask_time,
            dependency_mask_channel=mask_chan,
            position_ids_time=pid_t,
            position_ids_channel=pid_c,
            position_embeddings_time=rope_t,
            position_embeddings_channel=rope_c,
        )
        self._ctx_cache[key] = ctx
        return ctx

    # ------------------------------------------------------------------
    def forward(self, time_series):
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

        # ---- patching ----
        x = x.transpose(1, 2).reshape(B, N, P, self.patch_len)
        unary_obs = self.unary_factors(x)  # [B, N, P, d]
        dtype = unary_obs.dtype

        # ============================================================
        # Phase 1: Encoder MFVI (observed region only)
        # ============================================================
        enc_ctx = self._mfvi_ctx(B, N, P, device, dtype)
        qz = unary_obs.clone()
        for _ in range(self.num_enc_iter):
            qz = self.encoder_iterator(unary_obs, qz, **enc_ctx)
        z_enc = qz  # [B, N, P, d]

        # ============================================================
        # Phase 2: Dynamic prior from Z_enc
        # ============================================================
        z_summary = z_enc.mean(dim=2)  # [B, N, d]
        z_future = z_summary.unsqueeze(2).expand(B, N, Sf, self.dim_z).contiguous()

        # ============================================================
        # Phase 3: Decoder MFVI — Z_enc is fixed evidence
        # ============================================================
        full_ctx = self._mfvi_ctx(B, N, P + Sf, device, dtype)

        for _ in range(self.num_dec_iter):
            z_future = self.decoder_iterator(z_enc, z_future, **full_ctx)
        z_dec = z_future  # [B, N, Sf, d]

        # ============================================================
        # Phase 4: Per-patch prediction
        # ============================================================
        patches_out = self.patch_predictor(z_dec)  # [B, N, Sf, patch_len]
        dec_out = patches_out.reshape(B, N, -1)
        dec_out = dec_out[:, :, :self.pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, N]

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand_as(dec_out)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand_as(dec_out)

        return dec_out


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = PtLatentModelV2(args)

    def forward(
        self,
        x_enc: Optional[torch.Tensor] = None,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(time_series=x_enc)
