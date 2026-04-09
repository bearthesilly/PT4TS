"""
PT_forecast_latent_v4: Channel-Independent Encoder-Decoder MFVI.

Each channel runs its own PT independently (shared weights).
No cross-channel ternary interaction — purely temporal MFVI per channel.
Based on Plan A (pure MSE end-to-end) with channel independence.

Key insight: many time series benchmarks have weak inter-channel correlation.
Channel independence avoids "noisy channel pollution" and often outperforms
cross-channel models (see PatchTST, TimeMixer channel_independent mode).
"""
import math
import copy
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
    """Decoder MFVI: Z_enc as fixed evidence, only updates future Z."""

    def __init__(self, args, encoder_iterator: PtEncoderIterator):
        super().__init__()
        self.config = config
        self.head_selection = PtHeadSelection(args)
        enc_hs = encoder_iterator.head_selection
        self.head_selection.ternary_factor_u_channel = enc_hs.ternary_factor_u_channel
        self.head_selection.ternary_factor_v_channel = enc_hs.ternary_factor_v_channel
        self.topic_modeling = encoder_iterator.topic_modeling
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

    def forward(self, z_enc, z_future, **ctx):
        P = z_enc.size(2)
        old_z_future = z_future
        z_all = torch.cat([z_enc, z_future], dim=2)
        z_all_normed = self.norm(z_all)
        m_t_all, m_c_all, _, _ = self.head_selection(qz=z_all_normed, **ctx)
        m_t_f = m_t_all[:, :, P:, :]
        m_c_f = m_c_all[:, :, P:, :]
        m_g_f = self.topic_modeling(z_all_normed[:, :, P:, :])
        z_future_new = (m_t_f + m_c_f + m_g_f) / self.config.regularize_z
        return (z_future_new + old_z_future) * 0.5


class PtLatentModelV4(nn.Module):
    """
    Channel-Independent Encoder-Decoder MFVI.

    Each of the N channels is treated as an independent univariate series.
    Weights are shared across all channels. The MFVI graph is temporal-only
    (channel ternary is trivially zero since N_eff=1).
    """

    def __init__(self, args):
        super().__init__()
        self.dim_z = args.d_model
        self.patch_len = args.patch_len
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.real_enc_in = args.enc_in
        self.num_enc_iter = args.e_layers
        self.num_dec_iter = max(args.d_layers, 1)

        self.patch_num = self.seq_len // self.patch_len
        self.future_patch_num = math.ceil(self.pred_len / self.patch_len)

        # Build iterators with enc_in=1 (channel independent)
        args_ci = copy.copy(args)
        args_ci.enc_in = 1
        self.encoder_iterator = PtEncoderIterator(args_ci)
        self.decoder_iterator = DecoderIterator(args_ci, self.encoder_iterator)

        # RoPE (only temporal matters; channel RoPE is trivial with N=1)
        cfg = PtConfig.from_dict(config.to_dict())
        cfg.hidden_size = self.dim_z
        cfg.num_attention_heads = args.n_heads
        cfg.head_dim = self.dim_z // args.n_heads
        self.rotary_emb_time = LlamaRotaryEmbedding(config=cfg)
        self.rotary_emb_channel = LlamaRotaryEmbedding(config=cfg)

        self.unary_factors = nn.Sequential(
            nn.Linear(self.patch_len, self.dim_z),
            nn.GELU(),
            nn.Linear(self.dim_z, self.dim_z),
        )
        self.patch_predictor = nn.Sequential(
            nn.Linear(self.dim_z, self.dim_z),
            nn.GELU(),
            nn.Linear(self.dim_z, self.patch_len),
        )

        self._ctx_cache = {}

    def _mfvi_ctx(self, B_eff, L_time, device, dtype):
        """Build MFVI context for channel-independent mode (N_eff=1)."""
        key = (B_eff, L_time, device, dtype)
        if key in self._ctx_cache:
            return self._ctx_cache[key]

        N_eff = 1
        mask_time = _build_dep_mask(B_eff * N_eff, L_time, device, dtype)
        mask_chan = _build_dep_mask(B_eff * L_time, N_eff, device, dtype)
        pid_t = torch.arange(L_time, device=device, dtype=torch.long)[None]
        pid_c = torch.arange(N_eff, device=device, dtype=torch.long)[None]
        dummy_t = torch.zeros(B_eff, L_time, self.dim_z, device=device, dtype=dtype)
        dummy_c = torch.zeros(B_eff * L_time, N_eff, self.dim_z, device=device, dtype=dtype)
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

    def forward(self, time_series):
        device = time_series.device
        B, T, N = time_series.shape
        P = self.patch_num
        Sf = self.future_patch_num

        # ---- Channel Independence: [B, T, N] → [B*N, T, 1] ----
        means = time_series.mean(1, keepdim=True).detach()  # [B, 1, N]
        x = time_series - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev  # [B, T, N]

        # Reshape: each channel becomes an independent sample
        x = x.permute(0, 2, 1).reshape(B * N, T)           # [B*N, T]
        x = x.unsqueeze(1).reshape(B * N, 1, P, self.patch_len)  # [B*N, 1, P, patch_len]

        unary_obs = self.unary_factors(x)  # [B*N, 1, P, d]
        dtype = unary_obs.dtype

        # ============================================================
        # Phase 1: Encoder MFVI (per-channel, temporal only)
        # ============================================================
        enc_ctx = self._mfvi_ctx(B * N, P, device, dtype)
        qz = unary_obs.clone()
        for _ in range(self.num_enc_iter):
            qz = self.encoder_iterator(unary_obs, qz, **enc_ctx)
        z_enc = qz  # [B*N, 1, P, d]

        # ============================================================
        # Phase 2: Dynamic prior
        # ============================================================
        z_summary = z_enc.mean(dim=2)  # [B*N, 1, d]
        z_future = z_summary.unsqueeze(2).expand(B * N, 1, Sf, self.dim_z).contiguous()

        # ============================================================
        # Phase 3: Decoder MFVI (per-channel, Z_enc as evidence)
        # ============================================================
        full_ctx = self._mfvi_ctx(B * N, P + Sf, device, dtype)
        for _ in range(self.num_dec_iter):
            z_future = self.decoder_iterator(z_enc, z_future, **full_ctx)
        z_dec = z_future  # [B*N, 1, Sf, d]

        # ============================================================
        # Phase 4: Prediction + reshape back
        # ============================================================
        patches_out = self.patch_predictor(z_dec)  # [B*N, 1, Sf, patch_len]
        dec_out = patches_out.reshape(B * N, -1)[:, :self.pred_len]  # [B*N, pred_len]
        dec_out = dec_out.reshape(B, N, self.pred_len).permute(0, 2, 1)  # [B, pred_len, N]

        # ---- Reverse instance norm ----
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand_as(dec_out)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand_as(dec_out)

        return dec_out


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = PtLatentModelV4(args)

    def forward(self, x_enc=None, x_mark_enc=None, x_dec=None,
                x_mark_dec=None, **kwargs):
        return self.model(time_series=x_enc)
