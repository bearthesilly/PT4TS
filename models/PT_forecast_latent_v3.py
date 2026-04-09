"""
PT_forecast_latent_v3: Causal MFVI Encoder + NextLat + Decoder MFVI.

Architecture:
  Phase 1: Causal Encoder MFVI — Z_enc[t] is a belief state of X_{1:t}
    - Causal temporal mask (Z[t] only receives from Z[s], s < t)
    - Temporal messageG uses only Part 1 (no backward flow)
    - Channel direction remains bidirectional
  Phase 2: NextLat self-supervised loss (training only)
    - Predict Z_enc[t+1] from Z_enc[t] via small MLP
    - Forces Z_enc to capture latent dynamics
  Phase 3: Dynamic prior from Z_enc
  Phase 4: Decoder MFVI — Z_enc as fixed evidence, bidirectional
  Phase 5: Per-patch MLP prediction head
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
    RopeApplier,
)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


# ======================================================================
# Mask builders
# ======================================================================

def _build_dep_mask(batch_dim: int, seq_len: int, device, dtype):
    """Bidirectional mask with masked self-loops (diagonal)."""
    mask2d = torch.ones(batch_dim, seq_len, device=device)
    converter = AttentionMaskConverter(is_causal=False)
    mask4d = converter.to_4d(mask2d, seq_len, dtype=dtype)
    diag = torch.eye(seq_len, dtype=mask4d.dtype, device=mask4d.device)[None, None]
    return mask4d.masked_fill(diag.bool(), torch.finfo(dtype).min)


def _build_causal_dep_mask(batch_dim: int, seq_len: int, device, dtype):
    """Causal mask: position t attends only to s < t (strict lower triangular)."""
    min_val = torch.finfo(dtype).min
    causal = torch.triu(torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype), diagonal=0)
    return causal[None, None, :, :].expand(batch_dim, 1, seq_len, seq_len)


# ======================================================================
# CausalHeadSelection — causal temporal, bidirectional channel
# ======================================================================

class CausalHeadSelection(nn.Module):
    """
    Like PtHeadSelection but with causal temporal message passing.

    Temporal direction:
      - Causal mask: Z[t] attends only to Z[s] for s < t
      - messageG uses ONLY Part 1 (forward flow), Part 2 (backward flow) removed
    Channel direction:
      - Fully bidirectional (same as PtHeadSelection)
    """

    def __init__(self, args):
        super().__init__()
        self.config = config
        self.dim_z = args.d_model
        self.enc_in = args.enc_in
        self.num_channels = args.n_heads
        self.ternary_rank = self.dim_z // self.num_channels
        self.regularize_h = 1 / self.dim_z

        self.ternary_factor_u_time = nn.Parameter(
            torch.empty(self.num_channels * self.ternary_rank, self.dim_z)
        )
        self.ternary_factor_v_time = nn.Parameter(
            torch.empty(self.num_channels * self.ternary_rank, self.dim_z)
        )
        self.ternary_factor_u_channel = nn.Parameter(
            torch.empty(self.num_channels * self.ternary_rank, self.dim_z)
        )
        self.ternary_factor_v_channel = nn.Parameter(
            torch.empty(self.num_channels * self.ternary_rank, self.dim_z)
        )
        self.dropout = nn.Dropout(config.dropout_prob_h)
        self._init_ternary()

    def _init_ternary(self):
        for p in [
            self.ternary_factor_u_time, self.ternary_factor_v_time,
            self.ternary_factor_u_channel, self.ternary_factor_v_channel,
        ]:
            nn.init.normal_(p, mean=0.0, std=config.ternary_initializer_range)

    def _calculate_messageF(self, qz, dependency_mask, position_embeddings,
                            ternary_factor_u, ternary_factor_v):
        bsz, seq_len, _ = qz.size()
        qz_u = F.linear(qz, ternary_factor_u) * config.ternary_factor_scaling
        qz_v = F.linear(qz, ternary_factor_v) * config.ternary_factor_scaling
        qz_u = qz_u.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        qz_v = qz_v.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        cos, sin = position_embeddings
        rope = RopeApplier(cos, sin)
        qz_uo = rope.apply_o(qz_u)
        qz_u = rope.apply(qz_u)
        qz_v = rope.apply(qz_v)
        message_F = torch.matmul(qz_u, qz_v.transpose(2, 3))
        if dependency_mask is not None:
            message_F = message_F + dependency_mask
        return message_F, qz_u, qz_v, qz_uo, bsz, seq_len, qz_u.dtype

    def _calculate_messageG_causal(self, qh, qz_v, bsz, seq_len,
                                   position_embeddings, ternary_factor_u):
        """Causal temporal messageG: only Part 1 (forward flow)."""
        cos, sin = position_embeddings
        rope = RopeApplier(cos, sin)
        qh_v1 = torch.matmul(qh, qz_v)  # Part 1 only
        qh_v1 = rope.apply_o(qh_v1)
        qh_v1 = qh_v1.transpose(1, 2).contiguous()
        qh_v1 = qh_v1.reshape(bsz, seq_len, self.num_channels * self.ternary_rank)
        message_G = torch.matmul(qh_v1, ternary_factor_u) * config.ternary_factor_scaling
        return message_G

    def _calculate_messageG_full(self, qh, qz_uo, qz_v, bsz, seq_len,
                                 position_embeddings, ternary_factor_u, ternary_factor_v):
        """Bidirectional messageG: Part 1 + Part 2 (for channel direction)."""
        cos, sin = position_embeddings
        rope = RopeApplier(cos, sin)
        qh_v1 = torch.matmul(qh, qz_v)
        qh_v2 = torch.matmul(qh.transpose(2, 3), qz_uo)
        qh_v1 = rope.apply_o(qh_v1)
        qh_v2 = rope.apply(qh_v2)
        qh_v1 = qh_v1.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        qh_v2 = qh_v2.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        message_G = (
            torch.matmul(qh_v1, ternary_factor_u)
            + torch.matmul(qh_v2, ternary_factor_v)
        ) * config.ternary_factor_scaling
        return message_G

    def forward(
        self,
        qz: torch.Tensor,
        dependency_mask_channel: Optional[torch.Tensor] = None,
        dependency_mask_time: Optional[torch.Tensor] = None,
        position_ids_time=None,
        position_ids_channel=None,
        output_dependencies: bool = False,
        position_embeddings_time=None,
        position_embeddings_channel=None,
    ):
        bs, num_channel, length, _ = qz.size()

        # --- Temporal messageF (causal mask applied via dependency_mask_time) ---
        qz_t = qz.view(bs * num_channel, length, -1)
        mF_time, qz_u_t, qz_v_t, qz_uo_t, bsz_t, slen_t, dtype_t = \
            self._calculate_messageF(
                qz_t, dependency_mask_time, position_embeddings_time,
                self.ternary_factor_u_time, self.ternary_factor_v_time,
            )

        # --- Channel messageF (bidirectional) ---
        qz_c = qz.transpose(1, 2).reshape(bs * length, num_channel, -1)
        mF_chan, qz_u_c, qz_v_c, qz_uo_c, bsz_c, slen_c, dtype_c = \
            self._calculate_messageF(
                qz_c, dependency_mask_channel, position_embeddings_channel,
                self.ternary_factor_u_channel, self.ternary_factor_v_channel,
            )

        # --- Joint H normalization ---
        mF_time_r = mF_time.view(bs, num_channel, self.num_channels, length, length) \
                           .permute(0, 2, 1, 3, 4)
        mF_chan_r = mF_chan.view(bs, length, self.num_channels, num_channel, num_channel) \
                          .permute(0, 2, 3, 1, 4)
        combined = torch.cat([mF_time_r, mF_chan_r], dim=-1)
        combined_qh = F.softmax(combined / self.regularize_h, dim=-1, dtype=torch.float32)
        qh_time_comb, qh_chan_comb = torch.split(combined_qh, [length, num_channel], dim=-1)

        qh_time_out = qh_time_comb.permute(0, 2, 1, 3, 4)
        qh_time = qh_time_out.reshape(bs * num_channel, self.num_channels, length, length).to(dtype_t)

        qh_chan_out = qh_chan_comb.permute(0, 3, 1, 2, 4)
        qh_chan = qh_chan_out.reshape(bs * length, self.num_channels, num_channel, num_channel).to(dtype_c)

        # --- Temporal messageG: CAUSAL (Part 1 only) ---
        mG_time = self._calculate_messageG_causal(
            qh_time, qz_v_t, bsz_t, slen_t,
            position_embeddings_time, self.ternary_factor_u_time,
        ).reshape(bs, num_channel, length, -1)

        # --- Channel messageG: BIDIRECTIONAL (Part 1 + Part 2) ---
        mG_chan = self._calculate_messageG_full(
            qh_chan, qz_uo_c, qz_v_c, bsz_c, slen_c,
            position_embeddings_channel,
            self.ternary_factor_u_channel, self.ternary_factor_v_channel,
        ).reshape(bs, num_channel, length, -1)

        return mG_time, mG_chan, qh_time_out, qh_chan_out


# ======================================================================
# CausalEncoderIterator
# ======================================================================

class CausalEncoderIterator(nn.Module):
    """MFVI iterator with causal temporal message passing."""

    def __init__(self, args):
        super().__init__()
        self.config = config
        self.head_selection = CausalHeadSelection(args)
        self.topic_modeling = PtTopicModeling(args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

    def forward(self, unary_potentials, qz, **ctx):
        old_qz = qz
        qz = self.norm(qz)
        m_t, m_c, _, _ = self.head_selection(qz=qz, **ctx)
        m_g = self.topic_modeling(qz)
        qz = (m_t + m_c + m_g + unary_potentials) / self.config.regularize_z
        qz = (qz + old_qz) * 0.5
        return qz


# ======================================================================
# DecoderIterator (reused from v2 — bidirectional, Z_enc as evidence)
# ======================================================================

class DecoderIterator(nn.Module):
    """
    Bidirectional decoder MFVI where Z_enc is fixed evidence.
    Own temporal ternary; shared channel ternary + binary with encoder.
    """

    def __init__(self, args, encoder_iterator):
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


# ======================================================================
# NextLat Predictor
# ======================================================================

class NextLatPredictor(nn.Module):
    """Predict Z[t+1] from Z[t]. Shared across channels and positions."""

    def __init__(self, dim_z):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.GELU(),
            nn.Linear(dim_z, dim_z),
        )

    def forward(self, z_enc):
        """
        Args:
            z_enc: [B, N, P, d] — causal belief states
        Returns:
            loss: scalar NextLat loss
        """
        z_input = z_enc[:, :, :-1, :]   # [B, N, P-1, d]  positions 0..P-2
        z_target = z_enc[:, :, 1:, :]    # [B, N, P-1, d]  positions 1..P-1
        z_pred = self.net(z_input)        # [B, N, P-1, d]
        loss = F.smooth_l1_loss(z_pred, z_target.detach())
        return loss


# ======================================================================
# Main Model
# ======================================================================

class PtLatentModelV3(nn.Module):
    """Causal Encoder MFVI + NextLat + Decoder MFVI."""

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

        # Causal encoder
        self.encoder_iterator = CausalEncoderIterator(args)
        # Bidirectional decoder (Z_enc as evidence)
        self.decoder_iterator = DecoderIterator(args, self.encoder_iterator)
        # NextLat predictor
        self.nextlat = NextLatPredictor(self.dim_z)
        self.lambda_nextlat = 0.1

        # RoPE
        cfg = PtConfig.from_dict(config.to_dict())
        cfg.hidden_size = self.dim_z
        cfg.num_attention_heads = args.n_heads
        cfg.head_dim = self.dim_z // args.n_heads
        self.rotary_emb_time = LlamaRotaryEmbedding(config=cfg)
        self.rotary_emb_channel = LlamaRotaryEmbedding(config=cfg)

        # Unary MLP
        self.unary_factors = nn.Sequential(
            nn.Linear(self.patch_len, self.dim_z),
            nn.GELU(),
            nn.Linear(self.dim_z, self.dim_z),
        )
        # Prediction head
        self.patch_predictor = nn.Sequential(
            nn.Linear(self.dim_z, self.dim_z),
            nn.GELU(),
            nn.Linear(self.dim_z, self.patch_len),
        )

        self._ctx_cache = {}

    def _mfvi_ctx(self, B, N, L_time, device, dtype, causal=False):
        key = (B, N, L_time, device, dtype, causal)
        if key in self._ctx_cache:
            return self._ctx_cache[key]

        if causal:
            mask_time = _build_causal_dep_mask(B * N, L_time, device, dtype)
        else:
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

    def forward(self, time_series, is_training=True):
        device = time_series.device
        B = time_series.size(0)
        N = self.enc_in
        P = self.patch_num
        Sf = self.future_patch_num

        # ---- instance normalization ----
        means = time_series.mean(1, keepdim=True).detach()
        x = time_series - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # ---- patching + unary ----
        x = x.transpose(1, 2).reshape(B, N, P, self.patch_len)
        unary_obs = self.unary_factors(x)
        dtype = unary_obs.dtype

        # ============================================================
        # Phase 1: Causal Encoder MFVI
        # ============================================================
        enc_ctx = self._mfvi_ctx(B, N, P, device, dtype, causal=True)
        qz = unary_obs.clone()
        for _ in range(self.num_enc_iter):
            qz = self.encoder_iterator(unary_obs, qz, **enc_ctx)
        z_enc = qz  # [B, N, P, d] — causal belief states

        # ============================================================
        # Phase 2: NextLat loss (training only)
        # ============================================================
        nextlat_loss = None
        if is_training and P > 1:
            nextlat_loss = self.nextlat(z_enc)

        # ============================================================
        # Phase 3: Dynamic prior
        # ============================================================
        z_summary = z_enc.mean(dim=2)
        z_future = z_summary.unsqueeze(2).expand(B, N, Sf, self.dim_z).contiguous()

        # ============================================================
        # Phase 4: Decoder MFVI (bidirectional, Z_enc as evidence)
        # ============================================================
        full_ctx = self._mfvi_ctx(B, N, P + Sf, device, dtype, causal=False)
        for _ in range(self.num_dec_iter):
            z_future = self.decoder_iterator(z_enc, z_future, **full_ctx)
        z_dec = z_future

        # ============================================================
        # Phase 5: Prediction
        # ============================================================
        patches_out = self.patch_predictor(z_dec)
        dec_out = patches_out.reshape(B, N, -1)[:, :, :self.pred_len]
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand_as(dec_out)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand_as(dec_out)

        if nextlat_loss is not None:
            return dec_out, nextlat_loss
        return dec_out


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = PtLatentModelV3(args)

    def forward(self, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                y_true=None, **kwargs):
        if self.training:
            pred, nextlat_loss = self.model(time_series=x_enc, is_training=True)
            return pred, nextlat_loss
        return self.model(time_series=x_enc, is_training=False)
