"""
PT_forecast_latent_v5: Causal MFVI + AR Rollout + Decoder MFVI Correction.

Pipeline:
  Phase 1: Causal Encoder MFVI → Z_enc (belief states)
  Phase 2: NextLat loss (1-step + multi-step, training only)
  Phase 3: AR Rollout with residual transition → Z_future_init
  Phase 4: Decoder MFVI correction (Z_enc evidence + AR init) → Z_dec
  Phase 5: Per-patch MLP prediction

Key idea: AR rollout provides dynamics-informed initialization for decoder
MFVI, which then corrects accumulative errors via parallel CRF inference.
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

def _build_dep_mask(batch_dim, seq_len, device, dtype):
    mask2d = torch.ones(batch_dim, seq_len, device=device)
    converter = AttentionMaskConverter(is_causal=False)
    mask4d = converter.to_4d(mask2d, seq_len, dtype=dtype)
    diag = torch.eye(seq_len, dtype=mask4d.dtype, device=mask4d.device)[None, None]
    return mask4d.masked_fill(diag.bool(), torch.finfo(dtype).min)


def _build_causal_dep_mask(batch_dim, seq_len, device, dtype):
    min_val = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype), diagonal=0
    )
    return causal[None, None].expand(batch_dim, 1, seq_len, seq_len)


# ======================================================================
# CausalHeadSelection (from v3)
# ======================================================================

class CausalHeadSelection(nn.Module):
    """Causal temporal (Part 1 only) + bidirectional channel."""

    def __init__(self, args):
        super().__init__()
        self.config = config
        self.dim_z = args.d_model
        self.enc_in = args.enc_in
        self.num_channels = args.n_heads
        self.ternary_rank = self.dim_z // self.num_channels
        self.regularize_h = 1 / self.dim_z

        self.ternary_factor_u_time = nn.Parameter(
            torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_v_time = nn.Parameter(
            torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_u_channel = nn.Parameter(
            torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_v_channel = nn.Parameter(
            torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.dropout = nn.Dropout(config.dropout_prob_h)
        for p in [self.ternary_factor_u_time, self.ternary_factor_v_time,
                  self.ternary_factor_u_channel, self.ternary_factor_v_channel]:
            nn.init.normal_(p, mean=0.0, std=config.ternary_initializer_range)

    def _messageF(self, qz, mask, pe, u, v):
        bsz, L, _ = qz.size()
        qz_u = F.linear(qz, u) * config.ternary_factor_scaling
        qz_v = F.linear(qz, v) * config.ternary_factor_scaling
        qz_u = qz_u.view(bsz, L, self.num_channels, self.ternary_rank).transpose(1, 2)
        qz_v = qz_v.view(bsz, L, self.num_channels, self.ternary_rank).transpose(1, 2)
        rope = RopeApplier(*pe)
        qz_uo = rope.apply_o(qz_u)
        qz_u = rope.apply(qz_u)
        qz_v = rope.apply(qz_v)
        mF = torch.matmul(qz_u, qz_v.transpose(2, 3))
        if mask is not None:
            mF = mF + mask
        return mF, qz_u, qz_v, qz_uo, bsz, L, qz_u.dtype

    def _messageG_causal(self, qh, qz_v, bsz, L, pe, u):
        rope = RopeApplier(*pe)
        v1 = rope.apply_o(torch.matmul(qh, qz_v))
        v1 = v1.transpose(1, 2).contiguous().reshape(bsz, L, -1)
        return torch.matmul(v1, u) * config.ternary_factor_scaling

    def _messageG_full(self, qh, qz_uo, qz_v, bsz, L, pe, u, v):
        rope = RopeApplier(*pe)
        v1 = rope.apply_o(torch.matmul(qh, qz_v))
        v2 = rope.apply(torch.matmul(qh.transpose(2, 3), qz_uo))
        v1 = v1.transpose(1, 2).contiguous().reshape(bsz, L, -1)
        v2 = v2.transpose(1, 2).contiguous().reshape(bsz, L, -1)
        return (torch.matmul(v1, u) + torch.matmul(v2, v)) * config.ternary_factor_scaling

    def forward(self, qz, dependency_mask_channel=None, dependency_mask_time=None,
                position_ids_time=None, position_ids_channel=None,
                output_dependencies=False,
                position_embeddings_time=None, position_embeddings_channel=None):
        bs, N, L, _ = qz.size()
        qz_t = qz.view(bs * N, L, -1)
        mF_t, _, qz_v_t, _, bsz_t, slen_t, dt_t = self._messageF(
            qz_t, dependency_mask_time, position_embeddings_time,
            self.ternary_factor_u_time, self.ternary_factor_v_time)
        qz_c = qz.transpose(1, 2).reshape(bs * L, N, -1)
        mF_c, _, qz_v_c, qz_uo_c, bsz_c, slen_c, dt_c = self._messageF(
            qz_c, dependency_mask_channel, position_embeddings_channel,
            self.ternary_factor_u_channel, self.ternary_factor_v_channel)

        mF_t_r = mF_t.view(bs, N, self.num_channels, L, L).permute(0, 2, 1, 3, 4)
        mF_c_r = mF_c.view(bs, L, self.num_channels, N, N).permute(0, 2, 3, 1, 4)
        comb = F.softmax(torch.cat([mF_t_r, mF_c_r], -1) / self.regularize_h, dim=-1, dtype=torch.float32)
        qh_t_c, qh_c_c = torch.split(comb, [L, N], dim=-1)

        qh_t = qh_t_c.permute(0, 2, 1, 3, 4).reshape(bs * N, self.num_channels, L, L).to(dt_t)
        qh_c = qh_c_c.permute(0, 3, 1, 2, 4).reshape(bs * L, self.num_channels, N, N).to(dt_c)

        mG_t = self._messageG_causal(
            qh_t, qz_v_t, bsz_t, slen_t, position_embeddings_time,
            self.ternary_factor_u_time).reshape(bs, N, L, -1)
        mG_c = self._messageG_full(
            qh_c, qz_uo_c, qz_v_c, bsz_c, slen_c, position_embeddings_channel,
            self.ternary_factor_u_channel, self.ternary_factor_v_channel).reshape(bs, N, L, -1)
        return mG_t, mG_c, None, None


# ======================================================================
# CausalEncoderIterator
# ======================================================================

class CausalEncoderIterator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = config
        self.head_selection = CausalHeadSelection(args)
        self.topic_modeling = PtTopicModeling(args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

    def forward(self, unary, qz, **ctx):
        old = qz
        qz = self.norm(qz)
        m_t, m_c, _, _ = self.head_selection(qz=qz, **ctx)
        m_g = self.topic_modeling(qz)
        qz = (m_t + m_c + m_g + unary) / self.config.regularize_z
        return (qz + old) * 0.5


# ======================================================================
# DecoderIterator (bidirectional, Z_enc as evidence)
# ======================================================================

class DecoderIterator(nn.Module):
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
        old = z_future
        z_all = torch.cat([z_enc, z_future], dim=2)
        z_normed = self.norm(z_all)
        m_t, m_c, _, _ = self.head_selection(qz=z_normed, **ctx)
        m_g = self.topic_modeling(z_normed[:, :, P:, :])
        z_new = (m_t[:, :, P:, :] + m_c[:, :, P:, :] + m_g) / self.config.regularize_z
        return (z_new + old) * 0.5


# ======================================================================
# AR Transition + NextLat
# ======================================================================

class ResidualTransition(nn.Module):
    """f_psi(z) = z + MLP(z). Residual ensures stability over long rollouts."""

    def __init__(self, dim_z):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_z, dim_z),
            nn.GELU(),
            nn.Linear(dim_z, dim_z),
        )

    def forward(self, z):
        return z + self.net(z)

    def rollout(self, z_last, steps):
        """Autoregressive rollout from z_last for `steps` steps."""
        zs = []
        z = z_last
        for _ in range(steps):
            z = self.forward(z)
            zs.append(z)
        return torch.stack(zs, dim=2)  # [B, N, steps, d]

    def nextlat_loss(self, z_enc):
        """
        Delta-prediction NextLat loss: train MLP to predict HOW Z changes,
        not just the next Z (which is trivially close due to residual).

        1-step: MLP(Z_t) should match (Z_{t+1} - Z_t)
        2-step: MLP(Z_t) + MLP(Z_t + MLP(Z_t)) should match (Z_{t+2} - Z_t)
        """
        P = z_enc.size(2)
        if P < 2:
            return torch.tensor(0.0, device=z_enc.device)

        z_in = z_enc[:, :, :-1, :].detach()
        z_tgt = z_enc[:, :, 1:, :].detach()
        delta_true = z_tgt - z_in                # actual change
        delta_pred = self.net(z_in)               # predicted change
        loss_1 = F.smooth_l1_loss(delta_pred, delta_true)

        if P < 3:
            return loss_1

        z_in_2 = z_enc[:, :, :-2, :].detach()
        z_tgt_2 = z_enc[:, :, 2:, :].detach()
        delta_true_2 = z_tgt_2 - z_in_2          # 2-step actual change
        step1 = self.net(z_in_2)
        step2 = self.net(z_in_2 + step1)
        delta_pred_2 = step1 + step2              # 2-step predicted change
        loss_2 = F.smooth_l1_loss(delta_pred_2, delta_true_2)

        return loss_1 + 0.5 * loss_2


# ======================================================================
# Main Model
# ======================================================================

class PtLatentModelV5(nn.Module):
    """Causal Encoder + AR Rollout + Decoder MFVI Correction."""

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

        self.encoder_iterator = CausalEncoderIterator(args)
        self.decoder_iterator = DecoderIterator(args, self.encoder_iterator)
        self.transition = ResidualTransition(self.dim_z)
        self.lambda_nextlat = 0.1

        cfg = PtConfig.from_dict(config.to_dict())
        cfg.hidden_size = self.dim_z
        cfg.num_attention_heads = args.n_heads
        cfg.head_dim = self.dim_z // args.n_heads
        self.rotary_emb_time = LlamaRotaryEmbedding(config=cfg)
        self.rotary_emb_channel = LlamaRotaryEmbedding(config=cfg)

        self.unary_factors = nn.Sequential(
            nn.Linear(self.patch_len, self.dim_z), nn.GELU(),
            nn.Linear(self.dim_z, self.dim_z),
        )
        self.patch_predictor = nn.Sequential(
            nn.Linear(self.dim_z, self.dim_z), nn.GELU(),
            nn.Linear(self.dim_z, self.patch_len),
        )
        self._ctx_cache = {}

    def _mfvi_ctx(self, B, N, L, device, dtype, causal=False):
        key = (B, N, L, device, dtype, causal)
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        mask_t = (_build_causal_dep_mask if causal else _build_dep_mask)(B * N, L, device, dtype)
        mask_c = _build_dep_mask(B * L, N, device, dtype)
        pid_t = torch.arange(L, device=device, dtype=torch.long)[None]
        pid_c = torch.arange(N, device=device, dtype=torch.long)[None]
        dum_t = torch.zeros(B * N, L, self.dim_z, device=device, dtype=dtype)
        dum_c = torch.zeros(B * L, N, self.dim_z, device=device, dtype=dtype)
        ctx = dict(
            dependency_mask_time=mask_t, dependency_mask_channel=mask_c,
            position_ids_time=pid_t, position_ids_channel=pid_c,
            position_embeddings_time=self.rotary_emb_time(dum_t, pid_t),
            position_embeddings_channel=self.rotary_emb_channel(dum_c, pid_c),
        )
        self._ctx_cache[key] = ctx
        return ctx

    def forward(self, time_series, is_training=True):
        device = time_series.device
        B = time_series.size(0)
        N = self.enc_in
        P = self.patch_num
        Sf = self.future_patch_num

        means = time_series.mean(1, keepdim=True).detach()
        x = time_series - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = x.transpose(1, 2).reshape(B, N, P, self.patch_len)
        unary_obs = self.unary_factors(x)
        dtype = unary_obs.dtype

        # === Phase 1: Causal Encoder MFVI ===
        enc_ctx = self._mfvi_ctx(B, N, P, device, dtype, causal=True)
        qz = unary_obs.clone()
        for _ in range(self.num_enc_iter):
            qz = self.encoder_iterator(unary_obs, qz, **enc_ctx)
        z_enc = qz  # [B, N, P, d] belief states

        # === Phase 2: NextLat loss (training only) ===
        nl_loss = None
        if is_training and P > 1:
            nl_loss = self.transition.nextlat_loss(z_enc)

        # === Phase 3: AR Rollout ===
        z_last = z_enc[:, :, -1, :]  # [B, N, d] — last belief state
        z_future_init = self.transition.rollout(z_last, Sf)  # [B, N, Sf, d]

        # === Phase 4: Decoder MFVI Correction ===
        full_ctx = self._mfvi_ctx(B, N, P + Sf, device, dtype, causal=False)
        z_future = z_future_init
        for _ in range(self.num_dec_iter):
            z_future = self.decoder_iterator(z_enc, z_future, **full_ctx)
        z_dec = z_future  # [B, N, Sf, d]

        # === Phase 5: Prediction ===
        out = self.patch_predictor(z_dec)
        dec_out = out.reshape(B, N, -1)[:, :, :self.pred_len]
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand_as(dec_out)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand_as(dec_out)

        if nl_loss is not None:
            return dec_out, nl_loss
        return dec_out


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = PtLatentModelV5(args)

    def forward(self, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                y_true=None, **kwargs):
        if self.training:
            pred, nl_loss = self.model(time_series=x_enc, is_training=True)
            return pred, nl_loss
        return self.model(time_series=x_enc, is_training=False)
