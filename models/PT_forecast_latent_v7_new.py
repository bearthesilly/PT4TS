"""
PT_forecast_latent_v7: Strict AR with CRF(MFVI) + AutoTimes-style training.

Design philosophy — two-phase paradigm:

  Training (teacher-forced, like AutoTimes / GPT):
    Concatenate history + GT-future patches into one sequence, run causal
    MFVI over the whole thing in a single parallel pass.  At each position t
    the latent z_t predicts patch_{t+1} (next-patch prediction).  This gives
    CRF-quality latent representations as "free" supervision — no separate
    teacher path needed.  A transition-prior auxiliary loss ensures the
    inference-only transition network is also trained.

  Inference (strict AR rollout):
    Generate one patch at a time: z_t → predict patch → re-encode as unary
    + transition prior → single-step MFVI → z_{t+1} → repeat.
    The predicted patch IS fed back (strict AR).

Key components:
  - CausalEncoderIterator:  causal MFVI (time causal + channel bidirectional)
  - StepwiseARDecoder:      single-step MFVI for inference-time AR rollout
  - TransitionPrior:        d_model → d_model MLP, trained via aux loss,
                            provides latent-space shortcut during AR inference
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
    PtTopicModeling,
    RopeApplier,
)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _build_dep_mask(batch_dim, seq_len, device, dtype):
    """Bidirectional mask with self-loop removed (for channel direction)."""
    mask2d = torch.ones(batch_dim, seq_len, device=device)
    converter = AttentionMaskConverter(is_causal=False)
    mask4d = converter.to_4d(mask2d, seq_len, dtype=dtype)
    diag = torch.eye(seq_len, dtype=mask4d.dtype, device=mask4d.device)[None, None]
    return mask4d.masked_fill(diag.bool(), torch.finfo(dtype).min)


def _build_causal_dep_mask(batch_dim, seq_len, device, dtype):
    """Causal mask: each position attends to strictly earlier positions."""
    min_val = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype), diagonal=0
    )
    return causal[None, None].expand(batch_dim, 1, seq_len, seq_len)


# ---------------------------------------------------------------------------
# HeadSelection — shared by encoder and decoder
# ---------------------------------------------------------------------------

class HeadSelection(nn.Module):
    """Ternary-factor head selection (time + channel), following v15 style."""

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
        for p in [
            self.ternary_factor_u_time,
            self.ternary_factor_v_time,
            self.ternary_factor_u_channel,
            self.ternary_factor_v_channel,
        ]:
            nn.init.normal_(p, mean=0.0, std=config.ternary_initializer_range)

    def _messageF(self, qz, mask, pe, u, v):
        bsz, seq_len, _ = qz.size()
        qz_u = F.linear(qz, u) * config.ternary_factor_scaling
        qz_v = F.linear(qz, v) * config.ternary_factor_scaling
        qz_u = qz_u.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        qz_v = qz_v.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        rope = RopeApplier(*pe)
        qz_uo = rope.apply_o(qz_u)
        qz_u = rope.apply(qz_u)
        qz_v = rope.apply(qz_v)
        mF = torch.matmul(qz_u, qz_v.transpose(2, 3))
        if mask is not None:
            mF = mF + mask
        return mF, qz_u, qz_v, qz_uo, bsz, seq_len, qz_u.dtype

    def _messageG_causal(self, qh, qz_v, bsz, seq_len, pe, u):
        rope = RopeApplier(*pe)
        v1 = rope.apply_o(torch.matmul(qh, qz_v))
        v1 = v1.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        return torch.matmul(v1, u) * config.ternary_factor_scaling

    def _messageG_full(self, qh, qz_uo, qz_v, bsz, seq_len, pe, u, v):
        rope = RopeApplier(*pe)
        v1 = rope.apply_o(torch.matmul(qh, qz_v))
        v2 = rope.apply(torch.matmul(qh.transpose(2, 3), qz_uo))
        v1 = v1.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        v2 = v2.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        return (torch.matmul(v1, u) + torch.matmul(v2, v)) * config.ternary_factor_scaling

    def forward(self, qz, dependency_mask_channel=None, dependency_mask_time=None,
                position_embeddings_time=None, position_embeddings_channel=None, **_kw):
        bs, num_variates, seq_len, _ = qz.size()

        qz_t = qz.view(bs * num_variates, seq_len, -1)
        mF_t, _, qz_v_t, _, bsz_t, slen_t, dt_t = self._messageF(
            qz_t, dependency_mask_time, position_embeddings_time,
            self.ternary_factor_u_time, self.ternary_factor_v_time,
        )

        qz_c = qz.transpose(1, 2).reshape(bs * seq_len, num_variates, -1)
        mF_c, _, qz_v_c, qz_uo_c, bsz_c, slen_c, dt_c = self._messageF(
            qz_c, dependency_mask_channel, position_embeddings_channel,
            self.ternary_factor_u_channel, self.ternary_factor_v_channel,
        )

        mF_t_r = mF_t.view(bs, num_variates, self.num_channels, seq_len, seq_len).permute(0, 2, 1, 3, 4)
        mF_c_r = mF_c.view(bs, seq_len, self.num_channels, num_variates, num_variates).permute(0, 2, 3, 1, 4)
        combined = F.softmax(
            torch.cat([mF_t_r, mF_c_r], dim=-1) / self.regularize_h,
            dim=-1, dtype=torch.float32,
        )
        qh_t_c, qh_c_c = torch.split(combined, [seq_len, num_variates], dim=-1)

        qh_t = qh_t_c.permute(0, 2, 1, 3, 4).reshape(
            bs * num_variates, self.num_channels, seq_len, seq_len
        ).to(dt_t)
        qh_c = qh_c_c.permute(0, 3, 1, 2, 4).reshape(
            bs * seq_len, self.num_channels, num_variates, num_variates
        ).to(dt_c)

        mG_t = self._messageG_causal(
            qh_t, qz_v_t, bsz_t, slen_t,
            position_embeddings_time, self.ternary_factor_u_time,
        ).reshape(bs, num_variates, seq_len, -1)
        mG_c = self._messageG_full(
            qh_c, qz_uo_c, qz_v_c, bsz_c, slen_c,
            position_embeddings_channel,
            self.ternary_factor_u_channel, self.ternary_factor_v_channel,
        ).reshape(bs, num_variates, seq_len, -1)
        return mG_t, mG_c


# ---------------------------------------------------------------------------
# Encoder iterator (causal MFVI)
# ---------------------------------------------------------------------------

class CausalEncoderIterator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = config
        self.head_selection = HeadSelection(args)
        self.topic_modeling = PtTopicModeling(args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

    def forward(self, unary, qz, **ctx):
        old = qz
        qz = self.norm(qz)
        m_t, m_c = self.head_selection(qz=qz, **ctx)
        m_g = self.topic_modeling(qz)
        qz = (m_t + m_c + m_g + unary) / self.config.regularize_z
        return (qz + old) * 0.5


# ---------------------------------------------------------------------------
# Step-wise AR decoder (inference only, strict AR with dual evidence)
# ---------------------------------------------------------------------------

class StepwiseARDecoder(nn.Module):
    """
    Single-step MFVI for inference-time AR rollout.

    Shares ALL learned factors with the encoder (ternary + topic_modeling),
    so every parameter is trained during teacher-forced training.
    Only the learnable damping scalar is decoder-specific.
    """

    def __init__(self, args, encoder_iterator: CausalEncoderIterator):
        super().__init__()
        self.config = config
        self.head_selection = encoder_iterator.head_selection   # shared
        self.topic_modeling = encoder_iterator.topic_modeling    # shared
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)
        self.dim_z = args.d_model
        self.num_channels = args.n_heads
        self.ternary_rank = self.dim_z // self.num_channels
        self.regularize_h = 1 / self.dim_z

        self.damping_logit = nn.Parameter(torch.tensor(0.0))

    # --- time-direction V cache projections ---

    def project_time_values_seq(self, qz, pe):
        bsz, num_variates, seq_len, _ = qz.size()
        qz_flat = qz.view(bsz * num_variates, seq_len, -1)
        qz_v = F.linear(qz_flat, self.head_selection.ternary_factor_v_time) * config.ternary_factor_scaling
        qz_v = qz_v.view(bsz * num_variates, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        qz_v = RopeApplier(*pe).apply(qz_v)
        return qz_v.view(bsz, num_variates, self.num_channels, seq_len, self.ternary_rank)

    def _project_time_value_step(self, qz_step, pe):
        bsz, num_variates, _ = qz_step.size()
        qz = qz_step.reshape(bsz * num_variates, 1, -1)
        qz_v = F.linear(qz, self.head_selection.ternary_factor_v_time) * config.ternary_factor_scaling
        qz_v = qz_v.view(bsz * num_variates, 1, self.num_channels, self.ternary_rank).transpose(1, 2)
        qz_v = RopeApplier(*pe).apply(qz_v)
        return qz_v.view(bsz, num_variates, self.num_channels, 1, self.ternary_rank)

    def _project_time_query_step(self, qz_step, pe):
        bsz, num_variates, _ = qz_step.size()
        qz = qz_step.reshape(bsz * num_variates, 1, -1)
        qz_u = F.linear(qz, self.head_selection.ternary_factor_u_time) * config.ternary_factor_scaling
        qz_u = qz_u.view(bsz * num_variates, 1, self.num_channels, self.ternary_rank).transpose(1, 2)
        qz_u = RopeApplier(*pe).apply(qz_u)
        return qz_u.view(bsz, num_variates, self.num_channels, self.ternary_rank)

    def _apply_time_output_rope(self, value, pe):
        bsz, num_variates, _, _ = value.size()
        value = value.view(bsz * num_variates, self.num_channels, 1, self.ternary_rank)
        value = RopeApplier(*pe).apply_o(value)
        return value.view(bsz, num_variates, self.num_channels, self.ternary_rank)

    def step(self, past_time_v, patch_unary, trans_prior, time_pe_now,
             channel_mask, channel_pe, num_iter):
        """
        Single-step MFVI with dual evidence.

        Args:
            past_time_v:  [B, N, heads, L_past, rank]
            patch_unary:  [B, N, d_model]  (from predicted patch, strict AR)
            trans_prior:  [B, N, d_model]  (latent-space transition)
            time_pe_now:  RoPE at current position
            channel_mask: dep mask for channel
            channel_pe:   RoPE for channel
            num_iter:     MFVI iterations
        """
        bsz, num_variates, _ = patch_unary.size()
        alpha = torch.sigmoid(self.damping_logit)
        evidence = patch_unary + trans_prior
        q_new = evidence.clone()

        for _ in range(num_iter):
            old_q = q_new
            q_new_norm = self.norm(q_new)

            # time message
            time_q_u = self._project_time_query_step(q_new_norm, time_pe_now)
            logits_time = torch.einsum("bnhr,bnhlr->bnhl", time_q_u, past_time_v)
            logits_time = logits_time.permute(0, 2, 1, 3)

            # channel message
            mF_c, _, qz_v_c, qz_uo_c, _, _, channel_dtype = self.head_selection._messageF(
                q_new_norm, channel_mask, channel_pe,
                self.head_selection.ternary_factor_u_channel,
                self.head_selection.ternary_factor_v_channel,
            )

            # joint softmax
            combined = F.softmax(
                torch.cat([logits_time, mF_c], dim=-1) / self.regularize_h,
                dim=-1, dtype=torch.float32,
            )
            qh_time, qh_channel = torch.split(
                combined, [past_time_v.size(3), num_variates], dim=-1
            )
            qh_channel = qh_channel.to(channel_dtype)

            # time G
            time_msg = torch.einsum("bhnl,bnhlr->bnhr", qh_time, past_time_v)
            time_msg = self._apply_time_output_rope(time_msg, time_pe_now)
            time_msg = time_msg.reshape(bsz, num_variates, -1)
            time_msg = torch.matmul(
                time_msg, self.head_selection.ternary_factor_u_time
            ) * config.ternary_factor_scaling

            # channel G
            channel_msg = self.head_selection._messageG_full(
                qh_channel, qz_uo_c, qz_v_c, bsz, num_variates, channel_pe,
                self.head_selection.ternary_factor_u_channel,
                self.head_selection.ternary_factor_v_channel,
            ).reshape(bsz, num_variates, -1)

            # topic
            global_msg = self.topic_modeling(q_new_norm.unsqueeze(2)).squeeze(2)

            # update
            q_new = (evidence + time_msg + channel_msg + global_msg) / self.config.regularize_z
            q_new = alpha * q_new + (1 - alpha) * old_q

        q_new_norm = self.norm(q_new)
        new_time_v = self._project_time_value_step(q_new_norm, time_pe_now)
        return q_new, new_time_v


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PtLatentModelV7(nn.Module):
    """
    Training:  teacher-forced causal MFVI (AutoTimes paradigm)
    Inference: strict AR rollout with dual evidence
    """

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
        self.lambda_trans = 0.1

        # encoder
        self.encoder_iterator = CausalEncoderIterator(args)

        # decoder (shares all learned factors with encoder)
        self.decoder = StepwiseARDecoder(args, self.encoder_iterator)

        # transition prior: z_t -> prior for z_{t+1} in latent space
        self.transition_prior = nn.Sequential(
            nn.Linear(self.dim_z, self.dim_z),
            nn.GELU(),
            nn.Linear(self.dim_z, self.dim_z),
        )

        # patch I/O
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

        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

        cfg = PtConfig.from_dict(config.to_dict())
        cfg.hidden_size = self.dim_z
        cfg.num_attention_heads = args.n_heads
        cfg.head_dim = self.dim_z // args.n_heads
        self.rotary_emb_time = LlamaRotaryEmbedding(config=cfg)
        self.rotary_emb_channel = LlamaRotaryEmbedding(config=cfg)

        self._ctx_cache = {}

    # --- context helpers ---

    def _causal_enc_ctx(self, batch_size, num_variates, seq_len, device, dtype):
        key = ("enc", batch_size, num_variates, seq_len, device, dtype)
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        mask_t = _build_causal_dep_mask(batch_size * num_variates, seq_len, device, dtype)
        mask_c = _build_dep_mask(batch_size * seq_len, num_variates, device, dtype)
        pid_t = torch.arange(seq_len, device=device, dtype=torch.long)[None]
        pid_c = torch.arange(num_variates, device=device, dtype=torch.long)[None]
        dummy_t = torch.zeros(batch_size * num_variates, seq_len, self.dim_z, device=device, dtype=dtype)
        dummy_c = torch.zeros(batch_size * seq_len, num_variates, self.dim_z, device=device, dtype=dtype)
        ctx = dict(
            dependency_mask_time=mask_t,
            dependency_mask_channel=mask_c,
            position_embeddings_time=self.rotary_emb_time(dummy_t, pid_t),
            position_embeddings_channel=self.rotary_emb_channel(dummy_c, pid_c),
        )
        self._ctx_cache[key] = ctx
        return ctx

    def _channel_step_ctx(self, batch_size, device, dtype):
        key = ("channel", batch_size, self.enc_in, device, dtype)
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        mask_c = _build_dep_mask(batch_size, self.enc_in, device, dtype)
        pid_c = torch.arange(self.enc_in, device=device, dtype=torch.long)[None]
        dummy_c = torch.zeros(batch_size, self.enc_in, self.dim_z, device=device, dtype=dtype)
        ctx = (mask_c, self.rotary_emb_channel(dummy_c, pid_c))
        self._ctx_cache[key] = ctx
        return ctx

    def _time_pe(self, bsz_x_nvar, seq_len, device, dtype):
        key = ("time_pe", bsz_x_nvar, seq_len, device, dtype)
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        pid = torch.arange(seq_len, device=device, dtype=torch.long)[None]
        dummy = torch.zeros(bsz_x_nvar, seq_len, self.dim_z, device=device, dtype=dtype)
        pe = self.rotary_emb_time(dummy, pid)
        self._ctx_cache[key] = pe
        return pe

    def _time_step_pe(self, bsz_x_nvar, position, device, dtype):
        key = ("time_step", bsz_x_nvar, position, device, dtype)
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        pid = torch.full((1, 1), position, device=device, dtype=torch.long)
        dummy = torch.zeros(bsz_x_nvar, 1, self.dim_z, device=device, dtype=dtype)
        pe = self.rotary_emb_time(dummy, pid)
        self._ctx_cache[key] = pe
        return pe

    def _split_future_patches(self, y_norm):
        bsz = y_norm.size(0)
        total_len = self.future_patch_num * self.patch_len
        if y_norm.size(1) < total_len:
            pad_len = total_len - y_norm.size(1)
            y_norm = F.pad(y_norm, (0, 0, 0, pad_len))
        return y_norm.transpose(1, 2).reshape(bsz, self.enc_in, self.future_patch_num, self.patch_len)

    # =====================================================================
    # Training: teacher-forced next-patch prediction (AutoTimes paradigm)
    # =====================================================================

    def _forward_train(self, x, means, stdev, x_patches, y_true):
        device = x.device
        batch_size = x.size(0)
        num_variates = self.enc_in
        history_len = self.patch_num
        future_len = self.future_patch_num

        # prepare future GT patches
        y_norm = (y_true - means) / stdev
        future_gt = self._split_future_patches(y_norm)

        # full sequence: [B, N, P+F, patch_len]
        all_patches = torch.cat([x_patches, future_gt], dim=2)
        total_len = all_patches.size(2)
        dtype = all_patches.dtype

        # unary encode full sequence
        unary_all = self.unary_factors(all_patches)

        # causal MFVI over full sequence (teacher-forced)
        ctx = self._causal_enc_ctx(batch_size, num_variates, total_len, device, dtype)
        z_all = unary_all.clone()
        for _ in range(self.num_enc_iter):
            z_all = self.encoder_iterator(unary_all, z_all, **ctx)

        # next-patch prediction: z[t] predicts patch[t+1]
        # for future output: z at positions [P-1 .. P+F-2] predicts patches [P .. P+F-1]
        future_z = z_all[:, :, history_len - 1 : history_len + future_len - 1, :]
        pred_future = self.patch_predictor(future_z)  # [B, N, F, patch_len]

        # assemble denormalized output
        dec_out = pred_future.reshape(batch_size, num_variates, -1)[:, :, :self.pred_len]
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand_as(dec_out)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand_as(dec_out)

        # --- auxiliary loss: train transition prior ---
        # transition_prior(z[t]) should approximate z[t+1]
        z_src = z_all[:, :, :-1, :].detach()   # [B, N, T-1, d]
        z_tgt = z_all[:, :, 1:, :].detach()     # [B, N, T-1, d]
        trans_pred = self.transition_prior(z_src)
        aux_loss = self.lambda_trans * F.mse_loss(trans_pred, z_tgt)

        return dec_out, aux_loss

    # =====================================================================
    # Inference: strict AR rollout with dual evidence
    # =====================================================================

    def _forward_inference(self, x, means, stdev, x_patches):
        device = x.device
        batch_size = x.size(0)
        num_variates = self.enc_in
        history_len = self.patch_num
        future_len = self.future_patch_num
        dtype = x_patches.dtype

        # history encoding
        unary_hist = self.unary_factors(x_patches)
        enc_ctx = self._causal_enc_ctx(batch_size, num_variates, history_len, device, dtype)
        z_hist = unary_hist.clone()
        for _ in range(self.num_enc_iter):
            z_hist = self.encoder_iterator(unary_hist, z_hist, **enc_ctx)

        # prepare cache
        hist_norm = self.norm(z_hist)
        hist_time_pe = self._time_pe(batch_size * num_variates, history_len, device, dtype)
        cache_v = self.decoder.project_time_values_seq(hist_norm, hist_time_pe)
        channel_mask, channel_pe = self._channel_step_ctx(batch_size, device, dtype)

        # AR rollout
        z_prev = z_hist[:, :, -1, :]
        pred_patches = []

        for step in range(future_len):
            # strict AR: predict patch, feed back
            patch_pred = self.patch_predictor(z_prev)
            pred_patches.append(patch_pred)

            # dual evidence
            patch_unary = self.unary_factors(patch_pred)
            trans_unary = self.transition_prior(z_prev)

            # single-step MFVI
            time_pe_now = self._time_step_pe(
                batch_size * num_variates, history_len + step, device, dtype
            )
            z_new, new_v = self.decoder.step(
                past_time_v=cache_v,
                patch_unary=patch_unary,
                trans_prior=trans_unary,
                time_pe_now=time_pe_now,
                channel_mask=channel_mask,
                channel_pe=channel_pe,
                num_iter=self.num_dec_iter,
            )
            cache_v = torch.cat([cache_v, new_v], dim=3)
            z_prev = z_new

        # assemble
        pred_patches = torch.stack(pred_patches, dim=2)
        dec_out = pred_patches.reshape(batch_size, num_variates, -1)[:, :, :self.pred_len]
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand_as(dec_out)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand_as(dec_out)
        return dec_out

    # =====================================================================
    # Entry point
    # =====================================================================

    def forward(self, time_series, y_true=None, is_training=True):
        # instance normalization
        means = time_series.mean(1, keepdim=True).detach()
        x = time_series - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        batch_size = time_series.size(0)
        x_patches = x.transpose(1, 2).reshape(
            batch_size, self.enc_in, self.patch_num, self.patch_len
        )

        if is_training and y_true is not None:
            return self._forward_train(x, means, stdev, x_patches, y_true)
        else:
            return self._forward_inference(x, means, stdev, x_patches)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = PtLatentModelV7(args)

    def forward(self, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                y_true=None, **kwargs):
        if self.training:
            pred, aux_loss = self.model(time_series=x_enc, y_true=y_true, is_training=True)
            return pred, aux_loss
        return self.model(time_series=x_enc, y_true=None, is_training=False)
