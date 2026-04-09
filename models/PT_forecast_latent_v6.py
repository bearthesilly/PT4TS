"""
PT_forecast_latent_v6: Causal Encoder MFVI + step-wise AR decoder MFVI.

Design:
  1. Encode historical patches with causal MFVI -> historical belief states.
  2. Decode autoregressively like a transformer:
     - use the latest latent to predict the next patch
     - encode that patch into a unary factor
     - infer the next latent with a single-step MFVI update
  3. During training, build a teacher-forced latent target by running the same
     single-step MFVI decoder with ground-truth future patches.

This keeps the MFVI semantics explicit:
  - historical states are fixed evidence
  - the next latent is an approximate posterior conditioned on the new patch
  - no separate transition-prior network is introduced
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

    def forward(
        self,
        qz,
        dependency_mask_channel=None,
        dependency_mask_time=None,
        position_ids_time=None,
        position_ids_channel=None,
        output_dependencies=False,
        position_embeddings_time=None,
        position_embeddings_channel=None,
    ):
        bs, num_variates, seq_len, _ = qz.size()
        qz_t = qz.view(bs * num_variates, seq_len, -1)
        mF_t, _, qz_v_t, _, bsz_t, slen_t, dt_t = self._messageF(
            qz_t,
            dependency_mask_time,
            position_embeddings_time,
            self.ternary_factor_u_time,
            self.ternary_factor_v_time,
        )
        qz_c = qz.transpose(1, 2).reshape(bs * seq_len, num_variates, -1)
        mF_c, _, qz_v_c, qz_uo_c, bsz_c, slen_c, dt_c = self._messageF(
            qz_c,
            dependency_mask_channel,
            position_embeddings_channel,
            self.ternary_factor_u_channel,
            self.ternary_factor_v_channel,
        )

        mF_t_r = mF_t.view(bs, num_variates, self.num_channels, seq_len, seq_len).permute(0, 2, 1, 3, 4)
        mF_c_r = mF_c.view(bs, seq_len, self.num_channels, num_variates, num_variates).permute(0, 2, 3, 1, 4)
        combined = F.softmax(
            torch.cat([mF_t_r, mF_c_r], dim=-1) / self.regularize_h,
            dim=-1,
            dtype=torch.float32,
        )
        qh_t_c, qh_c_c = torch.split(combined, [seq_len, num_variates], dim=-1)

        qh_t = qh_t_c.permute(0, 2, 1, 3, 4).reshape(bs * num_variates, self.num_channels, seq_len, seq_len).to(dt_t)
        qh_c = qh_c_c.permute(0, 3, 1, 2, 4).reshape(bs * seq_len, self.num_channels, num_variates, num_variates).to(dt_c)

        mG_t = self._messageG_causal(
            qh_t,
            qz_v_t,
            bsz_t,
            slen_t,
            position_embeddings_time,
            self.ternary_factor_u_time,
        ).reshape(bs, num_variates, seq_len, -1)
        mG_c = self._messageG_full(
            qh_c,
            qz_uo_c,
            qz_v_c,
            bsz_c,
            slen_c,
            position_embeddings_channel,
            self.ternary_factor_u_channel,
            self.ternary_factor_v_channel,
        ).reshape(bs, num_variates, seq_len, -1)
        return mG_t, mG_c, None, None


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


class StepwiseAutoregressiveDecoder(nn.Module):
    """
    Single-step MFVI decoder.

    Historical states are fixed evidence, represented only through cached
    time-direction value projections. The new step jointly infers:
      - time dependencies from all past steps in the same variate
      - channel dependencies among current-step variates
    """

    def __init__(self, args, encoder_iterator):
        super().__init__()
        self.config = config
        self.head_selection = encoder_iterator.head_selection
        self.topic_modeling = encoder_iterator.topic_modeling
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)
        self.dim_z = args.d_model
        self.num_channels = args.n_heads
        self.ternary_rank = self.dim_z // self.num_channels
        self.regularize_h = 1 / self.dim_z

    def _project_time_values_sequence(self, qz, pe):
        bsz, num_variates, seq_len, _ = qz.size()
        qz = qz.view(bsz * num_variates, seq_len, -1)
        qz_v = F.linear(qz, self.head_selection.ternary_factor_v_time) * config.ternary_factor_scaling
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

    def step(self, past_time_v, unary_current, time_pe_now, channel_mask, channel_pe, num_iter):
        bsz, num_variates, _ = unary_current.size()
        q_new = unary_current.clone()

        for _ in range(num_iter):
            old_q = q_new
            q_new_norm = self.norm(q_new)

            time_q_u = self._project_time_query_step(q_new_norm, time_pe_now)
            logits_time = torch.einsum("bnhr,bnhlr->bnhl", time_q_u, past_time_v).permute(0, 2, 1, 3)

            mF_c, _, qz_v_c, qz_uo_c, _, _, channel_dtype = self.head_selection._messageF(
                q_new_norm,
                channel_mask,
                channel_pe,
                self.head_selection.ternary_factor_u_channel,
                self.head_selection.ternary_factor_v_channel,
            )

            combined = F.softmax(
                torch.cat([logits_time, mF_c], dim=-1) / self.regularize_h,
                dim=-1,
                dtype=torch.float32,
            )
            qh_time, qh_channel = torch.split(combined, [past_time_v.size(3), num_variates], dim=-1)
            qh_channel = qh_channel.to(channel_dtype)

            time_msg = torch.einsum("bhnl,bnhlr->bnhr", qh_time, past_time_v)
            time_msg = self._apply_time_output_rope(time_msg, time_pe_now)
            time_msg = time_msg.reshape(bsz, num_variates, -1)
            time_msg = torch.matmul(time_msg, self.head_selection.ternary_factor_u_time) * config.ternary_factor_scaling

            channel_msg = self.head_selection._messageG_full(
                qh_channel,
                qz_uo_c,
                qz_v_c,
                bsz,
                num_variates,
                channel_pe,
                self.head_selection.ternary_factor_u_channel,
                self.head_selection.ternary_factor_v_channel,
            ).reshape(bsz, num_variates, -1)

            global_msg = self.topic_modeling(q_new_norm.unsqueeze(2)).squeeze(2)
            q_new = (time_msg + channel_msg + global_msg + unary_current) / self.config.regularize_z
            q_new = 0.5 * (q_new + old_q)

        q_new_norm = self.norm(q_new)
        new_time_v = self._project_time_value_step(q_new_norm, time_pe_now)
        return q_new, new_time_v


class PtLatentModelV6(nn.Module):
    """Causal Encoder MFVI + step-wise AR decoder MFVI."""

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
        self.lambda_latent = 0.1

        self.encoder_iterator = CausalEncoderIterator(args)
        self.decoder = StepwiseAutoregressiveDecoder(args, self.encoder_iterator)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

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
            position_ids_time=pid_t,
            position_ids_channel=pid_c,
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
        ctx = (
            mask_c,
            self.rotary_emb_channel(dummy_c, pid_c),
        )
        self._ctx_cache[key] = ctx
        return ctx

    def _time_prefix_pe(self, batch_size_times_variates, seq_len, device, dtype):
        key = ("time_prefix", batch_size_times_variates, seq_len, device, dtype)
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        pid_t = torch.arange(seq_len, device=device, dtype=torch.long)[None]
        dummy_t = torch.zeros(batch_size_times_variates, seq_len, self.dim_z, device=device, dtype=dtype)
        pe = self.rotary_emb_time(dummy_t, pid_t)
        self._ctx_cache[key] = pe
        return pe

    def _time_step_pe(self, batch_size_times_variates, position, device, dtype):
        key = ("time_step", batch_size_times_variates, position, device, dtype)
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        pid_t = torch.full((1, 1), position, device=device, dtype=torch.long)
        dummy_t = torch.zeros(batch_size_times_variates, 1, self.dim_z, device=device, dtype=dtype)
        pe = self.rotary_emb_time(dummy_t, pid_t)
        self._ctx_cache[key] = pe
        return pe

    def _split_future_patches(self, y_norm):
        bsz = y_norm.size(0)
        total_len = self.future_patch_num * self.patch_len
        if y_norm.size(1) < total_len:
            pad_len = total_len - y_norm.size(1)
            y_norm = F.pad(y_norm, (0, 0, 0, pad_len))
        return y_norm.transpose(1, 2).reshape(bsz, self.enc_in, self.future_patch_num, self.patch_len)

    def _latent_kl(self, student_logits, teacher_logits):
        teacher_prob = self.norm(teacher_logits).detach()
        student_prob = self.norm(student_logits)
        teacher_log = torch.log(teacher_prob.clamp_min(1e-8))
        student_log = torch.log(student_prob.clamp_min(1e-8))
        return (teacher_prob * (teacher_log - student_log)).sum(dim=-1).mean()

    def forward(self, time_series, y_true=None, is_training=True):
        device = time_series.device
        batch_size = time_series.size(0)
        num_variates = self.enc_in
        history_len = self.patch_num
        future_len = self.future_patch_num

        means = time_series.mean(1, keepdim=True).detach()
        x = time_series - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = x.transpose(1, 2).reshape(batch_size, num_variates, history_len, self.patch_len)
        unary_obs = self.unary_factors(x)
        dtype = unary_obs.dtype

        enc_ctx = self._causal_enc_ctx(batch_size, num_variates, history_len, device, dtype)
        z_hist = unary_obs.clone()
        for _ in range(self.num_enc_iter):
            z_hist = self.encoder_iterator(unary_obs, z_hist, **enc_ctx)

        hist_norm = self.norm(z_hist)
        hist_time_pe = self._time_prefix_pe(batch_size * num_variates, history_len, device, dtype)
        hist_time_v = self.decoder._project_time_values_sequence(hist_norm, hist_time_pe)
        channel_mask, channel_pe = self._channel_step_ctx(batch_size, device, dtype)

        student_cache = hist_time_v
        z_prev_student = z_hist[:, :, -1, :]
        pred_patches = []

        latent_loss = None
        teacher_cache = None
        future_gt = None
        if is_training and y_true is not None:
            y_norm = (y_true - means) / stdev
            future_gt = self._split_future_patches(y_norm)
            teacher_cache = hist_time_v.detach()
            latent_loss = unary_obs.new_zeros(())

        for step in range(future_len):
            patch_pred = self.patch_predictor(z_prev_student)
            pred_patches.append(patch_pred)
            unary_student = self.unary_factors(patch_pred)
            time_pe_now = self._time_step_pe(batch_size * num_variates, history_len + step, device, dtype)
            z_student, new_student_v = self.decoder.step(
                past_time_v=student_cache,
                unary_current=unary_student,
                time_pe_now=time_pe_now,
                channel_mask=channel_mask,
                channel_pe=channel_pe,
                num_iter=self.num_dec_iter,
            )
            student_cache = torch.cat([student_cache, new_student_v], dim=3)
            z_prev_student = z_student

            if future_gt is not None:
                with torch.no_grad():
                    unary_teacher = self.unary_factors(future_gt[:, :, step, :])
                    z_teacher, new_teacher_v = self.decoder.step(
                        past_time_v=teacher_cache,
                        unary_current=unary_teacher,
                        time_pe_now=time_pe_now,
                        channel_mask=channel_mask,
                        channel_pe=channel_pe,
                        num_iter=self.num_dec_iter,
                    )
                    teacher_cache = torch.cat([teacher_cache, new_teacher_v], dim=3)
                latent_loss = latent_loss + self._latent_kl(z_student, z_teacher)

        pred_patches = torch.stack(pred_patches, dim=2)
        dec_out = pred_patches.reshape(batch_size, num_variates, -1)[:, :, :self.pred_len]
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand_as(dec_out)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand_as(dec_out)

        if latent_loss is not None:
            latent_loss = self.lambda_latent * (latent_loss / future_len)
            return dec_out, latent_loss
        return dec_out


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = PtLatentModelV6(args)

    def forward(self, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None, y_true=None, **kwargs):
        if self.training:
            pred, aux_loss = self.model(time_series=x_enc, y_true=y_true, is_training=True)
            return pred, aux_loss
        return self.model(time_series=x_enc, y_true=None, is_training=False)
