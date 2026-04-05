'''
ST-PT + Source Separation Prior (Known Mixing Matrix)

Based on PT_forecast_v15 (backbone).

Prior: Given a known mixing matrix A (C x K), we introduce source latent
variables s_{k,t} and binary factors psi_mix(z_{c,t}, s_{k,t}) = exp(A_{c,k} / d_z * z^T s).

MFVI yields:
  Step 1 (unmix):  qs = normalize(A^T @ qz)       -- aggregate obs -> source space
  Step 2 (re-mix): m_source = lambda * A @ qs      -- project source belief back

This is structurally analogous to PtTopicModeling but operates on the
CHANNEL axis with a FIXED (non-learnable) potential table A.

Graph primitive:  New Latent Layer (source variables)
Injection point:  m_source added to qz update in PtEncoderIterator
'''
import math
import warnings
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Union
from layers.Embed import PTDataEmbedding
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .configuration_pt import PtConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,rotate_half
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, MaskedLMOutput
from transformers.utils import (
    logging,
    ModelOutput
)

from transformers.modeling_attn_mask_utils import AttentionMaskConverter


logger = logging.get_logger(__name__)
from typing import List, Optional

@dataclass
class Config:
    potential_func_z: str = "square"
    potential_func_g: str = "abs"
    binary_initializer_range: float = 0.2
    ternary_initializer_range: float = 0.2
    binary_factor_scaling: float = 1.0
    ternary_factor_scaling: float = 1.0
    potential_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout_prob_z: float = 0.0
    dropout_prob_h: float = 0.0
    regularize_z: float = 1
    regularize_g: float = 1.0

    def to_dict(self):
        return asdict(self)

config = Config()

class RopeApplier:
    def __init__(self, cos, sin, position_ids=None, unsqueeze_dim=1) -> None:
        self.cos = cos.unsqueeze(unsqueeze_dim)
        self.sin = sin.unsqueeze(unsqueeze_dim)

    def apply(self, qkv):
        return (qkv * self.cos) + (rotate_half(qkv) * self.sin)

    def apply_o(self, o):
        return (o * self.cos) - (rotate_half(o) * self.sin)


class SquaredSoftmax(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states.pow(2)
        hidden_states = F.normalize(hidden_states, p=1, dim=self.dim, eps=self.eps)
        return hidden_states.to(input_dtype)


class AbsNormalization(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = F.relu(hidden_states)
        hidden_states = F.normalize(hidden_states, p=1, dim=self.dim, eps=self.eps)
        return hidden_states.to(input_dtype)


class Softmax(nn.Softmax):
    def __init__(self, dim=-1, eps=None):
        super().__init__(dim=dim)


POTENTIAL2ACT = {
    "exp": Softmax,
    "abs": AbsNormalization,
    "square": SquaredSoftmax,
}


# =====================================================================
# Known Mixing Matrix A  (must match data generation!)
# =====================================================================
import numpy as np
_MIXING_MATRIX_NP = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.7, 0.7, 0.0],
    [0.0, 0.6, 0.8],
    [0.8, 0.0, 0.6],
    [0.5, 0.5, 0.5],
    [0.9, 0.3, 0.1],
    [0.1, 0.3, 0.9],
], dtype=np.float32)  # [C=9, K=3]


class PtSourceSeparation(nn.Module):
    """Source Separation prior via known mixing matrix.

    Implements the MFVI update for source latent variables:
        qs = normalize(A^T @ qz)        (unmix: obs -> source space)
        m_source = lambda * A @ qs       (re-mix: source -> obs space)

    Structurally analogous to PtTopicModeling, but:
    - Operates on the CHANNEL axis (not feature axis)
    - Uses a FIXED matrix A (not learnable)
    """
    def __init__(self, args):
        super().__init__()
        self.enc_in = args.enc_in   # C (number of observed channels)
        self.dim_z = args.d_model

        # Register A and A^T as non-learnable buffers
        A = torch.from_numpy(_MIXING_MATRIX_NP)  # [C, K]
        assert A.shape[0] == self.enc_in, \
            f"Mixing matrix rows ({A.shape[0]}) must match enc_in ({self.enc_in})"
        self.num_sources = A.shape[1]  # K
        self.register_buffer('A', A)                # [C, K]
        self.register_buffer('A_T', A.T.contiguous())  # [K, C]

        # Normalization for source beliefs (same as topic modeling)
        self.act = POTENTIAL2ACT[config.potential_func_g](dim=-1, eps=config.potential_eps)
        self.regularize_s = 1.0

        # Strength of the source prior message
        self.source_prior_scaling = 200.0

    def forward(self, qz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            qz: normalized beliefs [bs, C, P, dim_z]
        Returns:
            m_source: source separation message [bs, C, P, dim_z]
        """
        # Step 1: Unmix  (Z -> S)
        # qs_{k,t} = A^T_{k,:} @ qz_{:,t}  for each (k, t)
        # qz: [bs, C, P, dim_z],  A_T: [K, C]
        # einsum: 'kc, bcpd -> bkpd'
        qs = torch.einsum('kc, bcpd -> bkpd', self.A_T, qz)  # [bs, K, P, dim_z]
        qs = self.act(qs / self.regularize_s)

        # Step 2: Re-mix  (S -> Z)
        # m_source_{c,t} = A_{c,:} @ qs_{:,t}
        # einsum: 'ck, bkpd -> bcpd'
        m_source = torch.einsum('ck, bkpd -> bcpd', self.A, qs)  # [bs, C, P, dim_z]

        return m_source * self.source_prior_scaling


class PtHeadSelection(nn.Module):
    """Multi-channel head selection from 'Probabilistic Transformer' paper"""

    def __init__(self, args):
        super().__init__()
        self.config = config
        self.dim_z = args.d_model
        self.enc_in = args.enc_in
        self.num_channels = args.n_heads
        self.ternary_rank = self.dim_z // self.num_channels
        self.rope_theta = config.rope_theta
        self.regularize_h = 1/self.dim_z
        self.ternary_factor_u_time = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_v_time = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_u_channel = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_v_channel = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.dropout = nn.Dropout(config.dropout_prob_h)
        self._init_ternary()

    def _init_ternary(self):
        nn.init.normal_(self.ternary_factor_u_channel, mean=0.0, std=self.config.ternary_initializer_range)
        nn.init.normal_(self.ternary_factor_v_channel, mean=0.0, std=self.config.ternary_initializer_range)
        nn.init.normal_(self.ternary_factor_u_time, mean=0.0, std=self.config.ternary_initializer_range)
        nn.init.normal_(self.ternary_factor_v_time, mean=0.0, std=self.config.ternary_initializer_range)

    def calculate_messageF(self, qz, dependency_mask, position_ids, position_embeddings, ternary_factor_u, ternary_factor_v):
        bsz, seq_len, _ = qz.size()
        qz_u = nn.functional.linear(qz, ternary_factor_u) * self.config.ternary_factor_scaling
        qz_v = nn.functional.linear(qz, ternary_factor_v) * self.config.ternary_factor_scaling
        qz_u = qz_u.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        qz_v = qz_v.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        cos, sin = position_embeddings
        rope_applier = RopeApplier(cos, sin, position_ids)
        qz_uo = rope_applier.apply_o(qz_u)
        qz_u = rope_applier.apply(qz_u)
        qz_v = rope_applier.apply(qz_v)
        message_F = torch.matmul(qz_u, qz_v.transpose(2, 3))
        if message_F.size() != (bsz, self.num_channels, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_channels, seq_len, seq_len)}, but is"
                f" {message_F.size()}"
            )
        if dependency_mask is not None:
            if dependency_mask.size() != (bsz, 1, seq_len, seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, seq_len)}, but is {dependency_mask.size()}"
                )
            message_F = message_F + dependency_mask
        return message_F, qz_u, qz_v, qz_uo, bsz, seq_len, qz_u.dtype

    def calculate_messageG(self, qh, qz_uo, qz_v, bsz, seq_len, position_ids, position_embeddings, ternary_factor_u, ternary_factor_v):
        cos, sin = position_embeddings
        rope_applier = RopeApplier(cos, sin, position_ids)
        qh_v1 = torch.matmul(qh, qz_v)
        qh_v2 = torch.matmul(qh.transpose(2, 3), qz_uo)
        qh_v1 = rope_applier.apply_o(qh_v1)
        qh_v2 = rope_applier.apply(qh_v2)
        if qh_v1.size() != (bsz, self.num_channels, seq_len, self.ternary_rank):
            raise ValueError(
                f"`qh_v1` should be of size {(bsz, self.num_channels, seq_len, self.ternary_rank)}, but is"
                f" {qh_v1.size()}"
            )
        if qh_v2.size() != (bsz, self.num_channels, seq_len, self.ternary_rank):
            raise ValueError(
                f"`qh_v2` should be of size {(bsz, self.num_channels, seq_len, self.ternary_rank)}, but is"
                f" {qh_v2.size()}"
            )
        qh_v1 = qh_v1.transpose(1, 2).contiguous()
        qh_v2 = qh_v2.transpose(1, 2).contiguous()
        qh_v1 = qh_v1.reshape(bsz, seq_len, self.num_channels * self.ternary_rank)
        qh_v2 = qh_v2.reshape(bsz, seq_len, self.num_channels * self.ternary_rank)
        message_G = (torch.matmul(qh_v1, ternary_factor_u) + torch.matmul(qh_v2, ternary_factor_v)) * self.config.ternary_factor_scaling
        return message_G

    def forward(
        self,
        qz: torch.Tensor,
        dependency_mask_channel: Optional[torch.Tensor] = None,
        dependency_mask_time: Optional[torch.Tensor] = None,
        position_ids_time: Optional[torch.LongTensor] = None,
        position_ids_channel: Optional[torch.LongTensor] = None,
        output_dependencies: bool = False,
        position_embeddings_time: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_channel: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bs, num_channel, length, _ = qz.size()
        qz = qz.view(bs*num_channel, length, -1)
        message_F_time, qz_u_time, qz_v_time, qz_uo_time, bsz_time, seq_len_time, type_time= self.calculate_messageF(
            qz,
            dependency_mask=dependency_mask_time,
            position_ids=position_ids_time,
            position_embeddings=position_embeddings_time,
            ternary_factor_u=self.ternary_factor_u_time,
            ternary_factor_v=self.ternary_factor_v_time
        )
        qz = qz.view(bs, num_channel, length, -1)
        qz = qz.transpose(1,2).reshape(bs*length, num_channel, -1)
        message_F_channel, qz_u_channel, qz_v_channel, qz_uo_channel, bsz_channel, seq_len_channel, type_channel = self.calculate_messageF(
            qz,
            dependency_mask=dependency_mask_channel,
            position_ids=position_ids_channel,
            position_embeddings=position_embeddings_channel,
            ternary_factor_u=self.ternary_factor_u_channel,
            ternary_factor_v=self.ternary_factor_v_channel
        )
        qz =qz.view(bs,length, num_channel,-1).transpose(1,2)
        mF_time_reshaped = message_F_time.view(bs, num_channel, self.num_channels, length, length).permute(0, 2, 1, 3, 4)
        mF_channel_reshaped = message_F_channel.view(bs, length, self.num_channels, num_channel, num_channel).permute(0, 2, 3, 1, 4)
        combined_qh_logits = torch.cat([mF_time_reshaped, mF_channel_reshaped], dim=-1)
        combined_qh = nn.functional.softmax(combined_qh_logits / self.regularize_h, dim=-1, dtype=torch.float32)
        qh_time_combined, qh_channel_combined = torch.split(combined_qh, [length, num_channel], dim=-1)
        qh_time_output = qh_time_combined.permute(0, 2, 1, 3, 4)
        qh_time = qh_time_output.reshape(bs * num_channel, self.num_channels, length, length).to(type_time)
        qh_channel_output = qh_channel_combined.permute(0, 3, 1, 2, 4)
        qh_channel = qh_channel_output.reshape(bs * length, self.num_channels, num_channel, num_channel).to(type_channel)

        message_G_channel = self.calculate_messageG(
            qh=qh_channel, qz_uo=qz_uo_channel, qz_v=qz_v_channel,
            bsz=bsz_channel, seq_len=seq_len_channel,
            position_ids=position_ids_channel, position_embeddings=position_embeddings_channel,
            ternary_factor_u=self.ternary_factor_u_channel, ternary_factor_v=self.ternary_factor_v_channel
        ).reshape(bs, num_channel, length, -1)
        message_G_time = self.calculate_messageG(
            qh=qh_time, qz_uo=qz_uo_time, qz_v=qz_v_time,
            bsz=bsz_time, seq_len=seq_len_time,
            position_ids=position_ids_time, position_embeddings=position_embeddings_time,
            ternary_factor_u=self.ternary_factor_u_time, ternary_factor_v=self.ternary_factor_v_time
        ).reshape(bs, num_channel, length, -1)
        return message_G_time, message_G_channel, qh_time_output, qh_channel_output


class PtTopicModeling(nn.Module):
    """Topic modeling w/ global nodes."""
    def __init__(self, args):
        super().__init__()
        self.config = config
        self.dim_z = args.d_model
        self.dim_g = args.d_ff
        self.binary_factor = nn.Parameter(torch.empty(self.dim_g, self.dim_z))
        self.act = POTENTIAL2ACT[config.potential_func_g](dim=-1, eps=config.potential_eps)
        self._init_binary()

    def _init_binary(self):
        nn.init.normal_(self.binary_factor, mean=0.0, std=self.config.binary_initializer_range)

    def forward(self, qz: torch.Tensor):
        qg = nn.functional.linear(qz, self.binary_factor) * self.config.binary_factor_scaling
        qg = self.act(qg / self.config.regularize_g)
        message_G = qg @ self.binary_factor * self.config.binary_factor_scaling
        return message_G


<<<<<<< HEAD
class PtGroupPooling(nn.Module):
    """Channel Grouping Prior: parameter-free within-group consensus message.

    For each group of channels, compute the average Z representation across
    group members and send it directly (with a large fixed scaling) as an
    additional message to each group member.  No learnable projection —
    the prior IS the structural knowledge of which channels to pool.
    """
    SCALING = 5.0  # match the magnitude used by the successful PtLagPrior

    def __init__(self, args):
        super().__init__()
        self.enc_in = args.enc_in
        # Default groups for 9-channel synthetic dataset
        self.groups = getattr(args, 'channel_groups',
                              [(0, 1, 2), (3, 4, 5), (6, 7, 8)])

    def forward(self, qz_norm: torch.Tensor) -> torch.Tensor:
        """Compute group-pooling messages.
        Args:
            qz_norm: normalized qz, shape [bs, enc_in, patch_num, dim_z]
        Returns:
            group_message: same shape, to be added into qz update.
        """
        msg = torch.zeros_like(qz_norm)
        for group in self.groups:
            group = [ch for ch in group if ch < qz_norm.shape[1]]
            if len(group) < 2:
                continue
            # Compute within-group average (parameter-free denoising)
            group_avg = qz_norm[:, group].mean(dim=1)  # [bs, patch_num, dim_z]
            # Send directly with large scaling — no learnable W matrix
            for ch in group:
                msg[:, ch] = self.SCALING * group_avg
        return msg


=======
>>>>>>> 271d57e3538bd3b9b4b75420f058fc6a4e21295c
class PtEncoderIterator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = config
        self.dim_z = args.d_model
        self.head_selection = PtHeadSelection(args)
        self.topic_modeling = PtTopicModeling(args)
        self.source_separation = PtSourceSeparation(args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

    def forward(
        self,
        unary_potentials: torch.Tensor,
        qz: torch.Tensor,
        dependency_mask_channel: Optional[torch.Tensor] = None,
        dependency_mask_time: Optional[torch.Tensor] = None,
        position_ids_time: Optional[torch.LongTensor] = None,
        position_ids_channel: Optional[torch.LongTensor] = None,
        output_dependencies: Optional[bool] = False,
        position_embeddings_time: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_channel: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        old_qz = qz
        qz = self.norm(qz)

        # Source separation prior message (MFVI: unmix -> normalize -> re-mix)
        m_source = self.source_separation(qz)

        # head selection (ternary factors)
        m_t, m_c, qh_t, qh_c = self.head_selection(
            qz=qz,
            dependency_mask_channel=dependency_mask_channel,
            dependency_mask_time=dependency_mask_time,
            position_ids_time=position_ids_time,
            position_ids_channel=position_ids_channel,
            output_dependencies=output_dependencies,
            position_embeddings_time=position_embeddings_time,
            position_embeddings_channel=position_embeddings_channel,
        )

        # topic modeling (binary factors)
        m_g = self.topic_modeling(qz)

        # MFVI update: qz = (m_t + m_c + m_g + m_source + unary) / regularize_z
        qz = (m_t + m_c + m_g + m_source + unary_potentials) / self.config.regularize_z
        # damping
        qz = (qz + old_qz) * .5

        return qz



class PtModel(nn.Module):
    def __init__(self, args):
        super(PtModel, self).__init__()
        self.dim_z = args.d_model
        self.patch_len = args.patch_len

        self.iterator = PtEncoderIterator(args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

        config_copy = PtConfig.from_dict(config.to_dict())
        config_copy.hidden_size = self.dim_z
        config_copy.num_attention_heads = args.n_heads
        config_copy.head_dim = self.dim_z // args.n_heads
        self.rotary_emb_time = LlamaRotaryEmbedding(config = config_copy)
        self.rotary_emb_channel = LlamaRotaryEmbedding(config = config_copy)

        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.num_iteration = args.e_layers
        self.patch_num = self.seq_len // self.patch_len
        self.prediction = nn.Linear(self.patch_num * self.dim_z, self.pred_len, bias=True)

        self.unary_factors = nn.Sequential(
			nn.Linear(self.patch_len, self.dim_z),
            nn.GELU(),
			nn.Linear(self.dim_z, self.dim_z)
		)


    def forward(
        self,
        time_series: torch.LongTensor = None,
        x_mark_enc = None,
        x_dec = None,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        unary_potentials: Optional[torch.FloatTensor] = None,
        output_dependencies: Optional[bool] = None,
        output_qzs: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        device = time_series.device
        means = time_series.mean(1, keepdim=True).detach()
        time_series = time_series - means
        stdev = torch.sqrt(
            torch.var(time_series, dim=1, keepdim=True, unbiased=False) + 1e-5)
        time_series /= stdev

        time_series = time_series.transpose(1, 2)
        time_series = time_series.reshape(-1, self.enc_in, self.patch_num, self.patch_len)
        unary_potentials = self.unary_factors(time_series)
        batch_size, enc_in, seq_len, _ = unary_potentials.shape
        dependency_mask_time = torch.ones((batch_size*enc_in, seq_len), device = device)
        dependency_mask_channel = torch.ones((batch_size*seq_len, enc_in), device = device)

        unary_type = unary_potentials.dtype
        if position_ids is None:
            device = time_series.device if time_series is not None else unary_potentials.device
            position_ids_time = torch.arange(
                0, seq_len, dtype=torch.long, device=device
            )
            position_ids_time = position_ids_time.unsqueeze(0)
            position_ids_channel = torch.arange(
                0, enc_in, dtype=torch.long, device=device
            )
            position_ids_channel = position_ids_channel.unsqueeze(0)
        dependency_mask_channel = self._update_dependency_mask(
            dependency_mask_channel, enc_in, output_dependencies, unary_type
        )
        dependency_mask_time = self._update_dependency_mask(
            dependency_mask_time, seq_len, output_dependencies, unary_type
        )
        qz = unary_potentials

        position_embeddings_channel = self.rotary_emb_channel(unary_potentials.transpose(1, 2).reshape(batch_size * seq_len, enc_in, -1), position_ids_channel)
        position_embeddings_time = self.rotary_emb_time(unary_potentials.view(batch_size*enc_in, seq_len, -1), position_ids_time)

        all_qzs = () if output_qzs else None
        all_qhs = () if output_dependencies else None

        for idx in range(self.num_iteration):
            if output_qzs:
                all_qzs += (qz,)

            qz = self.iterator(
                unary_potentials,
                qz,
                dependency_mask_channel=dependency_mask_channel,
                dependency_mask_time = dependency_mask_time,
                position_ids_time=position_ids_time,
                position_ids_channel=position_ids_channel,
                output_dependencies=output_dependencies,
                position_embeddings_time=position_embeddings_time,
                position_embeddings_channel=position_embeddings_channel
            )

        dec_out = self.prediction(qz.reshape(batch_size, self.enc_in, -1)).permute(0, 2, 1)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        return dec_out



    def _update_dependency_mask(
        self, dependency_mask: torch.Tensor, seq_length, output_dependencies: bool, type
    ) -> torch.Tensor:

        attn_mask_converter = AttentionMaskConverter(is_causal=False)
        dependency_mask = attn_mask_converter.to_4d(
            dependency_mask, seq_length, dtype = type
        )

        diag_mask = torch.eye(seq_length, dtype=dependency_mask.dtype, device=dependency_mask.device).unsqueeze(0).unsqueeze(0)
        dependency_mask = dependency_mask.masked_fill(diag_mask.to(torch.bool), torch.finfo(dependency_mask.dtype).min)
        return dependency_mask


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.model = PtModel(args)

    def forward(
        self,
        x_enc: Optional[torch.Tensor] = None,
        x_mark_enc = None,
        x_dec = None,
        x_mark_dec = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        outputs = self.model(
            time_series = x_enc
        )
        return outputs
