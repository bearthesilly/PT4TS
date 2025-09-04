'''
In this model, we take the 'channel independence' skill
We treat the time series of one channel as one token, instead of taking time steps with multi-channel feature as token
The final task head is:
[bs, channel_in, dim_z] -> [bs, channel_in, pred_len]
-> transpose -> [bs, pred_len, channel_in = channel_out]
The prediction head is simple and its parameter amount accounts for small proportion
But this model has the best performance!

Also in this model, the code will be formulated well for TSLib format!!!
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
# from transformers.modeling_utils import PreTrainedModel

from transformers.modeling_attn_mask_utils import AttentionMaskConverter


logger = logging.get_logger(__name__)
from typing import List, Optional

@dataclass
class Config:
    # dim_z: int = 256
    # dim_g: int = 1024
    # num_iterations: int = 12
    num_channels: int = 8
    # ternary_rank: int = 32
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
    regularize_h: float = 0.00390625
    regularize_g: float = 1.0
    hidden_size: int = 768
    output_qzs: bool = False


    def to_dict(self):
        return asdict(self)

config = Config()

class RopeApplier:
    def __init__(self, cos, sin, position_ids=None, unsqueeze_dim=1) -> None:
        """Applies Rotary Position Embedding to the query, key and value tensors.

        Args:
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        """
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
    # This is a workaround to allow passing the eps
    def __init__(self, dim=-1, eps=None):
        super().__init__(dim=dim)


POTENTIAL2ACT = {
    "exp": Softmax,
    "abs": AbsNormalization,
    "square": SquaredSoftmax,
}


class PtHeadSelection(nn.Module):
    """Multi-channel head selection from 'Probabilistic Transformer' paper"""
    
    def __init__(self, args):
        super().__init__()
        self.config = config
        self.dim_z = args.d_model
        self.num_channels = config.num_channels
        self.ternary_rank = self.dim_z // self.num_channels
        self.rope_theta = config.rope_theta
        self.is_causal = False
        self.regularize_h = 1/self.dim_z
        self.ternary_factor_u = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_v = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.dropout = nn.Dropout(config.dropout_prob_h)
        self._init_ternary()
    
    def _init_ternary(self):
        nn.init.normal_(self.ternary_factor_u, mean=0.0, std=self.config.ternary_initializer_range)
        nn.init.normal_(self.ternary_factor_v, mean=0.0, std=self.config.ternary_initializer_range)

    def forward(
        self,
        qz: torch.Tensor,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_dependencies: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = qz.size()
        qz_u = nn.functional.linear(qz, self.ternary_factor_u) * self.config.ternary_factor_scaling
        qz_v = nn.functional.linear(qz, self.ternary_factor_v) * self.config.ternary_factor_scaling
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
            message_F = message_F + dependency_mask # need mask diag
        # upcast attention to fp32
        qh = nn.functional.softmax(message_F / self.regularize_h, dim=-1, dtype=torch.float32).to(qz_u.dtype)
        torch.cuda.empty_cache()  
        qh_v1 = torch.matmul(qh, qz_v)
        # raise RuntimeError(qh.shape, qz_uo.shape) -> [4, 12, 1024, 1024], [4, 12, 1024, 64]
        qh_v2 = torch.matmul(qh.transpose(2, 3), qz_uo)
        # apply rotary position embedding to the output
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
        message_G = (torch.matmul(qh_v1, self.ternary_factor_u) + torch.matmul(qh_v2, self.ternary_factor_v)) * self.config.ternary_factor_scaling
        if not output_dependencies:
            qh = None
        # raise RuntimeError(message_G.shape) -> [4, 1024, 768]
        return message_G, qh

class PTPeriod(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seq_len = self.args.seq_len
        self.dim_z = self.args.d_model
        self.period = 24
        self._lambda = 0
        self.period_mat_list = nn.ParameterList()
        self.period_attn_list = nn.ParameterList()
        self._init_matrix()
        
    def _init_matrix(self):
        mats = []
        for i in range(2*self._lambda + 1):
            x = torch.arange(self.seq_len).view(-1, 1)
            y = torch.arange(self.seq_len).view(1, -1)
            matrix = (torch.abs(x - y) == (self.period - self._lambda + i)).float()
            mats.append(nn.Parameter(matrix, requires_grad=False))
        self.period_mat_list.extend(mats)
        attns = []
        for i in range(2*self._lambda + 1):
            attn_matrix = self.create_decaying_matrix(scale = (abs(i-self._lambda)+1))
            attns.append(nn.Parameter(attn_matrix, requires_grad=False))
        self.period_attn_list.extend(attns)

    def create_decaying_matrix(self, a=0.5, p=2, k=1, scale = 1):
        x = torch.arange(self.dim_z).view(-1, 1)
        y = torch.arange(self.dim_z).view(1, -1)
        dist = torch.abs(x - y).float()
        matrix = torch.exp(-a * dist ** p) / scale
        matrix[dist >= k] = 0.0
        return matrix
    
    def forward(self, qz):
        info = torch.zeros_like(qz)
        for i in range(2*self._lambda + 1):
            '''
            qz: [bs, seq_len, dim_z]
            period_mat_list[i]: [seq_len, seq_len]
            period_attn_list[i]: [dim_z, dim_z]
            '''
            # info += period_mat_list[i] * qz * period_attn_list[i]
            info += self.period_mat_list[i].unsqueeze(0) @ qz @ self.period_attn_list[i].unsqueeze(0)
        return info

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

class PtEncoderIterator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = config
        self.dim_z = args.d_model
        self.head_selection = PtHeadSelection(args)
        self.topic_modeling = PtTopicModeling(args)
        self.period = PTPeriod(args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)
    
    def forward(
        self,
        unary_potentials: torch.Tensor,
        qz: torch.Tensor,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_dependencies: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """z
        Args:
            qz (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            dependency_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_dependencies (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        old_qz = qz

        qz = self.norm(qz)

        # head selection
        m1, qh = self.head_selection(
            qz=qz,
            dependency_mask=dependency_mask,
            position_ids=position_ids,
            output_dependencies=output_dependencies,
            position_embeddings=position_embeddings,
        )

        # topic modeling
        m2 = self.topic_modeling(qz)

        period = self.period(qz)
        # unary potentials
        qz = (m1 + m2 + unary_potentials + period) * self.dim_z
        # damping
        qz = (qz + old_qz) * .5
        outputs = (qz,)
        if output_dependencies:
            outputs += (qh,)

        return outputs
    


class PtModel(nn.Module):
    def __init__(self, args):
        super(PtModel, self).__init__()
        self.dim_z = args.d_model
        self.head_dim = self.dim_z // config.num_channels
        # self.padding_idx = config.pad_token_id

        # Simple linear projection. Bad performance!
        # self.unary_factors = nn.Linear(args.enc_in, self.dim_z)

        self.iterator = PtEncoderIterator(args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

        # XXX: This is a workaround to initialize the rotary embeddings
        config_copy = PtConfig.from_dict(config.to_dict())
        config_copy.head_dim = self.head_dim
        config_copy.hidden_size = self.dim_z
        config_copy.num_attention_heads = config.num_channels
        self.rotary_emb = LlamaRotaryEmbedding(config = config_copy)
        
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.gradient_checkpointing = False
        # self.fc = nn.Linear(self.dim_z, args.num_class)
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.num_iteration = args.e_layers
        # self.predict_1 = nn.Linear(
        #         config.dim_z, args.c_out)
        # Instead of linear transformation, we use MLP to improve expressiveness
        self.prediction = nn.Linear(self.dim_z, self.pred_len, bias=True)
        
        self.predict_1 = nn.Sequential(
            nn.Linear(self.dim_z, 2*self.dim_z),
            nn.GELU(),
            nn.Linear(2*self.dim_z, args.c_out)
        )
        # self.predict_2 = nn.Linear(
        #         self.seq_len, args.pred_len)
        self.predict_2 = nn.Sequential(
            nn.Linear(self.seq_len, 2*self.seq_len),
            nn.GELU(),
            nn.Linear(2*self.seq_len, args.pred_len)
        )
        
        if args.dropout != None:
            self.dropout = nn.Dropout(args.dropout)

        self.unary_factors = nn.Sequential(
			nn.Linear(args.enc_in, self.dim_z*2),
            nn.GELU(),
			nn.Linear(self.dim_z*2, self.dim_z)
		)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        x_mark_enc = None,
        x_dec = None,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        unary_potentials: Optional[torch.FloatTensor] = None,
        output_dependencies: Optional[bool] = None,
        output_qzs: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # preparing mask
        means = input_ids.mean(1, keepdim=True).detach()
        input_ids = input_ids - means
        stdev = torch.sqrt(
            torch.var(input_ids, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input_ids /= stdev
        # input_ids = input_ids.transpose(1, 2) # [bs, length, enc_in] -> [bs, enc_in, length]
        device = input_ids.device
        dependency_mask = torch.ones((input_ids.shape[0], self.seq_len), device = device)
        # Encode or Embedding, and then project into longer timesteps
        unary_potentials = self.unary_factors(input_ids) 
        seq_length = unary_potentials.size(1)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else unary_potentials.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        dependency_mask = self._update_dependency_mask(
            dependency_mask, unary_potentials, output_dependencies
        )
        qz = unary_potentials
        position_embeddings = self.rotary_emb(qz, position_ids)

        # decoder layers
        all_qzs = () if output_qzs else None
        all_qhs = () if output_dependencies else None

        for idx in range(self.num_iteration):
            if output_qzs:
                all_qzs += (qz,)

            iter_outputs = self.iterator(
                unary_potentials,
                qz,
                dependency_mask=dependency_mask,
                position_ids=position_ids,
                output_dependencies=output_dependencies,
                position_embeddings=position_embeddings,
            )
            qz = iter_outputs[0]
            if output_dependencies:
                all_qhs += (iter_outputs[1],)
        # add hidden states from the last decoder layer
        if output_qzs:
            all_qzs += (qz,)

        # qz = self.norm(qz) # [bs, length, dim_z]
        # If removing norm away, the performance is better???
        bs = qz.shape[0]
        # The qz above is equivalent to the enc_out whose timesteps are extended
        dec_out = self.predict_1(qz)  # [bs, length, dim_z]
        dec_out = self.predict_2(dec_out.transpose(1, 2)).transpose(1, 2)  # [bs, pred_len, c_out]
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        return dec_out


    
    def _update_dependency_mask(
        self, dependency_mask: torch.Tensor, unary_potentials: torch.Tensor, output_dependencies: bool
    ) -> torch.Tensor:
        
        seq_length = unary_potentials.size(1)
        
        attn_mask_converter = AttentionMaskConverter(is_causal=False)
        dependency_mask = attn_mask_converter.to_4d(
            dependency_mask, seq_length, dtype=unary_potentials.dtype
        )
        
        # mask diagonals
        diag_mask = torch.eye(seq_length, dtype=dependency_mask.dtype, device=dependency_mask.device).unsqueeze(0).unsqueeze(0)
        dependency_mask = dependency_mask.masked_fill(diag_mask.to(torch.bool), torch.finfo(dependency_mask.dtype).min)
        return dependency_mask


class Model(nn.Module):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, args):
        super(Model, self).__init__()
        self.model = PtModel(args)
        # Initialize weights and apply final processing

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        
        outputs = self.model(
            input_ids = x_enc,
            x_mark_enc = x_mark_enc,
            x_dec = x_dec,
            dependency_mask=attention_mask,
            position_ids=position_ids,
            unary_potentials=inputs_embeds,
            output_dependencies=output_attentions,
            output_qzs=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs