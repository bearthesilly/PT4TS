'''
Prediction method: dim_z -> c_out
seq_len -> pred_len (with transpose function)
This model has explicitHMM enhancement
Compared to others, it use MLP instead of linear operation when predicting
Hoping this will improve the performance on 'summit prediction'
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
    _name_or_path: str = "meta-llama/Llama-2-7b-hf"
    architectures: List[str] = ("PtForMaskedLM",)
    bos_token_id: int = 1
    eos_token_id: int = 2
    model_type: str = "pt"
    dim_z: int = 512
    dim_g: int = 2048
    dim_m: int = 512
    num_iterations: int = 12
    num_channels: int = 8
    ternary_rank: int = 64
    potential_func_z: str = "square"
    potential_func_g: str = "abs"
    max_position_embeddings: int = 1024
    initializer_range: float = 0.02
    binary_table_initializer_range: float = 0.2
    binary_initializer_range: float = 0.2
    ternary_initializer_range: float = 0.2
    binary_factor_scaling: float = 1.0
    binary_table_scaling: float = 1.0
    ternary_factor_scaling: float = 1.0
    classifier_amplifier: float = 768.0
    potential_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    dropout_prob_z: float = 0.0
    dropout_prob_h: float = 0.0
    classifier_dropout: Optional[float] = None
    regularize_z: float = 1.0
    regularize_h: float = 0.0001953
    regularize_g: float = 1.0
    hidden_size: int = 768
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-05
    output_heads: bool = False
    output_qzs: bool = False
    torch_dtype: str = "float32"
    transformers_version: str = "4.31.0.dev0"
    vocab_size: int = 32000

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
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim_z = config.dim_z
        self.num_channels = config.num_channels
        self.ternary_rank = config.ternary_rank
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False

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
        qh = nn.functional.softmax(message_F / self.config.regularize_h, dim=-1, dtype=torch.float32).to(qz_u.dtype)

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

class PtExplicitHMM(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.dim_z = config.dim_z
        self.dim_m = config.dim_m
        self.device = args.device
        self.binary_potential_table = nn.Parameter(torch.empty(self.dim_z, self.dim_z)) # between Z and M
        self.markovian_potential = nn.Parameter(torch.empty(self.dim_z, self.dim_z)) # between M and M\
        self.act = POTENTIAL2ACT[config.potential_func_g](dim=-1, eps=config.potential_eps)
        self.length = args.seq_len
        self.bs = args.batch_size
        self.norm = POTENTIAL2ACT["square"](dim=-1, eps=config.potential_eps)
        self._init_ternary()
    
    def _init_ternary(self):
        nn.init.normal_(self.binary_potential_table, mean=0.0, std=self.config.binary_table_initializer_range)
        nn.init.normal_(self.markovian_potential, mean=0.0, std=self.config.binary_table_initializer_range)
    def forward(self, qz, qm):
        if (qm == None):
            qm = nn.functional.linear(qz, self.binary_potential_table) * self.config.binary_table_scaling
            qm = self.act(qm / self.config.regularize_g)
        # Updating M
        old_qm = qm
        back2front = nn.functional.linear(qm, self.markovian_potential.T) * self.config.binary_table_scaling
        front2back = nn.functional.linear(qm, self.markovian_potential) * self.config.binary_table_scaling
        b2f = back2front.clone()
        back2front[:, :-1, :] = b2f[:, 1:, :]
        back2front[:, -1, :] = 0
        f2b = front2back.clone()
        front2back[:, 1:, :] = f2b[:, :-1, :]
        front2back[:, 0, :] = 0
        z2markov = nn.functional.linear(qz, self.binary_potential_table) * self.config.binary_table_scaling
        qm = (old_qm + back2front + front2back + z2markov) * .5 # damping
        # Update Z's partial information which are related to Z and return. The returned information will be used to update Z
        qm = self.act(qm / self.config.regularize_g)
        markov2z = nn.functional.linear(qm, self.binary_potential_table.T) * self.config.binary_table_scaling
        return markov2z, qm

class PtTopicModeling(nn.Module):
    """Topic modeling w/ global nodes."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim_z = config.dim_z
        self.dim_g = config.dim_g
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
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.config = config
        self.dim_z = config.dim_z
        self.head_selection = PtHeadSelection(config)
        self.topic_modeling = PtTopicModeling(config)
        self.explicit_hmm = PtExplicitHMM(config, args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)
    
    def forward(
        self,
        unary_potentials: torch.Tensor,
        qz: torch.Tensor,
        qm,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_dependencies: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
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
        markov2z, qm = self.explicit_hmm(qz, qm)
        # topic modeling
        m2 = self.topic_modeling(qz)

        # unary potentials
        qz = (m1 + m2 + unary_potentials + markov2z) / self.config.regularize_z

        # damping
        qz = (qz + old_qz) * .5

        outputs = (qz,)

        if output_dependencies:
            outputs += (qh,)

        return outputs, qm
    


class PtModel(nn.Module):
    def __init__(self, args):
        super(PtModel, self).__init__()
        self.dim_z = config.dim_z
        # self.padding_idx = config.pad_token_id

        # Simple linear projection. Bad performance!
        # self.unary_factors = nn.Linear(args.enc_in, self.dim_z)
        self.iterator = PtEncoderIterator(config, args)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

        # XXX: This is a workaround to initialize the rotary embeddings
        config_copy = PtConfig.from_dict(config.to_dict())
        config_copy.head_dim = config.ternary_rank
        config_copy.hidden_size = config.dim_z
        config_copy.num_attention_heads = config.num_channels
        self.rotary_emb = LlamaRotaryEmbedding(config = config_copy)
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        # self.predict_1 = nn.Linear(
        #         config.dim_z, args.c_out)
        # Instead of linear transformation, we use MLP to improve expressiveness
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
        means = input_ids.mean(1, keepdim=True).detach()
        input_ids = input_ids - means
        stdev = torch.sqrt(
            torch.var(input_ids, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input_ids /= stdev
        device = input_ids.device
        dependency_mask = torch.ones((input_ids.shape[0], self.seq_len), device = device)
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
        qm = None
        for idx in range(config.num_iterations):
            if output_qzs:
                all_qzs += (qz,)

            iter_outputs, qm = self.iterator(
                unary_potentials,
                qz,
                qm,
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
        # raise RuntimeError(dependency_mask.shape) -> [4, 1, 1024, 1024]
        # print(dependency_mask) -> 前27个一样, diagonal, 对角线为-3.4028e+38
        return dependency_mask
    


class Model(nn.Module):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super(Model, self).__init__()
        self.model = PtModel(config)
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