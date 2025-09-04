import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class Model(nn.Module):
    """
    TSFusion: A Dual-Modality Fusion Model for Time Series Forecasting.
    This version introduces the CLIP idea as an auxiliary contrastive loss
    for end-to-end multi-task training, alongside FiLM fusion for the primary task.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads

        # --- Dual Encoders ---
        self.temporal_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.temporal_encoder = Encoder(
            [EncoderLayer(AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation) for l in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.variate_embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.variate_encoder = Encoder(
            [EncoderLayer(AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation) for l in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # --- Projection Heads for Auxiliary Contrastive Loss ---
        # projection_dim should be a new hyperparameter, e.g., 128
        projection_dim = getattr(configs, 'projection_dim', 128)
        self.temporal_projection_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, projection_dim)
        )
        self.variate_projection_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, projection_dim)
        )

        # --- FiLM Fusion Module for Forecasting Task ---
        self.gamma_generator = nn.Linear(self.d_model, self.d_model)
        self.beta_generator = nn.Linear(self.d_model, self.d_model)
        self.fusion_norm1 = nn.LayerNorm(self.d_model)
        self.fusion_norm2 = nn.LayerNorm(self.d_model)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, self.d_model)
        )
        self.fusion_dropout = nn.Dropout(configs.dropout)

        # --- Prediction Head ---
        self.predict_1 = nn.Linear(self.d_model, configs.c_out)
        self.predict_2 = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Channels]
        
        # 1. Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm /= stdev
        
        # 2. Dual Encoding
        temporal_enc_out = self.temporal_embedding(x_enc_norm, x_mark_enc)
        H_t, _ = self.temporal_encoder(temporal_enc_out, attn_mask=None)

        variate_enc_in = x_enc_norm.permute(0, 2, 1)
        variate_enc_out = self.variate_embedding(variate_enc_in)
        H_v, _ = self.variate_encoder(variate_enc_out, attn_mask=None)

        # --- Path for Auxiliary Contrastive Task ---
        # Get global features for contrastive loss
        h_t_global = H_t.mean(dim=1) # Shape: [Batch, d_model]
        h_v_global = H_v.mean(dim=1) # Shape: [Batch, d_model]
        
        # Project to contrastive space
        temporal_proj = self.temporal_projection_head(h_t_global)
        variate_proj = self.variate_projection_head(h_v_global)

        # --- Path for Primary Forecasting Task ---
        # 3. FiLM Fusion
        gamma = self.gamma_generator(h_v_global).unsqueeze(1)
        beta = self.beta_generator(h_v_global).unsqueeze(1)
        modulated_out = gamma * H_t + beta
        fused_out = self.fusion_norm1(modulated_out)
        fused_out = fused_out + self.fusion_dropout(self.fusion_ffn(fused_out))
        fused_out = self.fusion_norm2(fused_out)

        # 4. Prediction Head
        dec_out = self.predict_1(self.predict_2(fused_out.permute(0, 2, 1)).permute(0, 2, 1))

        # 5. De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Return both prediction and features for contrastive loss
        # The training loop will handle the loss calculation
        if self.training:
            return dec_out, temporal_proj, variate_proj
        else:
            return dec_out

# For TSLib compatibility
Model = Model