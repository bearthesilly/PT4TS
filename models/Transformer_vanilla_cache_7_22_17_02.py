import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Sequential(
            nn.Linear(c_in, 2*c_in),
            nn.GELU(),
            nn.Linear(2*c_in, d_model)
        )
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x, x_mark):
        value_embedding = self.value_embedding(x)
        x = value_embedding + self.position_embedding(x)
        return self.dropout(x), value_embedding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(0.0)
        self.activation = F.relu if activation == "relu" else F.gelu

        # Attention projections
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # ImplicitHMM
        self.binary_potential_table = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.identity = torch.eye(self.d_model)
        self._init_()

    def _init_(self):
        nn.init.normal_(self.binary_potential_table, mean=0.0, std=0.01)

    def forward(self, x, attn_mask=None, embedding=None):
        B, L, _ = x.shape
        H = self.n_heads
        d_head = self.d_model // H
        S = L
        # Project queries, keys, values
        queries = self.query_projection(x).view(B, L, H, d_head)
        keys = self.key_projection(x).view(B, L, H, d_head)
        values = self.value_projection(x).view(B, L, H, d_head)

        # Compute attention scores
        scores = torch.matmul(
            queries.view(B * H, L, d_head),  
            keys.view(B * H, S, d_head).transpose(-1, -2)  
        ).view(B, H, L, S) / math.sqrt(d_head)  # 将结果 reshape 回 [B, H, L, S] 并进行缩放
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -float('inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        attn_output = torch.einsum("bhls,bshd->blhd", attn_weights, values).reshape(B, L, -1)

        # Following implicitHMM
        self.identity = self.identity.to(x.device)
        back2front = nn.functional.linear(x, (self.binary_potential_table+self.identity).T) * 1.0
        front2back = nn.functional.linear(x, (self.binary_potential_table+self.identity)) * 1.0
        b2f = back2front.clone()
        back2front[:, :-1, :] = b2f[:, 1:, :]
        back2front[:, -1, :] = 0
        f2b = front2back.clone()
        front2back[:, 1:, :] = f2b[:, :-1, :]
        front2back[:, 0, :] = 0
        # Experiment shows that F.softmax is really important
        attn_output = F.softmax(attn_output + back2front + front2back)

        # Residual connection and normalization
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward network
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)
        return x, attn_weights

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation, num_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layer = EncoderLayer(d_model, n_heads, d_ff, dropout, activation)
        self.num_layers = num_layers
        self.norm = norm_layer # 2: MSE 0.597 MAE 0.326; 3 with 1 epoch: MSE 0.595 MAE 0.336

    def forward(self, x, attn_mask=None, embedding=None):
        attns = []
        for i in range(self.num_layers):
            x, attn = self.layer(x, attn_mask, embedding)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)
        self.encoder = Encoder(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            num_layers=configs.e_layers,
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.dropout)
            self.decoder1 = nn.Sequential(
                nn.Linear(configs.d_model, 2*configs.d_model),
                nn.GELU(),
                nn.Linear(2*configs.d_model, configs.c_out)
            )
            self.decoder2 = nn.Sequential(
                nn.Linear(configs.seq_len, 2*configs.seq_len),
                nn.GELU(),
                nn.Linear(2*configs.seq_len, configs.pred_len)
            )
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        enc_out, embedding = self.enc_embedding(x_enc, x_mark_enc)
        mask = None
        enc_out, _ = self.encoder(enc_out, attn_mask=mask, embedding = embedding)
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.decoder2(enc_out.transpose(1, 2)).transpose(1, 2)
            dec_out = self.decoder1(dec_out)
            dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                            1, self.pred_len, 1))
            dec_out = dec_out + \
                        (means[:, 0, :].unsqueeze(1).repeat(
                            1, self.pred_len, 1))
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name in ['imputation', 'anomaly_detection']:
            return self.projection(enc_out)
        elif self.task_name == 'classification':
            enc_out = self.act(enc_out)
            enc_out = self.dropout(enc_out)
            enc_out = enc_out.reshape(enc_out.shape[0], -1)
            return self.projection(enc_out)
        return None