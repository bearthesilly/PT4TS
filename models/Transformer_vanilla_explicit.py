import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
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
        # ExplicitHMM
        self.act = AbsNormalization()
        self.binary_potential_table = nn.Parameter(torch.empty(self.d_model, self.d_model)) # between Z and M
        self.markovian_potential = nn.Parameter(torch.empty(self.d_model, self.d_model)) # between M and M\
        self._init_()
    def _init_(self):
        nn.init.normal_(self.binary_potential_table, mean=0.0, std=0.2)
        nn.init.normal_(self.markovian_potential, mean=0.0, std=0.2)

    def forward(self, x, attn_mask=None, qm = None):
        B, L, _ = x.shape
        H = self.n_heads
        d_head = self.d_model // H
        S = L
        # Project queries, keys, values
        queries = self.query_projection(x).view(B, L, H, d_head)
        keys = self.key_projection(x).view(B, L, H, d_head)
        values = self.value_projection(x).view(B, L, H, d_head)

        # ExplicitHMM
        if (qm == None):
            qm = nn.functional.linear(x, self.binary_potential_table) 
            qm = self.act(qm)
        old_qm = qm
        back2front = nn.functional.linear(qm, self.markovian_potential.T)
        front2back = nn.functional.linear(qm, self.markovian_potential)
        b2f = back2front.clone()
        back2front[:, :-1, :] = b2f[:, 1:, :]
        back2front[:, -1, :] = 0
        f2b = front2back.clone()
        front2back[:, 1:, :] = f2b[:, :-1, :]
        front2back[:, 0, :] = 0
        z2markov = nn.functional.linear(x, self.binary_potential_table)
        qm = (old_qm + back2front + front2back + z2markov) * .5 # damping
        # Update Z's partial information which are related to Z and return. The returned information will be used to update Z
        qm = F.softmax(qm)
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

        attn_output = F.softmax(self.out_projection(attn_output) + z2markov)

        # Residual connection and normalization
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward network
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)
        return x, attn_weights, qm

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation, num_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layer = EncoderLayer(d_model, n_heads, d_ff, dropout, activation)
        self.num_layers = num_layers
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        qm = None
        for _ in range(self.num_layers):
            x, attn, qm = self.layer(x, attn_mask, qm)
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
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # mask = torch.ones(self.configs.seq_len, self.configs.seq_len).bool().to(x_enc.device)
        # mask.fill_diagonal_(False)  # 构造对角线为 False 的掩码
        mask = None
        enc_out, _ = self.encoder(enc_out, attn_mask=mask)
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