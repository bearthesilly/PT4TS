import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


'''
Probable improvement:
Low rank of A_b and A_f matrix
Multi set of Markov binary factor
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankHeadwiseLinear(nn.Module):
    """
    Head-wise low-rank linear map.

    For each head h:
        A_h = U_h V_h^T

    Forward map:
        y_h = A_h x_h = U_h(V_h^T x_h)

    Transposed map:
        y_h = A_h^T x_h = V_h(U_h^T x_h)

    Input : x [B, L, n_heads, d_head]
    Output: y [B, L, n_heads, d_head]
    """
    def __init__(self, n_heads, d_head, rank):
        super().__init__()
        assert rank > 0, "rank must be positive"

        self.n_heads = n_heads
        self.d_head = d_head
        self.rank = rank

        # A_h = U_h V_h^T
        # V: [H, R, Dh]
        # U: [H, Dh, R]
        self.V = nn.Parameter(torch.empty(n_heads, rank, d_head))
        self.U = nn.Parameter(torch.empty(n_heads, d_head, rank))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.U)

    def forward(self, x):
        """
        Apply A = U V^T

        x: [B, L, H, Dh]
        return: [B, L, H, Dh]
        """
        # z = V^T x
        # x: [B, L, H, Dh]
        # V: [H, R, Dh]
        # z: [B, L, H, R]
        z = torch.einsum('blhd,hrd->blhr', x, self.V)

        # y = U z
        # U: [H, Dh, R]
        # y: [B, L, H, Dh]
        y = torch.einsum('blhr,hdr->blhd', z, self.U)
        return y

    def forward_transpose(self, x):
        """
        Apply A^T = V U^T

        x: [B, L, H, Dh]
        return: [B, L, H, Dh]
        """
        # z = U^T x
        # U: [H, Dh, R]
        # z: [B, L, H, R]
        z = torch.einsum('blhd,hdr->blhr', x, self.U)

        # y = V z
        # V: [H, R, Dh]
        # y: [B, L, H, Dh]
        y = torch.einsum('blhr,hrd->blhd', z, self.V)
        return y


class MarkovFFN(nn.Module):
    """
    Multi-head, low-rank MarkovFFN
    Input/Output: x [B, L, D]

    One message passing per block:
        H = in_proj(x)                                  # [B, L, d_hidden]
        H = reshape(H, [B, L, n_heads, d_head])

        H_left  = shift_right(H)                        # t-1 -> t
        H_right = shift_left(H)                         # t+1 -> t

        msg = H
              + A_b(H_left)                             # head-wise low-rank
              + A_f(H_right)                            # head-wise low-rank

        M = act(msg)
        M = reshape(M, [B, L, d_hidden])

        y = out_proj(M)
        out = x + dropout(y)                            # residual inside module

    Notes:
    - If n_heads = 1, this degenerates to the single-head version.
    - Factors A_b / A_f are head-wise low-rank maps.
    """
    def __init__(
        self,
        d_model,
        d_hidden=None,
        dropout=0.1,
        activation="gelu",
        use_gate=False,
        bias=True,
        n_heads=1,
        factor_rank=16,
        factor_bias=False
    ):
        super().__init__()
        d_hidden = d_hidden or (4 * d_model)

        assert d_hidden % n_heads == 0, \
            f"d_hidden ({d_hidden}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.use_gate = use_gate
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        self.factor_rank = factor_rank

        # Lift: X -> H
        self.in_proj = nn.Linear(d_model, d_hidden, bias=bias)

        # Markov messages in latent space (head-wise low-rank)
        self.A = LowRankHeadwiseLinear(
            n_heads=n_heads,
            d_head=self.d_head,
            rank=factor_rank
        )

        # Readout: M -> DeltaX
        self.out_proj = nn.Linear(d_hidden, d_model, bias=bias)

        if use_gate:
            self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        else:
            self.gate_proj = None

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    @staticmethod
    def _shift_right(x):
        """
        For message t-1 -> t
        x: [B, L, H, Dh]
        """
        B, L, H, Dh = x.shape
        zeros = torch.zeros(B, 1, H, Dh, device=x.device, dtype=x.dtype)
        return torch.cat([zeros, x[:, :-1, :, :]], dim=1)

    @staticmethod
    def _shift_left(x):
        """
        For message t+1 -> t
        x: [B, L, H, Dh]
        """
        B, L, H, Dh = x.shape
        zeros = torch.zeros(B, 1, H, Dh, device=x.device, dtype=x.dtype)
        return torch.cat([x[:, 1:, :, :], zeros], dim=1)

    def forward(self, x):
        """
        x: [B, L, D]
        return: [B, L, D]  (includes residual x + delta)
        """
        B, L, _ = x.shape

        # [B, L, d_hidden]
        H = self.in_proj(x)

        # [B, L, n_heads, d_head]
        H = H.view(B, L, self.n_heads, self.d_head)

        # Neighbor states
        H_left = self._shift_right(H)   # t-1 -> t
        H_right = self._shift_left(H)   # t+1 -> t

        # Head-wise Markov message passing
        msg = H + self.A(H_left) + self.A.forward_transpose(H_right)
        M = self.activation(msg)

        # Merge heads back
        M = M.reshape(B, L, self.d_hidden)

        delta = self.out_proj(M)
        delta = self.dropout(delta)

        if self.use_gate:
            gate = torch.sigmoid(self.gate_proj(x))
            delta = gate * delta

        return x + delta


class EncoderLayerWithMarkovFFN(nn.Module):
    """
    Drop-in replacement for EncoderLayer.
    Keeps the same forward signature and returns (x, attn).
    """
    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
        markov_use_gate=False,
        markov_n_heads=1,
        markov_factor_rank=16,
        markov_factor_bias=False
    ):
        super(EncoderLayerWithMarkovFFN, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        # Replace conv1/conv2 FFN by multi-head low-rank MarkovFFN
        self.markov_ffn = MarkovFFN(
            d_model=d_model,
            d_hidden=d_ff,
            dropout=dropout,
            activation=activation,
            use_gate=markov_use_gate,
            n_heads=markov_n_heads,
            factor_rank=markov_factor_rank,
            factor_bias=markov_factor_bias
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Attention branch (same as original)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        # FFN branch replaced by MarkovFFN
        y = x = self.norm1(x)
        y = self.markov_ffn(y)   # returns y + delta internally

        # Keep the same style as your original code
        return self.norm2(y), attn