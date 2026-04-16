import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    LSTNet baseline with CNN, recurrent temporal encoder, optional skip-RNN,
    and autoregressive highway component.

    This follows the classic LSTNet spirit. It is a direct multi-horizon model
    with a linear AR/highway branch, not a strict step-by-step rollout model.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.hid_c = int(getattr(configs, "lstnet_cnn_hidden", max(16, configs.d_model // 2)))
        self.hid_r = int(getattr(configs, "ar_hidden", configs.d_model))
        self.hid_s = int(getattr(configs, "skip_hidden", max(4, configs.d_model // 16)))
        self.cnn_kernel = int(getattr(configs, "cnn_kernel", 6))
        self.skip = int(getattr(configs, "skip", 24))
        self.highway_window = int(getattr(configs, "highway_window", 24))
        self.dropout = nn.Dropout(float(configs.dropout))

        self.cnn_kernel = min(self.cnn_kernel, self.seq_len)
        self.conv = nn.Conv2d(1, self.hid_c, kernel_size=(self.cnn_kernel, self.enc_in))
        self.gru = nn.GRU(self.hid_c, self.hid_r, batch_first=True)

        conv_steps = self.seq_len - self.cnn_kernel + 1
        self.pt = conv_steps // self.skip if self.skip > 0 else 0
        if self.skip > 0 and self.pt > 0:
            self.skip_gru = nn.GRU(self.hid_c, self.hid_s, batch_first=True)
            head_in = self.hid_r + self.skip * self.hid_s
        else:
            self.skip_gru = None
            head_in = self.hid_r

        self.proj = nn.Linear(head_in, self.pred_len * self.c_out)
        if self.highway_window > 0:
            self.highway_window = min(self.highway_window, self.seq_len)
            self.highway = nn.Linear(self.highway_window, self.pred_len)
        else:
            self.highway = None

    def _normalize(self, x):
        means = x.mean(dim=1, keepdim=True).detach()
        x_centered = x - means
        stdev = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return x_centered / stdev, means, stdev

    def forecast(self, x_enc):
        x_norm, means, stdev = self._normalize(x_enc)
        batch_size = x_norm.size(0)

        c = x_norm.unsqueeze(1)
        c = F.relu(self.conv(c)).squeeze(3)  # [B, hid_c, conv_steps]
        c = self.dropout(c)

        r = c.transpose(1, 2)
        _, r_hidden = self.gru(r)
        r_hidden = self.dropout(r_hidden[-1])

        if self.skip_gru is not None:
            s = c[:, :, -self.pt * self.skip:]
            s = s.view(batch_size, self.hid_c, self.pt, self.skip)
            s = s.permute(0, 3, 2, 1).contiguous().view(batch_size * self.skip, self.pt, self.hid_c)
            _, s_hidden = self.skip_gru(s)
            s_hidden = s_hidden[-1].view(batch_size, self.skip * self.hid_s)
            s_hidden = self.dropout(s_hidden)
            hidden = torch.cat([r_hidden, s_hidden], dim=1)
        else:
            hidden = r_hidden

        out = self.proj(hidden).view(batch_size, self.pred_len, self.c_out)

        if self.highway is not None:
            z = x_norm[:, -self.highway_window:, :].permute(0, 2, 1)
            highway = self.highway(z).permute(0, 2, 1)
            out = out + highway

        return out * stdev[:, 0:1, :] + means[:, 0:1, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, y_true=None, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc)
        raise NotImplementedError("LSTNet only supports forecasting tasks.")
