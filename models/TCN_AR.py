import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.norm(x + self.net(x))


class Model(nn.Module):
    """
    Causal TCN one-step forecaster rolled out autoregressively.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_size = int(getattr(configs, "ar_hidden", configs.d_model))
        self.num_layers = max(1, int(getattr(configs, "ar_layers", configs.e_layers)))
        self.kernel_size = int(getattr(configs, "tcn_kernel_size", 3))
        self.dropout = float(configs.dropout)
        self.teacher_forcing_ratio = float(getattr(configs, "teacher_forcing_ratio", 1.0))

        self.input_proj = nn.Conv1d(self.enc_in, self.hidden_size, kernel_size=1)
        blocks = []
        for i in range(self.num_layers):
            blocks.append(
                TemporalBlock(
                    channels=self.hidden_size,
                    kernel_size=self.kernel_size,
                    dilation=2 ** i,
                    dropout=self.dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.output = nn.Linear(self.hidden_size, self.c_out)

    def _normalize(self, x):
        means = x.mean(dim=1, keepdim=True).detach()
        x_centered = x - means
        stdev = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return x_centered / stdev, means, stdev

    def _future_norm(self, y_true, means, stdev):
        if y_true is None:
            return None
        future = y_true[:, -self.pred_len:, :]
        return (future - means) / stdev

    def _one_step(self, history):
        # history: [B, T, K]
        h = self.input_proj(history.transpose(1, 2))
        h = self.tcn(h)
        return self.output(h[:, :, -1])

    def forecast(self, x_enc, y_true=None):
        x_norm, means, stdev = self._normalize(x_enc)
        teacher = self._future_norm(y_true, means, stdev)
        history = x_norm
        preds = []

        for step in range(self.pred_len):
            y_hat = self._one_step(history)
            preds.append(y_hat.unsqueeze(1))

            if self.training and teacher is not None and self.teacher_forcing_ratio > 0:
                if self.teacher_forcing_ratio >= 1.0:
                    next_y = teacher[:, step, :]
                else:
                    use_teacher = (
                        torch.rand(y_hat.size(0), 1, device=y_hat.device)
                        < self.teacher_forcing_ratio
                    ).to(y_hat.dtype)
                    next_y = use_teacher * teacher[:, step, :] + (1.0 - use_teacher) * y_hat
            else:
                next_y = y_hat

            history = torch.cat([history[:, 1:, :], next_y.unsqueeze(1)], dim=1)

        out = torch.cat(preds, dim=1)
        return out * stdev[:, 0:1, :] + means[:, 0:1, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, y_true=None, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc, y_true=y_true)
        raise NotImplementedError("TCN_AR only supports forecasting tasks.")
