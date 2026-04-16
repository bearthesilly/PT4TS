import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Autoregressive LSTM decoder baseline.

    The model encodes the observed context once, then rolls out one future
    timestep at a time. During training, y_true can be supplied by the
    experiment loop for standard teacher forcing.
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
        self.dropout = float(configs.dropout)
        self.teacher_forcing_ratio = float(getattr(configs, "teacher_forcing_ratio", 1.0))

        self.encoder = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.decoder_cell = nn.LSTMCell(self.enc_in, self.hidden_size)
        self.output = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.c_out),
        )

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

    def forecast(self, x_enc, y_true=None):
        x_norm, means, stdev = self._normalize(x_enc)
        _, (h, c) = self.encoder(x_norm)
        h_t = h[-1]
        c_t = c[-1]

        teacher = self._future_norm(y_true, means, stdev)
        prev_y = x_norm[:, -1, :]
        preds = []

        for step in range(self.pred_len):
            h_t, c_t = self.decoder_cell(prev_y, (h_t, c_t))
            y_hat = self.output(h_t)
            preds.append(y_hat.unsqueeze(1))

            if self.training and teacher is not None and self.teacher_forcing_ratio > 0:
                if self.teacher_forcing_ratio >= 1.0:
                    prev_y = teacher[:, step, :]
                else:
                    use_teacher = (
                        torch.rand(prev_y.size(0), 1, device=prev_y.device)
                        < self.teacher_forcing_ratio
                    ).to(prev_y.dtype)
                    prev_y = use_teacher * teacher[:, step, :] + (1.0 - use_teacher) * y_hat
            else:
                prev_y = y_hat

        out = torch.cat(preds, dim=1)
        return out * stdev[:, 0:1, :] + means[:, 0:1, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, y_true=None, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc, y_true=y_true)
        raise NotImplementedError("LSTM_AR only supports forecasting tasks.")
