import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    DeepVAR-style recurrent autoregressive baseline.

    This implementation uses a shared recurrent state for all variables and
    produces a diagonal Gaussian mean/scale at each step. The experiment
    framework evaluates the mean forecast with MSE/MAE, while the scale head is
    kept available for probabilistic extensions.
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

        self.input_proj = nn.Linear(self.enc_in, self.hidden_size)
        self.encoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.decoder_cell = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.mean_head = nn.Linear(self.hidden_size, self.c_out)
        self.scale_head = nn.Linear(self.hidden_size, self.c_out)
        self.dropout_layer = nn.Dropout(self.dropout)

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
        enc_in = self.input_proj(x_norm)
        _, h = self.encoder(enc_in)
        state = h[-1]

        teacher = self._future_norm(y_true, means, stdev)
        prev_y = x_norm[:, -1, :]
        preds = []
        scales = []

        for step in range(self.pred_len):
            dec_input = self.input_proj(prev_y)
            state = self.decoder_cell(dec_input, state)
            state_drop = self.dropout_layer(state)
            mean = self.mean_head(state_drop)
            scale = F.softplus(self.scale_head(state_drop)) + 1e-4
            preds.append(mean.unsqueeze(1))
            scales.append(scale.unsqueeze(1))

            if self.training and teacher is not None and self.teacher_forcing_ratio > 0:
                if self.teacher_forcing_ratio >= 1.0:
                    prev_y = teacher[:, step, :]
                else:
                    use_teacher = (
                        torch.rand(prev_y.size(0), 1, device=prev_y.device)
                        < self.teacher_forcing_ratio
                    ).to(prev_y.dtype)
                    prev_y = use_teacher * teacher[:, step, :] + (1.0 - use_teacher) * mean
            else:
                prev_y = mean

        mean_norm = torch.cat(preds, dim=1)
        scale_norm = torch.cat(scales, dim=1)
        self._last_mean_norm = mean_norm
        self._last_scale_norm = scale_norm
        self._last_means = means
        self._last_stdev = stdev
        self.last_scale = scale_norm * stdev[:, 0:1, :]
        return mean_norm * stdev[:, 0:1, :] + means[:, 0:1, :]

    def nll_loss(self, target):
        target_norm = (target - self._last_means[:, 0:1, :]) / self._last_stdev[:, 0:1, :]
        scale = self._last_scale_norm.clamp_min(1e-4)
        squared = ((target_norm - self._last_mean_norm) / scale) ** 2
        return 0.5 * (squared + 2.0 * torch.log(scale)).mean()

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, y_true=None, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc, y_true=y_true)
        raise NotImplementedError("DeepVAR only supports forecasting tasks.")
