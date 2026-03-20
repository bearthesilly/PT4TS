import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Vector Autoregression (VAR) baseline for multivariate time series forecasting.

    Mathematical formulation:
        y_t = c + A_1 * y_{t-1} + A_2 * y_{t-2} + ... + A_p * y_{t-p} + e_t

    where:
        y_t     : K-dimensional observation at time t
        c       : K-dimensional intercept vector
        A_i     : K x K coefficient matrix for lag i
        p       : lag order
        e_t     : K-dimensional white noise with covariance Sigma

    Estimation uses Ridge-regularized OLS on the companion form:
        Y = X @ B + E
    where B = [c, A_1, ..., A_p]^T  (shape: (1+pK) x K)

    This model is a non-learning statistical baseline. It fits per-sample at
    inference time (each input window gets its own VAR coefficients), so it
    requires NO gradient-based training. The training loop is effectively a
    no-op; the model evaluates identically whether you "train" 0 or 100 epochs.

    Compatibility:
        - Inherits nn.Module so it plugs into the exp/ framework.
        - A dummy parameter exists so the optimizer doesn't crash.
        - forward() signature matches the standard (x_enc, x_mark_enc, x_dec, x_mark_dec).
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)

        default_lag = min(5, max(1, self.seq_len // 10))
        self.lag_order = int(getattr(configs, 'lag_order', default_lag))
        self.ridge = float(getattr(configs, 'var_ridge', 1e-2))

        # Dummy parameter so optimizer / state_dict don't break
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    # ------------------------------------------------------------------
    # Core VAR math
    # ------------------------------------------------------------------

    def _fit_and_forecast(self, x, pred_len):
        """
        Fit a VAR(p) model on each sample independently, then forecast.

        Args:
            x: (B, T, K) — input time series (already scaled by the dataloader)
            pred_len: int — number of future steps to predict

        Returns:
            preds: (B, pred_len, K)
        """
        B, T, K = x.shape
        p = self.lag_order
        device = x.device
        dtype = x.dtype

        if T <= p:
            # Not enough history — fall back to naive persistence
            return x[:, -1:, :].expand(B, pred_len, K).clone()

        # ---- Per-sample instance normalization (reversible) ----
        # This removes per-sample level/scale so the VAR operates on
        # standardized residuals, which is standard practice.
        seq_mean = x.mean(dim=1, keepdim=True)          # (B, 1, K)
        seq_std = x.std(dim=1, keepdim=True) + 1e-8     # (B, 1, K)
        x_norm = (x - seq_mean) / seq_std

        # ---- Build regression matrices per sample ----
        # Y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p}
        # Stack into:  Y = X @ B   where B is (1+pK, K)
        #
        # Y: (B, T-p, K)
        # X: (B, T-p, 1+pK)   — leading 1 column for intercept
        Y = x_norm[:, p:, :]                             # (B, T-p, K)
        Teff = T - p

        X_parts = [torch.ones(B, Teff, 1, device=device, dtype=dtype)]
        for i in range(1, p + 1):
            X_parts.append(x_norm[:, p - i: T - i, :])  # lag i
        X = torch.cat(X_parts, dim=-1)                   # (B, Teff, 1+pK)

        D = 1 + p * K

        # ---- Ridge OLS:  B = (X'X + lambda I)^{-1} X'Y  per sample ----
        # We batch this with bmm.
        Xt = X.transpose(1, 2)                           # (B, D, Teff)
        XtX = torch.bmm(Xt, X)                           # (B, D, D)
        XtY = torch.bmm(Xt, Y)                           # (B, D, K)

        eye = torch.eye(D, device=device, dtype=dtype).unsqueeze(0)  # (1, D, D)
        XtX_reg = XtX + self.ridge * eye

        # Use torch.linalg.solve for numerical stability (solves AX=B)
        # coef: (B, D, K)
        try:
            coef = torch.linalg.solve(XtX_reg, XtY)
        except Exception:
            # Fallback: heavier regularization
            XtX_reg = XtX + eye
            coef = torch.linalg.solve(XtX_reg, XtY)

        # Clamp to prevent explosive coefficients
        coef = torch.clamp(coef, -10.0, 10.0)

        # ---- Extract intercept c and lag matrices A_i ----
        c = coef[:, 0, :]                                # (B, K)
        A_list = []
        for i in range(p):
            start = 1 + i * K
            A_list.append(coef[:, start: start + K, :])  # (B, K, K)

        # ---- Iterative forecasting ----
        history = x_norm[:, -p:, :].clone()               # (B, p, K)
        preds = []

        for _ in range(pred_len):
            y_new = c.clone()                             # (B, K)
            for i, A in enumerate(A_list):
                y_lag = history[:, -(i + 1), :]           # (B, K)
                # y_lag @ A  gives (B, K) — each sample's lag multiplied
                # by its own coefficient matrix
                y_new = y_new + torch.bmm(
                    y_lag.unsqueeze(1), A
                ).squeeze(1)

            # Clamp predictions in normalized space
            y_new = torch.clamp(y_new, -10.0, 10.0)
            preds.append(y_new.unsqueeze(1))

            # Shift history window
            history = torch.cat([history[:, 1:, :], y_new.unsqueeze(1)], dim=1)

        result = torch.cat(preds, dim=1)                  # (B, pred_len, K)

        # ---- Reverse instance normalization ----
        result = result * seq_std + seq_mean

        # Safety net: if anything blew up, fall back to persistence
        bad = torch.isnan(result) | torch.isinf(result)
        if bad.any():
            fallback = x[:, -1:, :].expand_as(result)
            result = torch.where(bad, fallback, result)

        return result

    # ------------------------------------------------------------------
    # Task-specific entry points
    # ------------------------------------------------------------------

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self._fit_and_forecast(x_enc, self.pred_len)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Simple imputation: forecast seq_len steps from a zero-history seed.
        For a proper VAR imputation you'd use the Kalman smoother, but this
        keeps the interface consistent.
        """
        # Use the observed (masked) portion to forecast the full window
        out = self._fit_and_forecast(x_enc, self.seq_len)
        # Blend: keep observed values, fill missing with VAR prediction
        return out * (1 - mask) + x_enc * mask

    def anomaly_detection(self, x_enc):
        """
        Return one-step-ahead predictions for the full input window.
        Anomaly score = |x_t - x_hat_t|.
        """
        B, T, K = x_enc.shape
        p = self.lag_order
        if T <= p + 1:
            return x_enc.clone()

        # Fit on the full window, get one-step-ahead fitted values
        x_norm, seq_mean, seq_std = self._normalize_instance(x_enc)
        Y, X, coef = self._fit_ols(x_norm)

        fitted = torch.bmm(X, coef)                      # (B, T-p, K)
        fitted = fitted * seq_std + seq_mean

        # Pad the first p steps with the actual values (no prediction possible)
        prefix = x_enc[:, :p, :]
        return torch.cat([prefix, fitted], dim=1)

    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError("VAR does not support classification.")

    # ------------------------------------------------------------------
    # Helpers for anomaly_detection (avoid code duplication)
    # ------------------------------------------------------------------

    def _normalize_instance(self, x):
        seq_mean = x.mean(dim=1, keepdim=True)
        seq_std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - seq_mean) / seq_std, seq_mean, seq_std

    def _fit_ols(self, x_norm):
        B, T, K = x_norm.shape
        p = self.lag_order
        device = x_norm.device
        dtype = x_norm.dtype

        Y = x_norm[:, p:, :]
        Teff = T - p
        D = 1 + p * K

        X_parts = [torch.ones(B, Teff, 1, device=device, dtype=dtype)]
        for i in range(1, p + 1):
            X_parts.append(x_norm[:, p - i: T - i, :])
        X = torch.cat(X_parts, dim=-1)

        Xt = X.transpose(1, 2)
        XtX = torch.bmm(Xt, X)
        XtY = torch.bmm(Xt, Y)
        eye = torch.eye(D, device=device, dtype=dtype).unsqueeze(0)
        XtX_reg = XtX + self.ridge * eye

        try:
            coef = torch.linalg.solve(XtX_reg, XtY)
        except Exception:
            XtX_reg = XtX + eye
            coef = torch.linalg.solve(XtX_reg, XtY)

        coef = torch.clamp(coef, -10.0, 10.0)
        return Y, X, coef

    # ------------------------------------------------------------------
    # Standard forward — matches the exp/ framework signature
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        return None
