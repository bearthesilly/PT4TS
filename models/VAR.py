import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Shared-coefficient VAR baseline (数值稳定版本)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)

        default_lag = min(5, max(1, self.seq_len // 10))
        self.lag_order = int(getattr(configs, 'lag_order', default_lag))

        # 增大 Ridge 正则化强度
        self.ridge = float(getattr(configs, 'var_ridge', 1e-2))  # 从 1e-4 改为 1e-2

        self.coefficients = None
        self.sigma = None
        self.fitted = False

        # 数据标准化参数
        self.data_mean = None
        self.data_std = None

        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def reset_fit(self):
        self.coefficients = None
        self.sigma = None
        self.fitted = False
        self.data_mean = None
        self.data_std = None

    def _check_input(self, y):
        """检查输入数据是否有效"""
        if torch.isnan(y).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(y).any():
            raise ValueError("Input contains Inf values")

    def _normalize(self, y):
        """标准化数据"""
        self.data_mean = y.mean(dim=(0, 1), keepdim=True)  # (1, 1, K)
        self.data_std = y.std(dim=(0, 1), keepdim=True) + 1e-8  # (1, 1, K)
        return (y - self.data_mean) / self.data_std

    def _denormalize(self, y):
        """反标准化"""
        if self.data_mean is None or self.data_std is None:
            return y
        return y * self.data_std + self.data_mean

    def _prepare_var_data(self, y):
        B, T, K = y.shape
        p = self.lag_order

        if K != self.enc_in:
            raise ValueError(f"Input feature dim K={K} != enc_in={self.enc_in}")
        if T <= p:
            raise ValueError(f"seq_len T={T} must be larger than lag_order p={p}")

        Y = y[:, p:, :]

        X_parts = [torch.ones(B, T - p, 1, device=y.device, dtype=y.dtype)]
        for i in range(1, p + 1):
            X_parts.append(y[:, p - i:T - i, :])
        X = torch.cat(X_parts, dim=-1)
        return Y, X

    def _ols_estimate_shared(self, Y, X):
        """
        数值稳定的 OLS 估计
        """
        Bsz, Teff, K = Y.shape
        D = X.shape[-1]

        X2 = X.reshape(Bsz * Teff, D)
        Y2 = Y.reshape(Bsz * Teff, K)

        XtX = X2.transpose(0, 1) @ X2
        XtY = X2.transpose(0, 1) @ Y2

        # 方法1: 更强的 Ridge 正则化
        eye = torch.eye(D, device=X.device, dtype=X.dtype)
        XtX_reg = XtX + self.ridge * eye

        # 方法2: 检查条件数，如果太大则增加正则化
        try:
            cond = torch.linalg.cond(XtX_reg)
            if cond > 1e10:
                # 条件数过大，增加正则化
                XtX_reg = XtX + (self.ridge * 10) * eye
        except:
            pass

        # 方法3: 使用 lstsq 代替 solve（更稳定）
        try:
            # lstsq 更加数值稳定
            coef, _, _, _ = torch.linalg.lstsq(XtX_reg, XtY)
        except:
            # 备用方案：使用 solve
            coef = torch.linalg.solve(XtX_reg, XtY)

        # 检查结果是否有效
        if torch.isnan(coef).any() or torch.isinf(coef).any():
            # 如果结果无效，使用更强的正则化重试
            XtX_reg = XtX + eye  # 正则化系数设为 1
            coef = torch.linalg.solve(XtX_reg, XtY)

        # 限制系数范围，防止过大
        coef = torch.clamp(coef, min=-10.0, max=10.0)

        return coef.transpose(0, 1).contiguous()

    def _check_stability(self, A_list):
        """
        检查 VAR 系统的稳定性
        如果特征值模大于1，系统不稳定，需要缩放系数
        """
        K = self.enc_in
        p = self.lag_order

        # 构建伴随矩阵
        companion = torch.zeros(K * p, K * p, device=A_list[0].device, dtype=A_list[0].dtype)

        # 填充第一行块
        for i, A in enumerate(A_list):
            companion[:K, i * K:(i + 1) * K] = A

        # 填充下方的单位矩阵块
        if p > 1:
            companion[K:, :K * (p - 1)] = torch.eye(K * (p - 1), device=companion.device, dtype=companion.dtype)

        # 计算特征值
        try:
            eigenvalues = torch.linalg.eigvals(companion)
            max_eigenvalue = torch.abs(eigenvalues).max().item()

            # 如果不稳定，缩放系数
            if max_eigenvalue > 0.99:
                scale = 0.95 / max_eigenvalue
                return [A * scale for A in A_list], True
        except:
            pass

        return A_list, False

    @torch.no_grad()
    def fit_global(self, x_enc):
        # 检查输入
        self._check_input(x_enc)

        # 标准化
        x_norm = self._normalize(x_enc)

        Y, X = self._prepare_var_data(x_norm)
        coef = self._ols_estimate_shared(Y, X)

        # 残差协方差
        Y_pred = X @ coef.transpose(0, 1)
        residuals = Y - Y_pred
        Bsz, Teff, K = residuals.shape
        R = residuals.reshape(Bsz * Teff, K)

        denom = max(R.shape[0] - (1 + self.enc_in * self.lag_order), 1)
        sigma = (R.transpose(0, 1) @ R) / float(denom)

        self.coefficients = coef
        self.sigma = sigma
        self.fitted = True

    def _extract_A_and_c(self):
        if not self.fitted or self.coefficients is None:
            raise RuntimeError("VAR model is not fitted yet.")

        coef = self.coefficients
        c = coef[:, 0]
        A_list = []
        for i in range(self.lag_order):
            start = 1 + i * self.enc_in
            end = 1 + (i + 1) * self.enc_in
            A_list.append(coef[:, start:end])

        # 检查并确保稳定性
        A_list, was_scaled = self._check_stability(A_list)

        return c, A_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if not self.fitted:
            self.fit_global(x_enc)

        B, T, K = x_enc.shape
        p = self.lag_order
        c, A_list = self._extract_A_and_c()

        # 对输入进行标准化
        x_norm = (x_enc - self.data_mean) / self.data_std

        history = x_norm[:, -p:, :].clone()
        preds = []

        for step in range(self.pred_len):
            y_new = c.unsqueeze(0).expand(B, -1).clone()
            for i, A in enumerate(A_list):
                y_lag = history[:, -(i + 1), :]
                y_new = y_new + (y_lag @ A.transpose(0, 1))

            # 限制预测值范围，防止爆炸
            y_new = torch.clamp(y_new, min=-10.0, max=10.0)

            preds.append(y_new.unsqueeze(1))
            history = torch.cat([history[:, 1:, :], y_new.unsqueeze(1)], dim=1)

        result = torch.cat(preds, dim=1)

        # 反标准化
        result = self._denormalize(result)

        # 最终检查
        if torch.isnan(result).any() or torch.isinf(result).any():
            # 如果仍有问题，返回简单的持续性预测
            result = x_enc[:, -1:, :].expand(B, self.pred_len, K).clone()

        return result

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, :self.seq_len, :]

    def anomaly_detection(self, x_enc):
        out = self.forecast(x_enc, None, None, None)
        return out[:, :self.seq_len, :]

    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError("VAR does not support classification")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        return None