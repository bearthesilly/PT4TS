import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Gaussian Process VAR
    使用高斯过程建模非线性依赖
    适合小样本 + 中低维多元时间序列 baseline
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.lag_order = max(1, int(getattr(configs, 'lag_order', min(5, configs.seq_len // 10))))
        self.jitter = float(getattr(configs, 'gp_jitter', 1e-5))
        self.max_train_points = int(getattr(configs, 'gp_max_train_points', min(self.seq_len, 128)))

        # 每个输出维度一组超参数；log 参数化保证正值
        self.log_length_scale = nn.Parameter(torch.zeros(self.enc_in))
        self.log_signal_var = nn.Parameter(torch.zeros(self.enc_in))
        self.log_noise_var = nn.Parameter(torch.ones(self.enc_in) * -2.0)

        # 与旧接口保持一致；当前实现不依赖 sklearn 对象
        self.gp_models = None

    def _rbf_kernel(self, X1, X2, length_scale, signal_var):
        """
        RBF (Squared Exponential) kernel
        X1: (n1, d), X2: (n2, d)
        """
        length_scale = torch.clamp(length_scale, min=1e-4)
        signal_var = torch.clamp(signal_var, min=1e-6)

        x1_sq = (X1 ** 2).sum(dim=-1, keepdim=True)          # (n1, 1)
        x2_sq = (X2 ** 2).sum(dim=-1).unsqueeze(0)           # (1, n2)
        dist_sq = torch.clamp(x1_sq + x2_sq - 2.0 * (X1 @ X2.T), min=0.0)

        return signal_var * torch.exp(-0.5 * dist_sq / (length_scale ** 2))

    def _prepare_gp_data(self, y):
        """准备 GP 训练数据: X=(batch, T-p, K*p), Y=(batch, T-p, K)"""
        batch_size, T, K = y.shape
        p = self.lag_order

        if T <= p:
            raise ValueError(
                f"seq_len ({T}) must be larger than lag_order ({p}) for GPVAR."
            )

        X_list = []
        for i in range(1, p + 1):
            X_list.append(y[:, p - i:T - i, :])
        X = torch.cat(X_list, dim=-1)
        Y = y[:, p:, :]
        return X, Y

    def _select_recent_train_data(self, X_tr, Y_tr):
        """
        对 exact GP 做一个非常关键的截断：
        仅使用最近的 max_train_points 个训练点，避免 O(N^3) 过大。
        对 small-sample baseline 这是合理且常见的。
        """
        n = X_tr.shape[0]
        if n <= self.max_train_points:
            return X_tr, Y_tr
        return X_tr[-self.max_train_points:], Y_tr[-self.max_train_points:]

    def _build_gp_cache(self, X_tr, Y_tr):
        """
        为单个样本构建 cache：
        - 每个输出维度只做一次 K(X,X)、Cholesky、alpha
        - 预测 horizon 内重复使用
        """
        X_tr, Y_tr = self._select_recent_train_data(X_tr, Y_tr)
        n = X_tr.shape[0]
        device = X_tr.device
        dtype = X_tr.dtype

        eye = torch.eye(n, device=device, dtype=dtype)

        caches = []
        for k in range(self.enc_in):
            length_scale = torch.exp(self.log_length_scale[k])
            signal_var = torch.exp(self.log_signal_var[k])
            noise_var = torch.exp(self.log_noise_var[k])

            K_xx = self._rbf_kernel(X_tr, X_tr, length_scale, signal_var)
            K_xx = K_xx + (noise_var + self.jitter) * eye

            # 数值稳定性保护：逐步增大 jitter
            jitter = 0.0
            L = None
            for _ in range(5):
                try:
                    L = torch.linalg.cholesky(K_xx + jitter * eye)
                    break
                except RuntimeError:
                    jitter = self.jitter if jitter == 0.0 else jitter * 10.0
            if L is None:
                # 最后一次尝试
                L = torch.linalg.cholesky(K_xx + 1e-3 * eye)

            yk = Y_tr[:, k:k + 1]  # (n, 1)
            tmp = torch.linalg.solve_triangular(L, yk, upper=False)
            alpha = torch.linalg.solve_triangular(L.transpose(-1, -2), tmp, upper=True)

            caches.append(
                {
                    "length_scale": length_scale,
                    "signal_var": signal_var,
                    "L": L,
                    "alpha": alpha,
                }
            )

        return X_tr, caches

    def _gp_predict_from_cache(self, X_tr, cache_k, X_test):
        """
        使用缓存做单输出 GP 预测
        X_test: (m, d)
        return:
            mu:  (m, 1)
            var: (m,)
        """
        K_s = self._rbf_kernel(X_tr, X_test, cache_k["length_scale"], cache_k["signal_var"])  # (n, m)
        mu = K_s.transpose(0, 1) @ cache_k["alpha"]  # (m, 1)

        v = torch.linalg.solve_triangular(cache_k["L"], K_s, upper=False)  # (n, m)
        prior_var = cache_k["signal_var"].expand(X_test.shape[0])
        var = torch.clamp(prior_var - (v ** 2).sum(dim=0), min=1e-9)
        return mu, var

    def _predict_one_step_all_dims(self, X_tr, caches, X_test):
        """
        对一个测试输入，同时预测所有输出维度
        X_test: (1, K*p)
        return: y_new (1, K)
        """
        preds = []
        for k in range(self.enc_in):
            mu, _ = self._gp_predict_from_cache(X_tr, caches[k], X_test)
            preds.append(mu)
        return torch.cat(preds, dim=-1)  # (1, K)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        GP-VAR 预测
        保持旧接口不变，输出: (batch, pred_len, K)
        """
        batch_size, T, K = x_enc.shape
        p = self.lag_order

        if K != self.enc_in:
            raise ValueError(
                f"Input feature dim K={K} does not match configs.enc_in={self.enc_in}."
            )
        if T <= p:
            raise ValueError(
                f"Input length T={T} must be larger than lag_order={p}."
            )

        X_train, Y_train = self._prepare_gp_data(x_enc)
        predictions_batch = []

        for b in range(batch_size):
            X_tr = X_train[b]  # (T-p, K*p)
            Y_tr = Y_train[b]  # (T-p, K)

            X_tr, caches = self._build_gp_cache(X_tr, Y_tr)

            history = x_enc[b, -p:, :].clone()  # (p, K)
            preds = []

            for _ in range(self.pred_len):
                # 与原实现保持相同的特征展开顺序
                X_test = history.flip(0).reshape(1, -1)  # (1, K*p)

                y_new = self._predict_one_step_all_dims(X_tr, caches, X_test)  # (1, K)
                preds.append(y_new)

                history = torch.cat([history[1:, :], y_new], dim=0)

            predictions_batch.append(torch.cat(preds, dim=0))

        return torch.stack(predictions_batch, dim=0)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # 保持接口兼容；GPVAR baseline 主要用于 forecast
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, :self.seq_len, :]

    def anomaly_detection(self, x_enc):
        return self.forecast(x_enc, None, None, None)[:, :self.seq_len, :]

    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError("GPVAR does not support classification")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        return None