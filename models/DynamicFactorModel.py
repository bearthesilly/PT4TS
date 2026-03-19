import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class DynamicFactorModel(nn.Module):
    """
    Dynamic Factor Model
    通过低维潜在因子建模高维时间序列
    特别适合变量数较多但样本量有限的情况
    """

    def __init__(self, configs):
        super(DynamicFactorModel, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 因子数量（通常远小于变量数）
        self.n_factors = getattr(configs, 'n_factors', max(1, self.enc_in // 3))
        self.factor_lag = getattr(configs, 'factor_lag', min(4, configs.seq_len // 10))
        
        # 模型参数
        self.Lambda = None  # 因子载荷矩阵 (K, r)
        self.Phi = None     # 因子动态系数 (r, r * factor_lag)
        self.Sigma_eps = None  # 观测噪声方差
        self.Sigma_eta = None  # 因子噪声方差

    def _extract_factors_pca(self, y):
        """
        使用 PCA 提取潜在因子
        y: (batch, T, K)
        """
        batch_size, T, K = y.shape
        r = self.n_factors
        
        # 标准化
        y_mean = y.mean(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True) + 1e-8
        y_norm = (y - y_mean) / y_std
        
        factors_batch = []
        loadings_batch = []
        
        for b in range(batch_size):
            # SVD 分解
            U, S, Vh = torch.linalg.svd(y_norm[b], full_matrices=False)
            
            # 取前 r 个因子
            factors = U[:, :r] * S[:r].unsqueeze(0)  # (T, r)
            loadings = Vh[:r, :].T  # (K, r)
            
            factors_batch.append(factors)
            loadings_batch.append(loadings)
        
        factors = torch.stack(factors_batch, dim=0)  # (batch, T, r)
        loadings = torch.stack(loadings_batch, dim=0)  # (batch, K, r)
        
        return factors, loadings, y_mean, y_std

    def _estimate_factor_var(self, factors):
        """
        估计因子的 VAR 动态
        factors: (batch, T, r)
        """
        batch_size, T, r = factors.shape
        p = self.factor_lag
        
        # 构建 VAR 数据
        Y = factors[:, p:, :]  # (batch, T-p, r)
        X_list = [torch.ones(batch_size, T - p, 1, device=factors.device)]
        for i in range(1, p + 1):
            X_list.append(factors[:, p - i:T - i, :])
        X = torch.cat(X_list, dim=-1)  # (batch, T-p, r*p + 1)
        
        # OLS 估计
        XtX = torch.bmm(X.transpose(1, 2), X)
        XtY = torch.bmm(X.transpose(1, 2), Y)
        reg = 1e-6 * torch.eye(XtX.shape[-1], device=XtX.device).unsqueeze(0)
        Phi = torch.linalg.solve(XtX + reg, XtY)  # (batch, r*p+1, r)
        
        # 残差协方差
        Y_pred = torch.bmm(X, Phi)
        residuals = Y - Y_pred
        Sigma_eta = torch.bmm(residuals.transpose(1, 2), residuals) / (T - p)
        
        return Phi.transpose(1, 2), Sigma_eta  # Phi: (batch, r, r*p+1)

    def fit(self, x_enc):
        """拟合动态因子模型"""
        # 提取因子
        factors, self.Lambda, self.y_mean, self.y_std = self._extract_factors_pca(x_enc)
        
        # 估计因子动态
        self.Phi, self.Sigma_eta = self._estimate_factor_var(factors)
        
        # 估计观测噪声
        y_reconstructed = torch.bmm(factors, self.Lambda.transpose(1, 2))
        y_norm = (x_enc - self.y_mean) / self.y_std
        residuals = y_norm - y_reconstructed
        self.Sigma_eps = (residuals ** 2).mean(dim=1)  # (batch, K)
        
        self.factors = factors

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """预测"""
        batch_size = x_enc.shape[0]
        device = x_enc.device
        
        self.fit(x_enc)
        
        r = self.n_factors
        p = self.factor_lag
        
        # 提取因子 VAR 系数
        c = self.Phi[:, :, 0:1]  # (batch, r, 1)
        A_list = []
        for i in range(p):
            start_idx = 1 + i * r
            end_idx = 1 + (i + 1) * r
            if end_idx <= self.Phi.shape[-1]:
                A_list.append(self.Phi[:, :, start_idx:end_idx])
        
        # 预测因子
        factor_history = self.factors[:, -p:, :].clone()
        factor_predictions = []
        
        for t in range(self.pred_len):
            f_new = c.squeeze(-1).clone()
            for i, A in enumerate(A_list):
                if i < factor_history.shape[1]:
                    f_lag = factor_history[:, -(i + 1), :]
                    f_new = f_new + torch.bmm(A, f_lag.unsqueeze(-1)).squeeze(-1)
            
            factor_predictions.append(f_new.unsqueeze(1))
            factor_history = torch.cat([factor_history[:, 1:, :], f_new.unsqueeze(1)], dim=1)
        
        factor_pred = torch.cat(factor_predictions, dim=1)  # (batch, pred_len, r)
        
        # 转换回观测空间
        y_pred_norm = torch.bmm(factor_pred, self.Lambda.transpose(1, 2))  # (batch, pred_len, K)
        y_pred = y_pred_norm * self.y_std + self.y_mean
        
        return y_pred

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, :self.seq_len, :]

    def anomaly_detection(self, x_enc):
        return self.forecast(x_enc, None, None, None)[:, :self.seq_len, :]

    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError("DynamicFactorModel does not support classification")

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