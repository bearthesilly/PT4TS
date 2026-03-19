import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class BVAR(nn.Module):
    """
    Bayesian Vector Autoregression with Minnesota Prior
    特别适合小样本场景
    """

    def __init__(self, configs):
        super(BVAR, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.lag_order = getattr(configs, 'lag_order', min(8, configs.seq_len // 5))
        
        # Minnesota 先验超参数
        self.lambda_overall = getattr(configs, 'bvar_lambda', 0.1)  # 整体紧缩
        self.lambda_lag = getattr(configs, 'bvar_lag_decay', 1.0)  # 滞后衰减
        self.lambda_cross = getattr(configs, 'bvar_cross', 0.5)  # 跨变量紧缩
        
        self.coefficients = None
        self.sigma = None

    def _minnesota_prior(self, y, device):
        """
        构建 Minnesota 先验
        返回先验均值和先验精度矩阵
        """
        K = self.enc_in
        p = self.lag_order
        n_params = K * p + 1  # 每个方程的参数数
        
        # 估计各变量的方差（用于调整先验）
        var_estimates = y.var(dim=1, keepdim=True).mean(dim=0).squeeze()  # (K,)
        
        # 先验均值：自回归系数为单位阵，其他为0
        prior_mean = torch.zeros(K, n_params, device=device)
        # 第一个滞后的自回归系数设为接近1（单位根先验）
        for k in range(K):
            prior_mean[k, 1 + k] = 0.9  # y_{k,t-1} 的系数
        
        # 先验精度（方差的倒数）
        prior_precision = torch.zeros(K, n_params, device=device)
        
        # 常数项的先验方差
        prior_precision[:, 0] = 1.0 / (self.lambda_overall ** 2 * 10)
        
        for i in range(p):
            for j in range(K):
                idx = 1 + i * K + j
                if idx < n_params:
                    # 自身变量的滞后
                    if j < K:
                        own_var = var_estimates[j] if j < len(var_estimates) else 1.0
                        for k in range(K):
                            target_var = var_estimates[k] if k < len(var_estimates) else 1.0
                            if j == k:  # 自身滞后
                                prior_var = (self.lambda_overall / (i + 1) ** self.lambda_lag) ** 2
                            else:  # 跨变量滞后
                                prior_var = (self.lambda_overall * self.lambda_cross / (i + 1) ** self.lambda_lag) ** 2
                                prior_var *= (target_var / own_var) if own_var > 0 else 1.0
                            prior_precision[k, idx] = 1.0 / prior_var
        
        # 确保数值稳定
        prior_precision = torch.clamp(prior_precision, min=1e-6, max=1e6)
        
        return prior_mean, prior_precision

    def _bayesian_estimate(self, Y, X, prior_mean, prior_precision):
        """
        贝叶斯估计：结合先验和似然
        """
        batch_size = Y.shape[0]
        K = Y.shape[-1]
        
        # 对每个方程分别估计
        coefficients = []
        
        for k in range(K):
            y_k = Y[:, :, k:k+1]  # (batch, T-p, 1)
            
            # OLS 部分
            XtX = torch.bmm(X.transpose(1, 2), X)  # (batch, n_params, n_params)
            Xty = torch.bmm(X.transpose(1, 2), y_k)  # (batch, n_params, 1)
            
            # 先验精度矩阵（对角）
            V0_inv = torch.diag(prior_precision[k]).unsqueeze(0).expand(batch_size, -1, -1)
            b0 = prior_mean[k].unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
            
            # 后验精度和均值
            V_post_inv = XtX + V0_inv
            reg = 1e-5 * torch.eye(V_post_inv.shape[-1], device=V_post_inv.device).unsqueeze(0)
            V_post_inv = V_post_inv + reg
            
            b_post = torch.linalg.solve(V_post_inv, Xty + torch.bmm(V0_inv, b0))
            
            coefficients.append(b_post.squeeze(-1))  # (batch, n_params)
        
        return torch.stack(coefficients, dim=1)  # (batch, K, n_params)

    def fit(self, x_enc):
        """拟合 BVAR 模型"""
        batch_size, T, K = x_enc.shape
        p = self.lag_order
        device = x_enc.device
        
        # 准备数据
        Y = x_enc[:, p:, :]
        X_list = [torch.ones(batch_size, T - p, 1, device=device)]
        for i in range(1, p + 1):
            X_list.append(x_enc[:, p - i:T - i, :])
        X = torch.cat(X_list, dim=-1)
        
        # 获取先验
        prior_mean, prior_precision = self._minnesota_prior(x_enc, device)
        
        # 贝叶斯估计
        self.coefficients = self._bayesian_estimate(Y, X, prior_mean, prior_precision)
        
        # 估计残差协方差
        Y_pred = torch.bmm(X, self.coefficients.transpose(1, 2))
        residuals = Y - Y_pred
        T_eff = residuals.shape[1]
        self.sigma = torch.bmm(residuals.transpose(1, 2), residuals) / T_eff

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """多步预测"""
        batch_size = x_enc.shape[0]
        device = x_enc.device
        
        self.fit(x_enc)
        
        # 提取系数
        c = self.coefficients[:, :, 0:1]
        A_list = []
        for i in range(self.lag_order):
            start_idx = 1 + i * self.enc_in
            end_idx = 1 + (i + 1) * self.enc_in
            A_list.append(self.coefficients[:, :, start_idx:end_idx])
        
        # 迭代预测
        history = x_enc[:, -self.lag_order:, :].clone()
        predictions = []
        
        for t in range(self.pred_len):
            y_new = c.squeeze(-1).clone()
            for i, A in enumerate(A_list):
                y_lag = history[:, -(i + 1), :]
                y_new = y_new + torch.bmm(A, y_lag.unsqueeze(-1)).squeeze(-1)
            
            predictions.append(y_new.unsqueeze(1))
            history = torch.cat([history[:, 1:, :], y_new.unsqueeze(1)], dim=1)
        
        return torch.cat(predictions, dim=1)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, :self.seq_len, :]

    def anomaly_detection(self, x_enc):
        return self.forecast(x_enc, None, None, None)[:, :self.seq_len, :]

    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError("BVAR does not support classification")

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