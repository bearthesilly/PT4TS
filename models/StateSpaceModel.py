import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Local Level State Space Model with Kalman Filter
    适合捕捉时间序列的趋势和季节性
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 状态维度（level + trend for each variable）
        self.state_dim = self.enc_in * 2
        
        # 可学习的噪声参数
        self.log_sigma_state = nn.Parameter(torch.zeros(self.enc_in))
        self.log_sigma_obs = nn.Parameter(torch.zeros(self.enc_in))

    def _build_matrices(self, batch_size, device):
        """构建状态空间矩阵"""
        K = self.enc_in
        
        # 状态转移矩阵 T (local linear trend)
        # state = [level_1, trend_1, level_2, trend_2, ...]
        T = torch.zeros(self.state_dim, self.state_dim, device=device)
        for i in range(K):
            T[2*i, 2*i] = 1.0      # level
            T[2*i, 2*i + 1] = 1.0  # level += trend
            T[2*i + 1, 2*i + 1] = 1.0  # trend
        T = T.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 观测矩阵 Z
        Z = torch.zeros(K, self.state_dim, device=device)
        for i in range(K):
            Z[i, 2*i] = 1.0  # 观测 = level
        Z = Z.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 状态噪声协方差 Q
        sigma_state = torch.exp(self.log_sigma_state)
        Q = torch.zeros(batch_size, self.state_dim, self.state_dim, device=device)
        for i in range(K):
            Q[:, 2*i, 2*i] = sigma_state[i] ** 2
            Q[:, 2*i + 1, 2*i + 1] = (sigma_state[i] * 0.1) ** 2
        
        # 观测噪声协方差 R
        sigma_obs = torch.exp(self.log_sigma_obs)
        R = torch.diag(sigma_obs ** 2).unsqueeze(0).expand(batch_size, -1, -1)
        
        return T, Z, Q, R

    def _kalman_filter(self, y, T, Z, Q, R):
        """
        Kalman Filter 前向传播
        y: (batch, T, K)
        """
        batch_size, seq_len, K = y.shape
        device = y.device
        
        # 初始化
        alpha = torch.zeros(batch_size, self.state_dim, device=device)
        P = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1) * 1e4
        
        filtered_states = []
        filtered_covs = []
        
        for t in range(seq_len):
            # 预测步骤
            alpha_pred = torch.bmm(T, alpha.unsqueeze(-1)).squeeze(-1)
            P_pred = torch.bmm(torch.bmm(T, P), T.transpose(1, 2)) + Q
            
            # 更新步骤
            y_t = y[:, t, :]  # (batch, K)
            
            # 观测预测
            y_pred = torch.bmm(Z, alpha_pred.unsqueeze(-1)).squeeze(-1)
            
            # 创新及其协方差
            v = y_t - y_pred
            F = torch.bmm(torch.bmm(Z, P_pred), Z.transpose(1, 2)) + R
            
            # Kalman 增益
            F_inv = torch.linalg.inv(F + 1e-6 * torch.eye(K, device=device).unsqueeze(0))
            K_gain = torch.bmm(torch.bmm(P_pred, Z.transpose(1, 2)), F_inv)
            
            # 更新
            alpha = alpha_pred + torch.bmm(K_gain, v.unsqueeze(-1)).squeeze(-1)
            P = P_pred - torch.bmm(torch.bmm(K_gain, Z), P_pred)
            
            filtered_states.append(alpha.unsqueeze(1))
            filtered_covs.append(P.unsqueeze(1))
        
        filtered_states = torch.cat(filtered_states, dim=1)  # (batch, T, state_dim)
        
        return filtered_states, alpha, P

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """使用 Kalman Filter 预测"""
        batch_size = x_enc.shape[0]
        device = x_enc.device
        
        # 构建状态空间矩阵
        T, Z, Q, R = self._build_matrices(batch_size, device)
        
        # Kalman Filter
        _, alpha, P = self._kalman_filter(x_enc, T, Z, Q, R)
        
        # 多步预测
        predictions = []
        for t in range(self.pred_len):
            # 状态预测
            alpha = torch.bmm(T, alpha.unsqueeze(-1)).squeeze(-1)
            
            # 观测预测
            y_pred = torch.bmm(Z, alpha.unsqueeze(-1)).squeeze(-1)
            predictions.append(y_pred.unsqueeze(1))
        
        return torch.cat(predictions, dim=1)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, :self.seq_len, :]

    def anomaly_detection(self, x_enc):
        return self.forecast(x_enc, None, None, None)[:, :self.seq_len, :]

    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError("StateSpaceModel does not support classification")

    @torch.no_grad()
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