import os
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt

def generate_hybrid_data(num_samples=1000, seq_len=192):
    """
    V6.0 Hybrid Dataset Generator (Numpy + Statsmodels)
    
    Output Shape: (num_samples, seq_len, 10)
    Channels: 10
    """
    print(f"Generating {num_samples} samples using Numpy + Statsmodels...")
    
    data_list = []
    mask_list = []
    
    for _ in range(num_samples):
        # ---------------------------------------------------------
        # 单样本生成逻辑 (复用 HybridToyDataset 的核心逻辑)
        # ---------------------------------------------------------
        L = seq_len
        # 时间轴：8个周期 (192 / 24 = 8)
        t = np.linspace(0, 8 * 2 * np.pi, L)
        
        channels = np.zeros((L, 10))
        event_mask = np.zeros((L, 1), dtype=int)
        
        # ============================================================
        # Group A: Periodicity & Trend (Prior 1 & 2)
        # 混合模式: Numpy Signal + Statsmodels Red Noise
        # ============================================================
        
        # Ch0: Periodicity with Red Noise
        # 信号：Numpy 生成标准正弦
        signal_base = np.sin(t)
        # 噪声：Statsmodels 生成红噪声 (AR(1) process)
        # ar=[1, -0.9] 意味着 X_t = 0.9 * X_{t-1} + e_t (低频噪声，比白噪声难预测)
        red_noise = arma_generate_sample(ar=[1, -0.9], ma=[1], nsample=L, scale=0.3)
        channels[:, 0] = signal_base + red_noise

        # Ch1: Sigmoid Trend (Global)
        # 趋势：Numpy 生成
        # 随机让转折点发生在 Input 或 Output 阶段
        shift = np.random.uniform(0.5, 5.0)
        t_norm = np.linspace(-6, 6, L) + shift
        trend = 2.0 / (1 + np.exp(-t_norm)) - 1.0
        # 叠加一点白噪声
        channels[:, 1] = trend + np.random.normal(0, 0.05, L)

        # ============================================================
        # Group B: Locality & Inertia (Prior 4 & 7)
        # 纯 Statsmodels 领域：生成严格的随机过程
        # ============================================================
        
        # Ch2: AR(1) High Inertia (Random Walk-ish)
        # X_t = 0.98 * X_{t-1} + e_t
        # 这种数据没有固定形状，完全靠"惯性"
        # Locality Prior 应该能极大帮助捕捉这种惯性
        channels[:, 2] = arma_generate_sample(ar=[1, -0.98], ma=[1], nsample=L, scale=0.15)

        # Ch3: AR(2) Stochastic Oscillation
        # X_t = 1.4 * X_{t-1} - 0.7 * X_{t-2} + e_t
        # 这是一种"随机周期性"，不同于 Ch0 的"确定性周期性"
        channels[:, 3] = arma_generate_sample(ar=[1, -1.4, 0.7], ma=[1], nsample=L, scale=0.2)

        # ============================================================
        # Group C: Causal Delay & Structure (Prior 5 & 6)
        # ============================================================
        
        # Ch4 (Source): Chirp Signal (变频信号)
        # 频率随时间线性增加 sin(t^2)
        # 这是对 Transfer Learning 的考验：Output 的频率比 Input 高
        t_chirp = np.linspace(0, 3.5, L)
        source = np.sin(np.pi * t_chirp**2)
        channels[:, 4] = source + np.random.normal(0, 0.05, L)

        # Ch5 (Target): Delayed Response
        # Lag = 12 steps (半个周期)
        lag = 12
        target = np.zeros(L)
        target[lag:] = 0.8 * channels[:-lag, 4] # 因果传递
        target[:lag] = channels[0, 4] # padding
        channels[:, 5] = target + np.random.normal(0, 0.05, L)

        # Ch6, Ch7: Independent Block (ARMA Process)
        # 这两路信号互相关，但与 Ch0-5 独立
        # 使用 Statsmodels 生成一个共享的驱动力
        shared_driver = arma_generate_sample(ar=[1, -0.8], ma=[1, 0.5], nsample=L, scale=0.1)
        channels[:, 6] = shared_driver + np.random.normal(0, 0.05, L)
        channels[:, 7] = -shared_driver + np.random.normal(0, 0.05, L) # 负相关

        # ============================================================
        # Group D: Special Events (Prior 8)
        # ============================================================
        
        # Ch8: Event in Output Window
        # 基底：慢速波
        base_c8 = np.cos(t * 0.5)
        channels[:, 8] = base_c8
        
        # 强制事件发生在 Output 窗口 (96 ~ 192)
        # 比如 t=120 时发生促销
        event_start = np.random.randint(110, 150)
        event_len = 16
        
        if event_start + event_len < L:
            channels[event_start : event_start+event_len, 8] += 2.0
            event_mask[event_start : event_start+event_len, 0] = 1
            
        # Ch9: White Noise Control
        channels[:, 9] = np.random.normal(0, 0.5, L)

        data_list.append(channels)
        mask_list.append(event_mask)

    # 转换为 Numpy 数组
    data_array = np.array(data_list).astype(np.float32) 
    mask_array = np.array(mask_list).astype(np.int32)
    
    return data_array, mask_array

if __name__ == "__main__":
    # 1. 定义保存路径
    save_dir = './dataset/toy_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 2. 设定参数
    # 适配 run.sh 中的 seq_len=96 + pred_len=96 = 192
    SEQ_LEN = 192 
    NUM_SAMPLES = 2000
    
    print("Starting Hybrid Data Generation...")
    data, masks = generate_hybrid_data(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN)
    
    # 3. 保存为 NPY 文件
    save_path_data = os.path.join(save_dir, 'toy_data_attempt2.npy')
    save_path_mask = os.path.join(save_dir, 'toy_mask_attempt2.npy')
    
    np.save(save_path_data, data)
    np.save(save_path_mask, masks)
    
    print(f"Data Generation Completed!")
    print(f"Main Data Shape: {data.shape}")
    print(f"Saved to: {save_path_data}")
