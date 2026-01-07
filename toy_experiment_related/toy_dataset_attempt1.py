import os
import numpy as np
import pandas as pd

def generate_prior_data(num_samples=2000, seq_len=192, noise_level=0.05):
    """
    生成包含 ST-PT 关键 Priors 的独立同分布(IID)时间序列样本。
    
    Output Shape: (num_samples, seq_len, 10)
    Channels: 10
    """
    print(f"Generating {num_samples} samples with Length={seq_len}...")
    
    data_list = []
    mask_list = []
    
    for _ in range(num_samples):
        # ---------------------------------------------------------
        # 单样本生成逻辑 (复用原 ToyDatasetV3 的核心逻辑)
        # ---------------------------------------------------------
        L = seq_len
        # 时间轴：覆盖约 8 个周期 (设 24 steps 为一个周期)
        t = np.linspace(0, 8 * 2 * np.pi, L) 
        
        channels = np.zeros((L, 10))
        event_mask = np.zeros((L, 1), dtype=int) 
        
        # --- Group A: Periodicity & Trend (Prior 1, 2) ---
        # Ch0: Strong Periodicity
        channels[:, 0] = np.sin(t) + 0.5 * np.cos(2 * t)

        # Ch1: Sigmoid Trend with Random Saturation
        # 随机让转折点发生在序列的中间区域
        t0_idx = np.random.randint(int(L * 0.15), int(L * 0.35)) 
        t0 = t[t0_idx] 
        channels[:, 1] = 2.0 / (1 + np.exp(-1.5 * (t - t0))) - 1.0

        # Ch2: Non-linear Trend + Seasonality
        channels[:, 2] = 0.05 * np.arange(L) + np.sin(t)

        # --- Group B: Locality & Inertia (Prior 4, 7) ---
        # Ch3: AR(1) Strong Inertia
        ar1 = np.zeros(L)
        ar1[0] = np.random.randn()
        for i in range(1, L):
            ar1[i] = 0.90 * ar1[i-1] + np.random.normal(0, 0.1)
        channels[:, 3] = ar1

        # Ch4: AR(2) with cyclical dependency
        ar2 = np.zeros(L)
        ar2[0:2] = np.random.randn(2)
        for i in range(2, L):
            ar2[i] = 1.2 * ar2[i-1] - 0.4 * ar2[i-2] + np.random.normal(0, 0.15)
        channels[:, 4] = ar2

        # --- Group C: Special Events (Prior 8) ---
        # Ch5: Regime Switch
        # 随机设定事件发生时间，偏向于后半段
        event_idx = np.random.randint(int(L * 0.4), int(L * 0.8))
        event_duration = 12 
        
        base_signal = np.sin(t)
        channels[:, 5] = base_signal
        
        if event_idx + event_duration < L:
            channels[event_idx : event_idx+event_duration, 5] += 2.0
            event_mask[event_idx : event_idx+event_duration, 0] = 1

        # --- Group D: Causal & Independent (Prior 3, 5, 6) ---
        # Ch6 (Source) & Ch7 (Target)
        source = np.sin(t) * np.exp(0.01 * np.arange(L)) 
        channels[:, 6] = source
        
        lag = 8
        channels[lag:, 7] = 0.8 * source[:-lag]
        channels[:lag, 7] = source[0] 
        
        # Ch8, Ch9: Independent Group (High Frequency)
        fast_wave = np.sin(t * 8) 
        channels[:, 8] = fast_wave
        channels[:, 9] = np.sign(fast_wave) 

        # --- Final Processing ---
        # Add Noise
        noise = np.random.normal(0, noise_level, (L, 10))
        sample_data = channels + noise
        
        data_list.append(sample_data)
        mask_list.append(event_mask)

    # 转换为 Numpy 数组
    data_array = np.array(data_list).astype(np.float32) # (N, L, 10)
    mask_array = np.array(mask_list).astype(np.int32)   # (N, L, 1)
    
    return data_array, mask_array

if __name__ == "__main__":
    # 1. 定义保存路径
    save_dir = './dataset/toy_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 2. 设定参数
    # 如果要适配 seq_len=96, pred_len=96，总长度建议保持 192 或更长
    SEQ_LEN = 192 
    NUM_SAMPLES = 200
    
    print("Starting data generation...")
    data, masks = generate_prior_data(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN)
    
    # 3. 保存为 NPY 文件
    save_path_data = os.path.join(save_dir, 'toy_data_attempt1.npy')
    save_path_mask = os.path.join(save_dir, 'toy_mask_attempt1.npy')
    
    np.save(save_path_data, data)
    np.save(save_path_mask, masks)
    
    print(f"Data Generation Completed!")
    print(f"Main Data Shape: {data.shape}")
    print(f"Saved to: {save_path_data}")
    print(f"Event Masks Saved to: {save_path_mask}")
    
    # 4. 可视化检查 (Optional)
    # 取第一个样本的前5个通道看看
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for c in range(10):
            plt.plot(data[0, :, c], label=f'Ch{c}')
        plt.title(f"Sample 0 Visualization (Length {SEQ_LEN})")
        plt.legend()
        plt.tight_layout()

    except ImportError:
        pass