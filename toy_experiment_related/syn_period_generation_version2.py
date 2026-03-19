import numpy as np
import os
import matplotlib.pyplot as plt

class PeriodicityGenerator:
    def __init__(self, num_samples=2000, sample_len=200, high_noise=True):
        """
        Args:
            num_samples (N): 样本总数 (建议尝试极小样本，如 100-300)
            sample_len (T): 样本长度
            high_noise (bool): 是否启用高噪声模式 (低信噪比)
        """
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.num_channels = 10 
        self.high_noise = high_noise
        
        # 基础振幅
        self.amp = 1.0
        
        # 噪声系数：如果是高噪声模式，由于信号振幅是1，噪声标准差设为 0.8-1.2 将极具挑战性
        self.noise_std = 1.0 if high_noise else 0.3
        self.red_noise_alpha = 0.95 # 更强的自相关，模拟更难区分的趋势噪声

    def _red_noise(self, length, alpha, scale):
        """生成红噪声 (随机游走趋势)"""
        noise = np.zeros(length)
        white = np.random.normal(0, scale, length)
        noise[0] = white[0]
        # 累积过程
        for t in range(1, length):
            noise[t] = alpha * noise[t-1] + np.sqrt(1 - alpha**2) * white[t]
        return noise

    def _generate_single_sample(self):
        L = self.sample_len
        channels = np.zeros((L, self.num_channels))
        t = np.arange(L)
        
        # === 随机相位 Shift ===
        # 关键：每个样本的相位独立。
        # 当 Look-back window 很短时 (e.g. 16)，不同的样本会看到正弦波的截然不同的局部 (有的在波峰，有的在上升沿)
        phase = np.random.uniform(0, 2*np.pi)
        
        # ==========================================
        # Group A: 基础周期 & 长周期 (此时观测窗口可能 < 周期)
        # ==========================================
        # Ch0: Standard Period = 24
        channels[:, 0] = self.amp * np.sin(2 * np.pi * t / 24 + phase)
        
        # Ch1: Long Period = 72
        # 如果 seq_len=48，模型永远无法在输入中看到一个完整的周期。
        # 这测试模型是否能理解 "弧线" 其实是 "圆的一部分"。
        channels[:, 1] = self.amp * np.cos(2 * np.pi * t / 72 + phase)
        
        # Ch2: High Freq = 12 (较容易捕捉)
        channels[:, 2] = self.amp * np.sin(2 * np.pi * t / 12 + phase)

        # ==========================================
        # Group B: 带有线性漂移的周期 (Trend + Seasonality)
        # ==========================================
        # 随机斜率，有时向上，有时向下，模拟非平稳
        slope = np.random.uniform(-0.01, 0.01)
        trend = slope * t
        
        # Ch3: P24 + Linear Trend
        channels[:, 3] = channels[:, 0] + trend

        # Ch4: P24 + P12 + Trend
        channels[:, 4] = channels[:, 0] + 0.5 * channels[:, 2] + trend

        # ==========================================
        # Group C: 高噪声抗干扰 (Robustness) - 重点!!
        # ==========================================
        # 这里的噪声不再只是微扰，而是可能淹没信号
        
        # Ch5: P24 + White Noise (高方差)
        channels[:, 5] = channels[:, 0] + np.random.normal(0, self.noise_std, L)
        
        # Ch6: P24 + Red Noise (强干扰)
        # 红噪声看起来像 "局部趋势"，会让模型误以为是数据本身的走势
        rn = self._red_noise(L, self.red_noise_alpha, self.noise_std)
        channels[:, 6] = channels[:, 0] + rn

        # Ch7: Long Period (P72) + Red Noise
        # 最难的情况：周期很长(看不全) + 噪声很像趋势。模型极易产生误判。
        channels[:, 7] = channels[:, 1] + rn

        # ==========================================
        # Group D: 极低信噪比 (Signal buried in Noise)
        # ==========================================
        # Ch8: Weak P24 (Amplitude 0.5) + Strong Noise
        channels[:, 8] = 0.5 * np.sin(2 * np.pi * t / 24 + phase) + \
                         np.random.normal(0, self.noise_std, L)

        # Ch9: 纯红噪声 (无信号，作为对照)
        channels[:, 9] = self._red_noise(L, self.red_noise_alpha, self.noise_std)
        
        return channels

    def generate(self, save_path):
        print(f"Generating {self.num_samples} samples | Length: {self.sample_len} | High Noise: {self.high_noise}")
        
        all_samples = []
        for i in range(self.num_samples):
            sample = self._generate_single_sample()
            all_samples.append(sample)
        
        final_data = np.stack(all_samples, axis=0).astype(np.float32)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, final_data)
        
        print(f"Dataset saved to: {save_path}")
        print(f"Shape: {final_data.shape}")
        return final_data

if __name__ == "__main__":
    # === 实验配置建议 ===
    # 1. num_samples: 设为 100 或 200，模拟 "Small Data"
    # 2. high_noise: True，模拟 "Noisy Data"
    # 3. 训练由于 seq_len 应该设为较小值 (e.g. 16, 24, 32)，以此测试 "Partial View"
    
    gen = PeriodicityGenerator(num_samples=150, sample_len=200, high_noise=True)
    
    # 命名体现难度
    save_path = './dataset/syn_data/periodicity_N150_HighNoise.npy'
    data = gen.generate(save_path=save_path)
    
    # 可视化前两个样本的 Ch6 (P24 + RedNoise) 以确认难度
    # 你会发现如果不画出原始的正弦波，肉眼几乎看不出规律，这正是我们要的
    plt.figure(figsize=(10, 4))
    plt.plot(data[0, :100, 6], label='Sample 0 (Noisy P24)', alpha=0.8)
    plt.plot(np.sin(2*np.pi*np.arange(100)/24), label='Ground Truth P24', linestyle='--', alpha=0.5, color='black')
    plt.title('Visualization of High Noise Channel 6 (First 100 steps)')
    plt.legend()
    # plt.show() # 如果在远程服务器，请注释此行并保存图片
    plt.savefig('noise_preview.png')
    print("Preview saved to noise_preview.png")