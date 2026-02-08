import numpy as np
import os
import matplotlib.pyplot as plt

class PeriodicityGenerator:
    def __init__(self, num_samples=2000, sample_len=200):
        """
        生成独立同分布 (IID) 的周期性数据样本，适配 data_loader.py 中的 Toy_Dataset 类。
        
        Args:
            num_samples (N): 生成的样本总数
            sample_len (T): 每个样本的时间步长 (通常需要 > seq_len + pred_len)
        """
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.num_channels = 10 # 生成 10 个特征通道
        
    def _red_noise(self, length, alpha=0.9, scale=0.2):
        """生成红噪声 (Autoregressive Noise)"""
        # 红噪声序列，模拟真实的随机游走趋势
        noise = np.zeros(length)
        white = np.random.normal(0, scale, length)
        noise[0] = white[0]
        for t in range(1, length):
            noise[t] = alpha * noise[t-1] + np.sqrt(1 - alpha**2) * white[t]
        return noise

    def _generate_single_sample(self):
        """生成单个 (T, C) 形状的样本，具有独立的随机相位"""
        L = self.sample_len
        channels = np.zeros((L, self.num_channels))
        t = np.arange(L) # 时间步 0 到 L-1
        
        # === 关键：每个样本拥有独立的随机相位，互不干扰 ===
        phase_shift = np.random.uniform(0, 2*np.pi)
        
        # ==========================================
        # Group A: 标准周期 (Base Check)
        # ==========================================
        # Ch0: Period = 24 (Base)
        channels[:, 0] = np.sin(2 * np.pi * t / 24 + phase_shift)
        
        # Ch1: Period = 12 (High Freq)
        channels[:, 1] = np.sin(2 * np.pi * t / 12 + phase_shift)
        
        # Ch2: Period = 48 (Low Freq)
        channels[:, 2] = np.cos(2 * np.pi * t / 48 + phase_shift)

        # ==========================================
        # Group B: 复杂叠加 (Harmonics)
        # ==========================================
        # Ch3: Period 24 + Period 12 (基波 + 二次谐波)
        channels[:, 3] = np.sin(2 * np.pi * t / 24 + phase_shift) + \
                         0.5 * np.sin(2 * np.pi * t / 12 + phase_shift)

        # Ch4: 拍频现象 (Beating) - P24 + P20
        # 拍频的包络取决于频率差，不加 phase_shift 保持拍频形态一致性，或者加上也可以
        channels[:, 4] = np.sin(2 * np.pi * t / 24) + np.sin(2 * np.pi * t / 20)

        # ==========================================
        # Group C: 抗噪测试 (Robustness)
        # ==========================================
        # Ch5: Period 24 + White Noise
        channels[:, 5] = channels[:, 0] + np.random.normal(0, 0.3, L)
        
        # Ch6: Period 24 + Red Noise
        # 每个样本的噪声也是独立的
        channels[:, 6] = channels[:, 0] + self._red_noise(L, alpha=0.9, scale=0.4)

        # Ch7: Harmonics (P24+P12) + Red Noise
        channels[:, 7] = channels[:, 3] + self._red_noise(L, alpha=0.9, scale=0.4)

        # ==========================================
        # Group D: 复杂叠加抗噪测试
        # ==========================================
        # Ch8: Beating (P24 + P20) + Red Noise
        channels[:, 8] = channels[:, 4] + self._red_noise(L, alpha=0.9, scale=0.4)

        # Ch9: 纯红噪声 (Control/Target) - 对应 'OT'
        channels[:, 9] = self._red_noise(L, alpha=0.9, scale=0.5)
        
        return channels

    def generate(self, save_path='./dataset/syn_data/toy_iid.npy'):
        print(f"Generating {self.num_samples} independent samples of length {self.sample_len}...")
        
        all_samples = []
        for i in range(self.num_samples):
            sample = self._generate_single_sample()
            all_samples.append(sample)
            if (i+1) % 500 == 0:
                print(f"Generated {i+1}/{self.num_samples} samples...")
        
        # 堆叠成 (N, T, C)
        final_data = np.stack(all_samples, axis=0).astype(np.float32)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存为 .npy 文件
        np.save(save_path, final_data)
        
        print(f"Synthesized IID dataset saved to: {save_path}")
        print(f"Data Shape: {final_data.shape} (N_samples, Sequence_Length, Channels)")
        
        return final_data

if __name__ == "__main__":
    # 配置生成参数
    # 如果你的 seq_len=96, pred_len=96，那么 sample_len 至少要是 192 (建议稍微大一点，如 200)
    # 生成 3000 个样本用于充分训练
    gen = PeriodicityGenerator(num_samples=150, sample_len=200)
    
    # 保存路径建议匹配 run.sh 中的 data_path
    data = gen.generate(save_path='./dataset/syn_data/periodicity_150.npy')
    
    # 可视化检查前两个样本的 第6通道 (Ch6: P24+RedNoise)
    # 验证它们的相位是否不同
    # try:
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(data[0, :, 6], label='Sample 0 - Ch6', alpha=0.8)
    #     plt.plot(data[1, :, 6], label='Sample 1 - Ch6 (Diff Phase)', alpha=0.8, linestyle='--')
    #     plt.title('Check Independence: Two Samples of Channel 6')
    #     plt.legend()
    #     plt.savefig('toy_iid_preview.png')
    #     print("Visualization saved to 'toy_iid_preview.png'")
    # except Exception as e:
    #     print(f"Skipping visualization: {e}")