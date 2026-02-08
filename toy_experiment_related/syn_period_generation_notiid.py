import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class PeriodicityGenerator:
    def __init__(self, total_len=15000):
        """
        生成契合 Time-Series-Library 标准 DataLoader 的 CSV 数据集。
        生成一条连续的时间序列而不是独立的切片，以便支持标准的数据加载和切分。
        
        Args:
            total_len: 生成序列的总长度 (建议 > 10000 以便划分 Train/Val/Test)
        """
        self.total_len = total_len
        
    def _red_noise(self, length, alpha=0.9, scale=0.2):
        """生成红噪声 (Autoregressive Noise)"""
        # 红噪声序列，模拟真实的随机游走趋势
        noise = np.zeros(length)
        white = np.random.normal(0, scale, length)
        noise[0] = white[0]
        for t in range(1, length):
            noise[t] = alpha * noise[t-1] + np.sqrt(1 - alpha**2) * white[t]
        return noise

    def generate(self, save_path='./dataset/toy_periodicity.csv'):
        L = self.total_len
        channels = np.zeros((L, 10))
        t = np.arange(L)
        
        # 使用统一的随机相位，保持序列的全局连续性
        phase_shift = np.random.uniform(0, 2*np.pi)
        
        # ==========================================
        # Group A: 标准周期 (Base Check)
        # ==========================================
        # Ch0: Period = 24
        # 对应 enc_in=0
        channels[:, 0] = np.sin(2 * np.pi * t / 24 + phase_shift)
        
        # Ch1: Period = 12 (高频)
        channels[:, 1] = np.sin(2 * np.pi * t / 12 + phase_shift)
        
        # Ch2: Period = 48 (低频)
        channels[:, 2] = np.cos(2 * np.pi * t / 48 + phase_shift)

        # ==========================================
        # Group B: 复杂叠加 (Harmonics)
        # ==========================================
        # Ch3: Period 24 + Period 12 (基波 + 二次谐波)
        channels[:, 3] = np.sin(2 * np.pi * t / 24 + phase_shift) + \
                         0.5 * np.sin(2 * np.pi * t / 12 + phase_shift)

        # Ch4: 拍频现象 (Beating) - 两个相近频率叠加
        channels[:, 4] = np.sin(2 * np.pi * t / 24) + np.sin(2 * np.pi * t / 20)

        # ==========================================
        # Group C: 抗噪测试 (Robustness)
        # ==========================================
        # Ch5: Period 24 + White Noise
        channels[:, 5] = channels[:, 0] + np.random.normal(0, 0.3, L)
        
        # Ch6: Period 24 + Red Noise (重要测试项)
        # 生成整条长序列的红噪声
        red_noise = self._red_noise(L, alpha=0.9, scale=0.4)
        channels[:, 6] = channels[:, 0] + red_noise

        # Ch7: Period 24 + Period 12 (Harmonics) + Red Noise
        channels[:, 7] = channels[:, 3] + self._red_noise(L, alpha=0.9, scale=0.4)

        # ==========================================
        # Group D: 复杂叠加抗噪测试 (Complex Robustness)
        # ==========================================
        # Ch8: Beating (P24 + P20) + Red Noise
        # 拍频信号叠加红噪声，测试由于拍频引起的振幅变化在噪声下的可识别性
        column_names = [f'col_{i}' for i in range(9)] + ['OT']
        df = pd.DataFrame(channels, columns=column_names)
        
        # 生成标准 Dataset_Custom 必需的 'date' 列
        df['date'] = pd.date_range(start='2020-01-01', periods=L, freq='H')
        
        # 将 date 移到第一列
        cols = ['date'] + [c for c in df.columns if c != 'date']
        df = df[cols]
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Synthesized dataset saved to: {save_path}")
        print(f"Data Shape: {df.shape}")
        return df

if __name__ == "__main__":
    # 生成 15000 个时间步的数据，足够切分 Train/Val/Test
    # 例如：70% Train (~10k), 10% Val, 20% Test
    gen = PeriodicityGenerator(total_len=1000)
    df = gen.generate('./dataset/syn_data/periodicity_not_iid_400.csv')
    
    # 预览
    print("Preview of generated data columns:")
    print(df.head())
    
    # 可视化检查（保存为图片）
    try:
        plt.figure(figsize=(12, 6))
        # 检查 Ch6 (P24 + Red Noise)
        plt.plot(df['col_6'].iloc[:300], label='Ch6: Period 24 + Red Noise', color='orange')
        # 检查 Ch0 (Ground Truth)
        plt.plot(df['col_0'].iloc[:300], label='Ch0: Clean Period 24', linestyle='--', alpha=0.6)
        plt.title('Generated Synthetic Data Preview (First 300 steps)')
        plt.legend()
        plt.savefig('toy_dataset_preview.png')
        print("Visualization saved to 'toy_dataset_preview.png'")
    except Exception as e:
        print(f"Skipping visualization: {e}")