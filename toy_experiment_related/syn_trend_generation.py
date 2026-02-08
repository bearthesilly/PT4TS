import numpy as np
import os
import matplotlib.pyplot as plt

class TrendGenerator:
    def __init__(self, num_samples=2000, sample_len=192, input_len=96, noise_level=0.1):
        """
        V2.0 Trend Validation Dataset Generator (Encoder-Specialized)
        
        Task: Input 96 -> Output 96 (Total 192)
        Core Logic: Extrapolation based on Momentum (Curvature)
        
        Channels Layout (10 Dim):
        - Group A (0-2): Linear Baseline (Zero Curvature)
        - Group B (3-6): Convex Growth (Positive Curvature / Acceleration)
        - Group C (7-9): Concave Decay (Negative Curvature / Deceleration)
        """
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.input_len = input_len  # Normalization anchor
        self.noise_level = noise_level
        self.num_channels = 10
        
    def _red_noise(self, length, alpha=0.9, scale=0.1):
        """生成红噪声 (AR(1) Process) - 模拟真实世界的有色噪声"""
        noise = np.zeros(length)
        white = np.random.normal(0, scale, length)
        noise[0] = white[0]
        for t in range(1, length):
            # X_t = alpha * X_{t-1} + noise
            noise[t] = alpha * noise[t-1] + np.sqrt(1 - alpha**2) * white[t]
        return noise

    def _generate_single_sample(self):
        L = self.sample_len
        # 原始时间步
        t = np.arange(L, dtype=np.float32)
        
        # [关键]: 时间归一化
        # 将 t=96 映射为 1.0。这样 Input 是 [0, 1], Output 是 [1, 2]
        # 这对多项式和指数函数的数值范围控制至关重要
        t_norm = t / float(self.input_len) 
        
        channels = np.zeros((L, self.num_channels))
        
        # 随机参数扰动 (Data Augmentation) - 确保样本独立性
        # 让每条样本的斜率/曲率略有不同，防止模型死记硬背
        slope_base = np.random.uniform(0.8, 1.2)
        curve_base = np.random.uniform(0.9, 1.1)
        
        # ============================================================
        # Group A: Linear Baseline (Zero Curvature)
        # 验证模型能否退化为简单的 DLinear
        # ============================================================
        
        # Ch0: 正斜率直线 (Positive Slope)
        channels[:, 0] = (2.0 * slope_base) * t_norm 
        
        # Ch1: 负斜率直线 (Negative Slope)
        channels[:, 1] = (-1.5 * slope_base) * t_norm + 3.0
        
        # Ch2: 缓慢漂移 (Slow Drift)
        channels[:, 2] = (0.2 * slope_base) * t_norm + 1.0

        # ============================================================
        # Group B: Convex Growth (Acceleration)
        # 验证 Prior 能否捕捉"向上弯曲"的动量 (二阶导 > 0)
        # ============================================================
        
        # Ch3: 抛物线 (Quadratic) y = x^2
        # 特性: Output 的增长速度明显快于 Input
        channels[:, 3] = 2.0 * (t_norm ** 2) * curve_base
        
        # Ch4: 指数 (Exponential) y = e^x
        # 特性: 爆发式增长
        channels[:, 4] = 0.5 * np.exp(1.5 * t_norm * curve_base)
        
        # Ch5: 幂律 (Cubic) y = x^3
        # 特性: 极其剧烈的加速
        channels[:, 5] = 1.0 * (t_norm ** 3) * curve_base
        
        # Ch6: 噪声掩盖的加速 (Noisy Acceleration)
        # 挑战: 噪声很大，肉眼看Input像直线，但实际上有加速度
        signal_c6 = 2.5 * (t_norm ** 2)
        noise_c6 = self._red_noise(L, alpha=0.95, scale=0.3) # 强红噪声
        channels[:, 6] = signal_c6 + noise_c6

        # ============================================================
        # Group C: Concave Decay (Deceleration)
        # 验证 Prior 能否捕捉"向下弯曲"的动量 (二阶导 < 0)
        # ============================================================
        
        # 为了防止 log(0) 或 div by zero，给时间加一个 offset
        t_safe = t_norm + 0.2
        
        # Ch7: 对数增长 (Logarithmic) y = ln(x)
        # 特性: 一直在涨，但越涨越慢
        channels[:, 7] = 2.0 * np.log(t_safe * 5.0) * curve_base
        
        # Ch8: 根号增长 (Square Root) y = sqrt(x)
        channels[:, 8] = 2.0 * np.sqrt(t_safe) * curve_base
        
        # Ch9: 倒数衰减 (Inverse) y = A - B/x
        # 特性: 快速拉升后趋于饱和 (类似于 Sigmoid 的前半段)
        channels[:, 9] = 3.0 - (1.0 / (t_safe * 2.0))

        # ============================================================
        # Final Touches
        # ============================================================
        
        # 添加全局微弱白噪声 (防止模型过拟合完美曲线)
        global_noise = np.random.normal(0, self.noise_level, (L, self.num_channels))
        
        return channels + global_noise

    def generate(self, save_path='./dataset/syn_data/trend_150.npy'):
        print(f"Generating {self.num_samples} independent samples of length {self.sample_len}...")
        
        all_samples = []
        for i in range(self.num_samples):
            # 每次调用都重新随机斜率和曲率，保证独立性
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
        
        print(f"Synthesized Trend dataset saved to: {save_path}")
        print(f"Data Shape: {final_data.shape} (N_samples, Sequence_Length, Channels)")
        
        return final_data

if __name__ == "__main__":
    # 1. 生成数据
    # 这里我们生成150个样本，长度192（适配 input 96 + output 96）
    gen = TrendGenerator(num_samples=15000, sample_len=192, input_len=96)
    
    # 保存路径建议匹配 run.sh 中的 data_path
    save_path = './dataset/syn_data/trend_15000.npy'
    data = gen.generate(save_path=save_path)
    
