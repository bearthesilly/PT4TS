import numpy as np
import os
import matplotlib.pyplot as plt

class CausalGenerator:
    def __init__(self, num_samples=2000, sample_len=192, tau=8, noise_level=0.05):
        """
        生成的因果滞后数据样本，适配 data_loader.py 中的 Toy_Dataset 类。
        Task: Logic & Lag Discovery
        Core: 
        1. Cause (Source) is predictable (Periodic/Linear).
        2. Effect (Target) is strictly lagged by Tau.
        
        Channels Layout (6 Dim):
        - Group A (0-1): Modulated Sine -> Delayed Copy
        - Group B (2-3): Periodic Pulses -> Step Accumulation
        - Group C (4-5): Sawtooth -> Linear Transform
        """
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.tau = tau
        self.noise_level = noise_level
        self.num_channels = 6
        
    def _generate_single_sample(self):
        L = self.sample_len
        tau = self.tau
        channels = np.zeros((L, self.num_channels))
        t = np.arange(L)
        
        # =========================================================
        # Group A: Ch0 -> Ch1 (Modulated Sine -> Delayed Copy)
        # 逻辑: 基础正弦波 (可预测)，带随机包络 (增加难度但保持连续性)
        # =========================================================
        
        # 1. 基础波形 (Base Wave): 周期 T=24 (可预测)
        T_base = 24
        # 随机相位，保证样本独立性
        phase = np.random.uniform(0, 2*np.pi)
        base_wave = np.sin(2 * np.pi * t / T_base + phase)
        
        # 2. 振幅包络 (Slow Envelope): 极低频变化，保证局部可预测
        envelope_freq = np.random.choice([1, 0.5]) 
        envelope = np.cos(np.pi * t / L * envelope_freq) + 1.5 # range [0.5, 2.5]
        
        ch0 = base_wave * envelope
        channels[:, 0] = ch0 + np.random.normal(0, self.noise_level, L)
        
        # Target: 严格滞后
        # 填充头部: 为了保持可预测性，我们假设 t<0 的部分延续了同样的波形逻辑
        # 这里简单处理，用 0 填充或噪声
        ch1 = np.zeros(L)
        ch1[tau:] = ch0[:-tau]
        ch1[:tau] = ch0[0] # 简单Padding
        channels[:, 1] = ch1 + np.random.normal(0, self.noise_level, L)

        # =========================================================
        # Group B: Ch2 -> Ch3 (Periodic Pulse -> Step Change)
        # 逻辑: Ch2 是像心跳一样的脉冲，完全可预测。
        # =========================================================
        
        ch2 = np.zeros(L)
        ch3 = np.zeros(L)
        current_level = 0.0
        
        # 1. 设定脉冲节奏 (Rhythm)
        interval = np.random.choice([16, 20, 24])
        # 随机相位偏移 (Phase shift)
        start_offset = np.random.randint(0, interval)
        
        # 生成脉冲序列
        pulse_indices = np.arange(start_offset, L - tau, interval)
        
        # 脉冲方向 (随机但固定): 这次样本全是正脉冲，或者全是负
        direction = np.random.choice([1, -1])
        
        ch2[pulse_indices] = direction * 2.0 # 明显的脉冲
        channels[:, 2] = ch2 + np.random.normal(0, 0.02, L)
        
        # 2. 生成响应 (Step)
        if len(pulse_indices) > 0:
            effect_indices = pulse_indices + tau
            # 过滤掉超出范围的索引
            effect_indices = effect_indices[effect_indices < L]
            
            # 构建 Ch3 状态
            events = np.zeros(L)
            events[effect_indices] = direction # 记录生效点
            
            for i in range(1, L):
                if events[i] != 0:
                    current_level += events[i] * 1.0 # 阶跃幅度
                ch3[i] = current_level
            
        channels[:, 3] = ch3 + np.random.normal(0, self.noise_level, L)

        # =========================================================
        # Group C: Ch4 -> Ch5 (Sawtooth Wave -> Linear Mapping)
        # 逻辑: 锯齿波/三角波 (线性特征强) -> mA + b
        # =========================================================
        
        # 1. Ch4 Source: 锯齿波 (Sawtooth)
        T_saw = 30
        # 随机开始位置
        phase_saw = np.random.randint(0, T_saw)
        ch4 = ((t + phase_saw) % T_saw) / T_saw * 2 - 1 # range [-1, 1]
        channels[:, 4] = ch4 + np.random.normal(0, 0.03, L)
        
        # 2. Ch5 Target: Linear Transform
        m = np.random.choice([-2.0, 0.5, 2.0]) # 明显的斜率变化
        b = np.random.uniform(-1, 1)
        
        ch5_clean = m * ch4 + b
        
        # 实施滞后
        ch5 = np.zeros(L)
        ch5[tau:] = ch5_clean[:-tau]
        ch5[:tau] = ch5_clean[0] # Pad
        
        channels[:, 5] = ch5 + np.random.normal(0, self.noise_level, L)

        return channels

    def generate(self, save_path='./dataset/syn_data/lag.npy'):
        print(f"Generating {self.num_samples} independent samples of length {self.sample_len} with Tau={self.tau}...")
        
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
        
        print(f"Synthesized Causal Lag dataset saved to: {save_path}")
        print(f"Data Shape: {final_data.shape} (N_samples, Sequence_Length, Channels)")
        
        return final_data

if __name__ == "__main__":
    # 配置生成参数
    TAU_VAL = 8
    gen = CausalGenerator(num_samples=150, sample_len=192, tau=TAU_VAL)
    
    # 保存路径建议匹配 run.sh 中的 data_path
    save_path = f'./dataset/syn_data/lag_{TAU_VAL}_150.npy'
    data = gen.generate(save_path=save_path)

    # 可选: 可视化检查第一个样本
    # try:
    #     t = np.arange(gen.sample_len)
    #     sample = data[0]
        
    #     fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
    #     # --- Group A ---
    #     axes[0].set_title(f"Group A: Modulated Sine (Period=24, Tau={TAU_VAL})")
    #     axes[0].plot(t, sample[:, 0], 'k-', label='Ch0 Source', alpha=0.6)
    #     axes[0].plot(t, sample[:, 1], 'r--', label='Ch1 Target', linewidth=2)
    #     axes[0].legend(loc='upper right')
    #     axes[0].grid(True, alpha=0.3)
        
    #     # --- Group B ---
    #     axes[1].set_title(f"Group B: Periodic Pulses (Predictable!) -> Step (Tau={TAU_VAL})")
    #     axes[1].plot(t, sample[:, 2], 'k-', label='Ch2 Source', linewidth=1.5)
    #     axes[1].plot(t, sample[:, 3], 'g-', label='Ch3 Target', linewidth=2)
    #     axes[1].legend(loc='upper right')
    #     axes[1].grid(True, alpha=0.3)
        
    #     # --- Group C ---
    #     axes[2].set_title(f"Group C: Sawtooth Linear Map (Tau={TAU_VAL})")
    #     axes[2].plot(t, sample[:, 4], 'k-', label='Ch4 Source', alpha=0.6)
    #     axes[2].plot(t, sample[:, 5], 'm--', label='Ch5 Target', linewidth=2)
    #     axes[2].legend(loc='upper right')
    #     axes[2].grid(True, alpha=0.3)
        
    #     plt.tight_layout()
    #     plt.savefig('toy_lag_preview.png')
    #     print("Visualization saved to 'toy_lag_preview.png'")
    # except Exception as e:
    #     print(f"Skipping visualization: {e}")