import numpy as np
import os

class TrendGeneratorNoisy:
    def __init__(self, num_samples=150, sample_len=192, input_len=96, noise_level=0.5):
        """
        High-noise version of TrendGenerator.
        noise_level increased from 0.1 -> 0.5 (5x).
        """
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.input_len = input_len
        self.noise_level = noise_level
        self.num_channels = 10

    def _red_noise(self, length, alpha=0.9, scale=0.1):
        noise = np.zeros(length)
        white = np.random.normal(0, scale, length)
        noise[0] = white[0]
        for t in range(1, length):
            noise[t] = alpha * noise[t-1] + np.sqrt(1 - alpha**2) * white[t]
        return noise

    def _generate_single_sample(self):
        L = self.sample_len
        t = np.arange(L, dtype=np.float32)
        t_norm = t / float(self.input_len)

        channels = np.zeros((L, self.num_channels))

        slope_base = np.random.uniform(0.8, 1.2)
        curve_base = np.random.uniform(0.9, 1.1)

        # Group A: Linear Baseline
        channels[:, 0] = (2.0 * slope_base) * t_norm
        channels[:, 1] = (-1.5 * slope_base) * t_norm + 3.0
        channels[:, 2] = (0.2 * slope_base) * t_norm + 1.0

        # Group B: Convex Growth
        channels[:, 3] = 2.0 * (t_norm ** 2) * curve_base
        channels[:, 4] = 0.5 * np.exp(1.5 * t_norm * curve_base)
        channels[:, 5] = 1.0 * (t_norm ** 3) * curve_base

        # Ch6: Noisy acceleration (red noise also scaled up)
        signal_c6 = 2.5 * (t_norm ** 2)
        noise_c6 = self._red_noise(L, alpha=0.95, scale=0.6)  # 0.3 -> 0.6
        channels[:, 6] = signal_c6 + noise_c6

        # Group C: Concave Decay
        t_safe = t_norm + 0.2
        channels[:, 7] = 2.0 * np.log(t_safe * 5.0) * curve_base
        channels[:, 8] = 2.0 * np.sqrt(t_safe) * curve_base
        channels[:, 9] = 3.0 - (1.0 / (t_safe * 2.0))

        # Global noise (5x original)
        global_noise = np.random.normal(0, self.noise_level, (L, self.num_channels))

        return channels + global_noise

    def generate(self, save_path='./dataset/syn_data/trend_noisy.npy'):
        print(f"Generating {self.num_samples} noisy samples (noise={self.noise_level}) ...")

        all_samples = []
        for i in range(self.num_samples):
            sample = self._generate_single_sample()
            all_samples.append(sample)

        final_data = np.stack(all_samples, axis=0).astype(np.float32)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, final_data)

        print(f"Saved to: {save_path}, Shape: {final_data.shape}")
        return final_data

if __name__ == "__main__":
    gen = TrendGeneratorNoisy(num_samples=150, sample_len=192, input_len=96, noise_level=0.5)
    data = gen.generate(save_path='./dataset/syn_data/trend_150_noisy.npy')
