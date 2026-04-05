import numpy as np
import os

class PeriodicityGeneratorNoisy:
    def __init__(self, num_samples=150, sample_len=200, extra_noise=0.6):
        """
        High-noise version of PeriodicityGenerator.
        Adds extra_noise=0.6 white noise to ALL channels (including clean ones).
        Original clean channels had 0 noise; noisy channels had 0.3-0.5.
        """
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.num_channels = 10
        self.extra_noise = extra_noise

    def _red_noise(self, length, alpha=0.9, scale=0.2):
        noise = np.zeros(length)
        white = np.random.normal(0, scale, length)
        noise[0] = white[0]
        for t in range(1, length):
            noise[t] = alpha * noise[t-1] + np.sqrt(1 - alpha**2) * white[t]
        return noise

    def _generate_single_sample(self):
        L = self.sample_len
        channels = np.zeros((L, self.num_channels))
        t = np.arange(L)

        phase_shift = np.random.uniform(0, 2*np.pi)

        # Group A: Base periodic signals (originally clean, now noisy)
        channels[:, 0] = np.sin(2 * np.pi * t / 24 + phase_shift)
        channels[:, 1] = np.sin(2 * np.pi * t / 12 + phase_shift)
        channels[:, 2] = np.cos(2 * np.pi * t / 48 + phase_shift)

        # Group B: Complex harmonics (originally clean, now noisy)
        channels[:, 3] = np.sin(2 * np.pi * t / 24 + phase_shift) + \
                         0.5 * np.sin(2 * np.pi * t / 12 + phase_shift)
        channels[:, 4] = np.sin(2 * np.pi * t / 24) + np.sin(2 * np.pi * t / 20)

        # Group C: Originally noisy, now noisier
        channels[:, 5] = channels[:, 0]  # base signal, noise added below
        channels[:, 6] = channels[:, 0]
        channels[:, 7] = channels[:, 3]
        channels[:, 8] = channels[:, 4]

        # Ch9: Pure red noise (control)
        channels[:, 9] = self._red_noise(L, alpha=0.9, scale=0.5)

        # Add extra_noise to ALL channels (the key difference)
        for c in range(self.num_channels):
            channels[:, c] += np.random.normal(0, self.extra_noise, L)

        # Group C also gets the original red noise on top
        channels[:, 6] += self._red_noise(L, alpha=0.9, scale=0.4)
        channels[:, 7] += self._red_noise(L, alpha=0.9, scale=0.4)
        channels[:, 8] += self._red_noise(L, alpha=0.9, scale=0.4)

        return channels

    def generate(self, save_path='./dataset/syn_data/periodicity_noisy.npy'):
        print(f"Generating {self.num_samples} noisy samples (extra_noise={self.extra_noise}) ...")

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
    gen = PeriodicityGeneratorNoisy(num_samples=150, sample_len=200, extra_noise=0.6)
    data = gen.generate(save_path='./dataset/syn_data/periodicity_150_noisy.npy')
