import numpy as np
import os

class CausalGeneratorNoisy:
    def __init__(self, num_samples=150, sample_len=192, tau=8, noise_level=0.3):
        """
        High-noise version of CausalGenerator for robustness testing.
        noise_level increased from 0.05 -> 0.3 (6x).
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

        # Group A: Ch0 -> Ch1 (Modulated Sine -> Delayed Copy)
        T_base = 24
        phase = np.random.uniform(0, 2*np.pi)
        base_wave = np.sin(2 * np.pi * t / T_base + phase)
        envelope_freq = np.random.choice([1, 0.5])
        envelope = np.cos(np.pi * t / L * envelope_freq) + 1.5
        ch0 = base_wave * envelope
        channels[:, 0] = ch0 + np.random.normal(0, self.noise_level, L)

        ch1 = np.zeros(L)
        ch1[tau:] = ch0[:-tau]
        ch1[:tau] = ch0[0]
        channels[:, 1] = ch1 + np.random.normal(0, self.noise_level, L)

        # Group B: Ch2 -> Ch3 (Periodic Pulse -> Step Change)
        ch2 = np.zeros(L)
        ch3 = np.zeros(L)
        current_level = 0.0

        interval = np.random.choice([16, 20, 24])
        start_offset = np.random.randint(0, interval)
        pulse_indices = np.arange(start_offset, L - tau, interval)
        direction = np.random.choice([1, -1])

        ch2[pulse_indices] = direction * 2.0
        channels[:, 2] = ch2 + np.random.normal(0, self.noise_level * 0.4, L)  # 0.02 -> 0.12

        if len(pulse_indices) > 0:
            effect_indices = pulse_indices + tau
            effect_indices = effect_indices[effect_indices < L]
            events = np.zeros(L)
            events[effect_indices] = direction
            for i in range(1, L):
                if events[i] != 0:
                    current_level += events[i] * 1.0
                ch3[i] = current_level
        channels[:, 3] = ch3 + np.random.normal(0, self.noise_level, L)

        # Group C: Ch4 -> Ch5 (Sawtooth -> Linear Mapping)
        T_saw = 30
        phase_saw = np.random.randint(0, T_saw)
        ch4 = ((t + phase_saw) % T_saw) / T_saw * 2 - 1
        channels[:, 4] = ch4 + np.random.normal(0, self.noise_level * 0.6, L)  # 0.03 -> 0.18

        m = np.random.choice([-2.0, 0.5, 2.0])
        b = np.random.uniform(-1, 1)
        ch5_clean = m * ch4 + b

        ch5 = np.zeros(L)
        ch5[tau:] = ch5_clean[:-tau]
        ch5[:tau] = ch5_clean[0]
        channels[:, 5] = ch5 + np.random.normal(0, self.noise_level, L)

        return channels

    def generate(self, save_path='./dataset/syn_data/lag_noisy.npy'):
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
    TAU_VAL = 8
    gen = CausalGeneratorNoisy(num_samples=150, sample_len=192, tau=TAU_VAL, noise_level=0.3)
    save_path = f'./dataset/syn_data/lag_{TAU_VAL}_150_noisy.npy'
    data = gen.generate(save_path=save_path)
