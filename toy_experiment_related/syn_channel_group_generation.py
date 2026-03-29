import numpy as np
import os


class ChannelGroupGenerator:
    """
    Synthetic dataset for the Channel Grouping prior.

    Layout: 9 channels in 3 groups of 3
        Group A: ch0, ch1, ch2
        Group B: ch3, ch4, ch5
        Group C: ch6, ch7, ch8

    Key design: each channel = shared_group_latent + HEAVY individual noise.
    The latent is smooth and forecastable (sine + slow drift).
    The noise is large (SNR << 1 per single channel).

    To forecast well, the model MUST average across same-group channels to
    denoise the latent signal.  Cross-group channels have DIFFERENT latents,
    so including them corrupts the estimate.

    A model that knows the group partition (blocks cross-group edges) can
    pool within-group on the channel axis → better latent estimate → better
    prediction.  A vanilla model with 150 samples struggles to discover the
    correct grouping and wastes capacity on cross-group noise.
    """

    def __init__(self, num_samples=150, sample_len=192, seed=2025):
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.num_channels = 9
        self.groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.rng = np.random.RandomState(seed)

    def _generate_single_sample(self):
        L = self.sample_len
        t = np.arange(L, dtype=np.float64)
        channels = np.zeros((L, self.num_channels))

        group_periods = [24, 36, 16]

        for g_idx, group in enumerate(self.groups):
            phase = self.rng.uniform(0, 2 * np.pi)
            period = group_periods[g_idx]
            drift_slope = self.rng.uniform(-0.005, 0.005)

            latent = (np.sin(2 * np.pi * t / period + phase)
                      + drift_slope * t)

            for ch in group:
                noise = self.rng.normal(0, 1.5, L)
                channels[:, ch] = latent + noise

        return channels.astype(np.float32)

    def generate(self, save_path='./dataset/syn_data/channel_group_150.npy'):
        print(f"Generating {self.num_samples} channel-grouping samples "
              f"(3 groups x 3 channels, len={self.sample_len}) ...")
        all_samples = [self._generate_single_sample()
                       for _ in range(self.num_samples)]
        final_data = np.stack(all_samples, axis=0)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, final_data)
        print(f"Saved to {save_path}  shape={final_data.shape}")
        return final_data


if __name__ == "__main__":
    gen = ChannelGroupGenerator(num_samples=150, sample_len=192, seed=2025)
    gen.generate(save_path='./dataset/syn_data/channel_group_150.npy')
