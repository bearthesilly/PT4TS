import numpy as np
import os


# =====================================================================
# Known Mixing Matrix A  (C=9 observed channels, K=3 latent sources)
# This matrix is shared between data generation and model prior.
# =====================================================================
MIXING_MATRIX = np.array([
    # Source 0   Source 1   Source 2
    [  1.0,       0.0,       0.0  ],   # Ch0: pure source 0
    [  0.0,       1.0,       0.0  ],   # Ch1: pure source 1
    [  0.0,       0.0,       1.0  ],   # Ch2: pure source 2
    [  0.7,       0.7,       0.0  ],   # Ch3: mix of source 0 & 1
    [  0.0,       0.6,       0.8  ],   # Ch4: mix of source 1 & 2
    [  0.8,       0.0,       0.6  ],   # Ch5: mix of source 0 & 2
    [  0.5,       0.5,       0.5  ],   # Ch6: equal mix of all
    [  0.9,       0.3,       0.1  ],   # Ch7: dominated by source 0
    [  0.1,       0.3,       0.9  ],   # Ch8: dominated by source 2
], dtype=np.float32)  # shape [9, 3]


class MixingMatrixGenerator:
    """
    Synthetic dataset for the Mixing Matrix (Source Separation) prior.

    Data model:  y(t) = A @ s(t) + noise
    - s(t) in R^3: three independent, smooth, predictable source signals
    - A in R^{9x3}: known mixing matrix (defined above)
    - noise ~ N(0, sigma^2): heavy i.i.d. noise per channel

    The source signals are smooth (sine + slow drift), so they are
    individually easy to forecast.  But each observed channel is a noisy
    linear combination, and with sigma large, single-channel SNR is low.

    A model that knows A can unmix (A^+ @ y) to recover clean sources,
    forecast in source space, then re-mix.  Without A, the model must
    simultaneously discover the mixing structure and forecast — infeasible
    with only 150 training samples.
    """

    def __init__(self, num_samples=150, sample_len=192, noise_sigma=1.5, seed=2025):
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.noise_sigma = noise_sigma
        self.A = MIXING_MATRIX  # [9, 3]
        self.num_channels = self.A.shape[0]  # 9
        self.num_sources = self.A.shape[1]   # 3
        self.rng = np.random.RandomState(seed)

    def _generate_sources(self, length):
        """Generate 3 independent, smooth, predictable source signals."""
        t = np.arange(length, dtype=np.float64)
        sources = np.zeros((length, self.num_sources))

        # Source 0: sine with period 24 + slow linear drift
        phase0 = self.rng.uniform(0, 2 * np.pi)
        drift0 = self.rng.uniform(-0.008, 0.008)
        sources[:, 0] = np.sin(2 * np.pi * t / 24 + phase0) + drift0 * t

        # Source 1: sine with period 36 + cosine modulation
        phase1 = self.rng.uniform(0, 2 * np.pi)
        mod_phase = self.rng.uniform(0, 2 * np.pi)
        sources[:, 1] = (np.sin(2 * np.pi * t / 36 + phase1)
                         * (1.0 + 0.3 * np.cos(2 * np.pi * t / 96 + mod_phase)))

        # Source 2: sawtooth-like (piecewise linear, period 20) + gentle curve
        phase2 = self.rng.randint(0, 20)
        saw = ((t + phase2) % 20) / 20.0 * 2 - 1  # range [-1, 1]
        curve = 0.3 * np.sin(2 * np.pi * t / 80)
        sources[:, 2] = saw + curve

        return sources.astype(np.float32)

    def _generate_single_sample(self):
        L = self.sample_len
        sources = self._generate_sources(L)       # [L, 3]
        clean = sources @ self.A.T                 # [L, 9]
        noise = self.rng.normal(0, self.noise_sigma, (L, self.num_channels))
        observed = clean + noise
        return observed.astype(np.float32)

    def generate(self, save_path='./dataset/syn_data/channel_group_v2_150.npy'):
        print(f"Generating {self.num_samples} mixing-matrix samples "
              f"(A: {self.num_channels}x{self.num_sources}, "
              f"sigma={self.noise_sigma}, len={self.sample_len}) ...")
        all_samples = [self._generate_single_sample()
                       for _ in range(self.num_samples)]
        final_data = np.stack(all_samples, axis=0)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, final_data)
        print(f"Saved to {save_path}  shape={final_data.shape}")
        return final_data


if __name__ == "__main__":
    gen = MixingMatrixGenerator(num_samples=150, sample_len=192,
                                noise_sigma=1.5, seed=2025)
    gen.generate(save_path='./dataset/syn_data/channel_group_v2_150.npy')
