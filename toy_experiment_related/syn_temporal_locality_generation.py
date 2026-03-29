import numpy as np
import os


class TemporalLocalityGenerator:
    """
    Synthetic dataset for the Temporal Locality / Decay prior.

    Key design: REGIME-SWITCHING signals.  Each channel alternates between
    clearly different regimes (different frequency / slope).  Regime changes
    occur every ~30-50 steps, so the 96-step input window spans 2-3 regimes.
    The forecast period continues the LATEST regime.

    Group A (ch0-3): alternates between two sine frequencies.
        The correct forecast uses the current frequency, but old history
        contains the previous (wrong) frequency and is misleading.

    Group B (ch4-6): alternates between two linear slopes.
        The correct forecast extrapolates the current slope.  Old slope
        segments push predictions in the wrong direction.

    Group C (ch7-9): STATIONARY periodic signals (control group).
        Distant context IS informative here.  The decay prior should be
        roughly neutral for these channels.

    A model with temporal decay down-weights distant (old-regime) patches
    → focuses on the current regime → better prediction.
    A vanilla model gives equal weight to all patches → confused by old regimes.
    """

    def __init__(self, num_samples=150, sample_len=192, seed=2025):
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.num_channels = 10
        self.rng = np.random.RandomState(seed)

    def _regime_switching_sine(self, length):
        """Generate sine wave that switches frequency at random points."""
        x = np.zeros(length)
        t = np.arange(length, dtype=np.float64)
        freqs = self.rng.choice([8, 48], size=10)  # wider frequency gap for stronger regime contrast
        seg_start = 0
        seg_idx = 0
        phase = self.rng.uniform(0, 2 * np.pi)

        while seg_start < length:
            seg_len = self.rng.randint(15, 30)  # shorter segments = more switches per window
            seg_end = min(seg_start + seg_len, length)
            period = freqs[seg_idx % len(freqs)]
            x[seg_start:seg_end] = np.sin(
                2 * np.pi * t[seg_start:seg_end] / period + phase)
            phase = 2 * np.pi * t[min(seg_end, length - 1)] / period + phase
            seg_start = seg_end
            seg_idx += 1

        return x

    def _regime_switching_linear(self, length):
        """Generate piecewise linear with alternating steep/flat slopes."""
        x = np.zeros(length)
        slopes = [0.15, -0.15, 0.03, -0.03]  # larger slope contrast
        seg_start = 0
        val = self.rng.uniform(-1, 1)
        seg_idx = 0

        while seg_start < length:
            seg_len = self.rng.randint(15, 30)  # shorter segments
            slope = slopes[seg_idx % len(slopes)]
            for s in range(seg_len):
                if seg_start + s >= length:
                    break
                x[seg_start + s] = val + slope * s
            seg_start += seg_len
            val = x[min(seg_start - 1, length - 1)]
            seg_idx += 1

        return x

    def _generate_single_sample(self):
        L = self.sample_len
        t = np.arange(L, dtype=np.float64)
        channels = np.zeros((L, self.num_channels))

        for c in range(4):
            channels[:, c] = self._regime_switching_sine(L)
            channels[:, c] += self.rng.normal(0, 0.1, L)

        for c in range(4, 7):
            channels[:, c] = self._regime_switching_linear(L)
            channels[:, c] += self.rng.normal(0, 0.05, L)

        stationary_periods = [24, 32, 20]
        for i, c in enumerate(range(7, 10)):
            phase = self.rng.uniform(0, 2 * np.pi)
            channels[:, c] = np.sin(
                2 * np.pi * t / stationary_periods[i] + phase)
            channels[:, c] += self.rng.normal(0, 0.1, L)

        return channels.astype(np.float32)

    def generate(self, save_path='./dataset/syn_data/temporal_locality_150.npy'):
        print(f"Generating {self.num_samples} temporal-locality samples "
              f"({self.num_channels} channels, len={self.sample_len}) ...")
        all_samples = [self._generate_single_sample()
                       for _ in range(self.num_samples)]
        final_data = np.stack(all_samples, axis=0)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, final_data)
        print(f"Saved to {save_path}  shape={final_data.shape}")
        return final_data


if __name__ == "__main__":
    gen = TemporalLocalityGenerator(num_samples=150, sample_len=192, seed=2025)
    gen.generate(save_path='./dataset/syn_data/temporal_locality_150.npy')
