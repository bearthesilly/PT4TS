import numpy as np
import os


class TemporalDecayGeneratorV2:
    """
    Synthetic dataset for the Temporal Decay / Changepoint prior (v2).

    Key design: each sample has a SINGLE CHANGEPOINT at random position
    tau in [20, 76] within the 96-step input window.

    Before tau: all channels follow PATTERN_A.
    After  tau: all channels follow PATTERN_B (very different from A).
    The prediction window (96 steps) continues PATTERN_B.

    Group A (ch0-3): Regime-switching SINE
        Before tau: sin(2*pi*t / 48 + phase) — slow oscillation
        After  tau: sin(2*pi*t / 12 + phase) — fast oscillation (4x freq)

    Group B (ch4-6): Regime-switching LINEAR
        Before tau: slope = +0.12
        After  tau: slope = -0.12 (direction reversal)

    Group C (ch7-9): STATIONARY periodic (control group, no changepoint)
        Verifies that decay prior doesn't hurt stationary signals.

    A model that equally weights all input patches mixes old/new patterns
    -> confused prediction.
    A model with temporal decay focuses on recent patches (pattern B)
    -> correct prediction.
    """

    def __init__(self, num_samples=150, sample_len=192, seed=2025):
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.num_channels = 10
        self.input_len = 96
        self.rng = np.random.RandomState(seed)

    def _generate_single_sample(self):
        L = self.sample_len
        t = np.arange(L, dtype=np.float64)
        channels = np.zeros((L, self.num_channels))

        # Random changepoint within input window
        tau = self.rng.randint(20, 77)  # [20, 76]

        # --- Group A (ch0-3): Sine regime switch ---
        phase_a = self.rng.uniform(0, 2 * np.pi)
        period_before = 48
        period_after = 12
        for c in range(4):
            ch_phase = phase_a + self.rng.uniform(-0.3, 0.3)
            signal = np.zeros(L)
            # Before changepoint: slow sine
            signal[:tau] = np.sin(2 * np.pi * t[:tau] / period_before + ch_phase)
            # After changepoint: fast sine (continuous phase)
            phase_at_tau = 2 * np.pi * tau / period_before + ch_phase
            signal[tau:] = np.sin(2 * np.pi * (t[tau:] - tau) / period_after + phase_at_tau)
            channels[:, c] = signal + self.rng.normal(0, 0.2, L)

        # --- Group B (ch4-6): Linear regime switch ---
        for c in range(4, 7):
            offset = self.rng.uniform(-1, 1)
            slope_before = 0.12
            slope_after = -0.12
            signal = np.zeros(L)
            # Before changepoint
            signal[:tau] = offset + slope_before * (t[:tau] - tau)
            # After changepoint (continuous at tau)
            val_at_tau = offset  # slope_before * 0 + offset
            signal[tau:] = val_at_tau + slope_after * (t[tau:] - tau)
            channels[:, c] = signal + self.rng.normal(0, 0.2, L)

        # --- Group C (ch7-9): Stationary control ---
        stationary_periods = [24, 32, 20]
        for i, c in enumerate(range(7, 10)):
            phase = self.rng.uniform(0, 2 * np.pi)
            channels[:, c] = np.sin(
                2 * np.pi * t / stationary_periods[i] + phase)
            channels[:, c] += self.rng.normal(0, 0.2, L)

        return channels.astype(np.float32)

    def generate(self, save_path='./dataset/syn_data/temporal_decay_v2_150.npy'):
        print(f"Generating {self.num_samples} temporal-decay-v2 samples "
              f"(changepoint design, {self.num_channels} channels, "
              f"len={self.sample_len}) ...")
        all_samples = [self._generate_single_sample()
                       for _ in range(self.num_samples)]
        final_data = np.stack(all_samples, axis=0)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, final_data)
        print(f"Saved to {save_path}  shape={final_data.shape}")
        return final_data


if __name__ == "__main__":
    gen = TemporalDecayGeneratorV2(num_samples=150, sample_len=192, seed=2025)
    gen.generate(save_path='./dataset/syn_data/temporal_decay_v2_150.npy')
