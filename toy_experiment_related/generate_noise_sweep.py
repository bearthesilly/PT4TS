"""
Generate synthetic datasets at multiple noise levels for robustness testing.
Usage: python toy_experiment_related/generate_noise_sweep.py
"""
import numpy as np
import os
import sys

# ============================================================
# Lag Generator (adapted from syn_lag_generation.py)
# ============================================================
def generate_lag(num_samples, sample_len, tau, noise_level, save_path):
    all_samples = []
    for _ in range(num_samples):
        L = sample_len
        channels = np.zeros((L, 6))
        t = np.arange(L)

        # Group A: Modulated Sine -> Delayed Copy
        phase = np.random.uniform(0, 2*np.pi)
        base_wave = np.sin(2 * np.pi * t / 24 + phase)
        envelope = np.cos(np.pi * t / L * np.random.choice([1, 0.5])) + 1.5
        ch0 = base_wave * envelope
        channels[:, 0] = ch0 + np.random.normal(0, noise_level, L)
        ch1 = np.zeros(L)
        ch1[tau:] = ch0[:-tau]
        ch1[:tau] = ch0[0]
        channels[:, 1] = ch1 + np.random.normal(0, noise_level, L)

        # Group B: Periodic Pulse -> Step
        ch2 = np.zeros(L)
        interval = np.random.choice([16, 20, 24])
        start_offset = np.random.randint(0, interval)
        pulse_indices = np.arange(start_offset, L - tau, interval)
        direction = np.random.choice([1, -1])
        ch2[pulse_indices] = direction * 2.0
        channels[:, 2] = ch2 + np.random.normal(0, noise_level * 0.4, L)

        ch3 = np.zeros(L)
        current_level = 0.0
        if len(pulse_indices) > 0:
            effect_indices = pulse_indices + tau
            effect_indices = effect_indices[effect_indices < L]
            events = np.zeros(L)
            events[effect_indices] = direction
            for i in range(1, L):
                if events[i] != 0:
                    current_level += events[i] * 1.0
                ch3[i] = current_level
        channels[:, 3] = ch3 + np.random.normal(0, noise_level, L)

        # Group C: Sawtooth -> Linear Mapping
        T_saw = 30
        phase_saw = np.random.randint(0, T_saw)
        ch4 = ((t + phase_saw) % T_saw) / T_saw * 2 - 1
        channels[:, 4] = ch4 + np.random.normal(0, noise_level * 0.6, L)
        m = np.random.choice([-2.0, 0.5, 2.0])
        b = np.random.uniform(-1, 1)
        ch5_clean = m * ch4 + b
        ch5 = np.zeros(L)
        ch5[tau:] = ch5_clean[:-tau]
        ch5[:tau] = ch5_clean[0]
        channels[:, 5] = ch5 + np.random.normal(0, noise_level, L)

        all_samples.append(channels)

    data = np.stack(all_samples, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data)
    print(f"  Lag (noise={noise_level:.2f}): {save_path}  shape={data.shape}")


# ============================================================
# Periodicity Generator (adapted from syn_period_generation.py)
# ============================================================
def _red_noise(length, alpha=0.9, scale=0.2):
    noise = np.zeros(length)
    white = np.random.normal(0, scale, length)
    noise[0] = white[0]
    for t in range(1, length):
        noise[t] = alpha * noise[t-1] + np.sqrt(1 - alpha**2) * white[t]
    return noise

def generate_period(num_samples, sample_len, extra_noise, save_path):
    all_samples = []
    for _ in range(num_samples):
        L = sample_len
        channels = np.zeros((L, 10))
        t = np.arange(L)
        phase = np.random.uniform(0, 2*np.pi)

        channels[:, 0] = np.sin(2 * np.pi * t / 24 + phase)
        channels[:, 1] = np.sin(2 * np.pi * t / 12 + phase)
        channels[:, 2] = np.cos(2 * np.pi * t / 48 + phase)
        channels[:, 3] = np.sin(2 * np.pi * t / 24 + phase) + 0.5 * np.sin(2 * np.pi * t / 12 + phase)
        channels[:, 4] = np.sin(2 * np.pi * t / 24) + np.sin(2 * np.pi * t / 20)

        channels[:, 5] = channels[:, 0]
        channels[:, 6] = channels[:, 0]
        channels[:, 7] = channels[:, 3]
        channels[:, 8] = channels[:, 4]
        channels[:, 9] = _red_noise(L, alpha=0.9, scale=0.5)

        # Extra white noise on ALL channels
        for c in range(10):
            channels[:, c] += np.random.normal(0, extra_noise, L)

        # Group C original red noise on top
        channels[:, 6] += _red_noise(L, alpha=0.9, scale=0.4)
        channels[:, 7] += _red_noise(L, alpha=0.9, scale=0.4)
        channels[:, 8] += _red_noise(L, alpha=0.9, scale=0.4)

        all_samples.append(channels)

    data = np.stack(all_samples, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data)
    print(f"  Period (extra_noise={extra_noise:.2f}): {save_path}  shape={data.shape}")


# ============================================================
# Trend Generator (adapted from syn_trend_generation.py)
# ============================================================
def generate_trend(num_samples, sample_len, input_len, noise_level, save_path):
    def _rn(length, alpha=0.9, scale=0.1):
        n = np.zeros(length)
        w = np.random.normal(0, scale, length)
        n[0] = w[0]
        for t in range(1, length):
            n[t] = alpha * n[t-1] + np.sqrt(1 - alpha**2) * w[t]
        return n

    all_samples = []
    for _ in range(num_samples):
        L = sample_len
        t = np.arange(L, dtype=np.float32)
        t_norm = t / float(input_len)
        channels = np.zeros((L, 10))

        slope_base = np.random.uniform(0.8, 1.2)
        curve_base = np.random.uniform(0.9, 1.1)

        channels[:, 0] = (2.0 * slope_base) * t_norm
        channels[:, 1] = (-1.5 * slope_base) * t_norm + 3.0
        channels[:, 2] = (0.2 * slope_base) * t_norm + 1.0

        channels[:, 3] = 2.0 * (t_norm ** 2) * curve_base
        channels[:, 4] = 0.5 * np.exp(1.5 * t_norm * curve_base)
        channels[:, 5] = 1.0 * (t_norm ** 3) * curve_base

        signal_c6 = 2.5 * (t_norm ** 2)
        noise_c6 = _rn(L, alpha=0.95, scale=noise_level * 1.2)
        channels[:, 6] = signal_c6 + noise_c6

        t_safe = t_norm + 0.2
        channels[:, 7] = 2.0 * np.log(t_safe * 5.0) * curve_base
        channels[:, 8] = 2.0 * np.sqrt(t_safe) * curve_base
        channels[:, 9] = 3.0 - (1.0 / (t_safe * 2.0))

        global_noise = np.random.normal(0, noise_level, (L, 10))
        all_samples.append(channels + global_noise)

    data = np.stack(all_samples, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data)
    print(f"  Trend (noise={noise_level:.2f}): {save_path}  shape={data.shape}")


# ============================================================
# Main: generate all noise levels
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    N = 150
    DATA_DIR = './dataset/syn_data'

    # Noise levels to sweep
    LAG_NOISES   = [0.05, 0.15, 0.30, 0.50]
    PERIOD_NOISES = [0.00, 0.30, 0.60, 1.00]  # extra noise on all channels
    TREND_NOISES  = [0.10, 0.30, 0.50, 0.80]

    print("=== Generating Lag datasets ===")
    for nl in LAG_NOISES:
        tag = f"{nl:.2f}".replace(".", "p")
        generate_lag(N, 192, tau=8,
                     noise_level=nl,
                     save_path=f'{DATA_DIR}/lag_8_150_n{tag}.npy')

    print("\n=== Generating Periodicity datasets ===")
    for nl in PERIOD_NOISES:
        tag = f"{nl:.2f}".replace(".", "p")
        generate_period(N, 200,
                        extra_noise=nl,
                        save_path=f'{DATA_DIR}/period_150_n{tag}.npy')

    print("\n=== Generating Trend datasets ===")
    for nl in TREND_NOISES:
        tag = f"{nl:.2f}".replace(".", "p")
        generate_trend(N, 192, input_len=96,
                       noise_level=nl,
                       save_path=f'{DATA_DIR}/trend_150_n{tag}.npy')

    print("\nAll datasets generated!")
