"""
Unified data generator for the scaling study (N in {150, 1500, 15000})
across the three working synthetic experiments: lag / periodicity / trend.

Produces 9 .npy files under dataset/syn_data/ that match the file-name
convention used by the existing training scripts.
"""
import numpy as np
import os
import sys

# Make sibling generator modules importable when run from repo root
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from syn_lag_generation import CausalGenerator
from syn_period_generation import PeriodicityGenerator
from syn_trend_generation import TrendGenerator


SAMPLE_SIZES = [150, 1500, 15000]
OUT_DIR = './dataset/syn_data'
SEED = 42


def gen_lag(n):
    np.random.seed(SEED)
    tau = 8
    gen = CausalGenerator(num_samples=n, sample_len=192, tau=tau, noise_level=0.05)
    path = os.path.join(OUT_DIR, f'lag_{tau}_{n}.npy')
    gen.generate(save_path=path)


def gen_period(n):
    np.random.seed(SEED)
    gen = PeriodicityGenerator(num_samples=n, sample_len=200)
    path = os.path.join(OUT_DIR, f'periodicity_{n}.npy')
    gen.generate(save_path=path)


def gen_trend(n):
    np.random.seed(SEED)
    gen = TrendGenerator(num_samples=n, sample_len=192, input_len=96, noise_level=0.1)
    path = os.path.join(OUT_DIR, f'trend_{n}.npy')
    gen.generate(save_path=path)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    for n in SAMPLE_SIZES:
        print(f'\n========= N = {n} =========')
        gen_lag(n)
        gen_period(n)
        gen_trend(n)
    print('\nAll 9 datasets generated.')
