#!/bin/bash
# 与 `writing/AR_baseline_and_latent_v7_merged.md` 对齐：仅 DeepVAR / LSTM_AR / LSTNet
# （不含 AutoTimes、TCN_AR）。数据集见 `run_ar_baseline.sh`（ETTh1/2、m1/2、ECL、weather × pl 96/192）。
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_LIST="DeepVAR LSTM_AR LSTNet" bash "$script_dir/run_ar_baseline.sh"

# 若历史日志仅含 ETTh1/ETTh2/ETTm1，可只补 ETTm2、ECL、weather：
# DATASET_FILTER="ETTm2 ECL weather" MODEL_LIST="DeepVAR LSTM_AR LSTNet" bash "$script_dir/run_ar_baseline.sh"
