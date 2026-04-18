export CUDA_VISIBLE_DEVICES=1
#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_LIST="AutoTimes DeepVAR LSTM_AR TCN_AR LSTNet" bash "$script_dir/run_ar_baseline.sh"
