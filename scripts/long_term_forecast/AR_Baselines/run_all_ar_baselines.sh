#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_LIST="AutoTimes DeepVAR GPVAR LSTM_AR TCN_AR LSTNet" bash "$script_dir/run_ar_baseline.sh"
