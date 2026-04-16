#!/bin/bash
# Generic runner for AR / AR-ish baselines under the unified LTSF setting:
#   datasets: ETTh1, ETTh2, ETTm1, ETTm2, ECL, weather
#   seq_len: 96
#   pred_len: 96, 192
#
# Example:
#   MODEL_LIST="LSTM_AR TCN_AR" bash scripts/long_term_forecast/AR_Baselines/run_ar_baseline.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
seed="${SEED:-2021}"
pred_lens="${PRED_LENS:-96 192}"
model_list="${MODEL_LIST:-AutoTimes DeepVAR GPVAR LSTM_AR TCN_AR LSTNet}"

datasets=(
  "ETTh1   ./dataset/ETT-small/   ETTh1.csv       ETTh1   7"
  "ETTh2   ./dataset/ETT-small/   ETTh2.csv       ETTh2   7"
  "ETTm1   ./dataset/ETT-small/   ETTm1.csv       ETTm1   7"
  "ETTm2   ./dataset/ETT-small/   ETTm2.csv       ETTm2   7"
  "ECL     ./dataset/electricity/ electricity.csv custom  321"
  "weather ./dataset/weather/     weather.csv     custom  21"
)

set_hparams() {
  local model_name="$1"
  local enc_in="$2"

  d_model=128
  d_ff=256
  e_layers=2
  d_layers=1
  n_heads=4
  dropout=0.1
  learning_rate=0.001
  train_epochs=10
  patience=3
  batch_size=32
  ar_hidden=128
  ar_layers=2
  teacher_forcing_ratio=1.0
  patch_len=16
  lag_order=5
  gp_max_train_points=96
  tcn_kernel_size=3
  cnn_kernel=6
  highway_window=24
  skip=24
  skip_hidden=5
  lstnet_cnn_hidden=64

  case "$model_name" in
    AutoTimes)
      d_model=256
      d_ff=512
      e_layers=2
      n_heads=4
      learning_rate=0.0001
      patch_len=96
      batch_size=32
      ;;
    DeepVAR)
      d_model=128
      d_ff=256
      ar_hidden=128
      ar_layers=2
      learning_rate=0.001
      teacher_forcing_ratio=1.0
      ;;
    GPVAR)
      d_model=64
      d_ff=128
      n_heads=1
      train_epochs=1
      learning_rate=0.001
      lag_order=5
      if (( enc_in > 100 )); then
        batch_size=1
        gp_max_train_points=32
      elif (( enc_in > 20 )); then
        batch_size=4
        gp_max_train_points=64
      else
        batch_size=16
        gp_max_train_points=96
      fi
      ;;
    LSTM_AR)
      d_model=128
      d_ff=256
      ar_hidden=128
      ar_layers=2
      learning_rate=0.001
      teacher_forcing_ratio=1.0
      ;;
    TCN_AR)
      d_model=128
      d_ff=256
      ar_hidden=128
      ar_layers=3
      e_layers=3
      learning_rate=0.001
      tcn_kernel_size=3
      teacher_forcing_ratio=1.0
      ;;
    LSTNet)
      d_model=100
      d_ff=200
      ar_hidden=100
      lstnet_cnn_hidden=64
      skip_hidden=5
      cnn_kernel=6
      highway_window=24
      skip=24
      learning_rate=0.001
      ;;
    *)
      echo "Unknown model: $model_name" >&2
      exit 1
      ;;
  esac

  if (( enc_in > 100 )) && [[ "$model_name" != "GPVAR" ]]; then
    batch_size=16
  fi
}

for ds_line in "${datasets[@]}"; do
  read -r ds_name root_path data_path data_key enc_in <<< "$ds_line"

  for model_name in $model_list; do
    set_hparams "$model_name" "$enc_in"

    for pred_len in $pred_lens; do
      echo "============================================================"
      echo "Model: $model_name | Dataset: $ds_name | seq_len: 96 | pred_len: $pred_len"
      echo "============================================================"

      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$root_path" \
        --data_path "$data_path" \
        --model_id "${ds_name}_${model_name}_sl96_pl${pred_len}" \
        --model "$model_name" \
        --data "$data_key" \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len "$pred_len" \
        --enc_in "$enc_in" \
        --dec_in "$enc_in" \
        --c_out "$enc_in" \
        --d_model "$d_model" \
        --d_ff "$d_ff" \
        --e_layers "$e_layers" \
        --d_layers "$d_layers" \
        --n_heads "$n_heads" \
        --dropout "$dropout" \
        --learning_rate "$learning_rate" \
        --train_epochs "$train_epochs" \
        --patience "$patience" \
        --batch_size "$batch_size" \
        --ar_hidden "$ar_hidden" \
        --ar_layers "$ar_layers" \
        --teacher_forcing_ratio "$teacher_forcing_ratio" \
        --patch_len "$patch_len" \
        --lag_order "$lag_order" \
        --gp_max_train_points "$gp_max_train_points" \
        --tcn_kernel_size "$tcn_kernel_size" \
        --cnn_kernel "$cnn_kernel" \
        --highway_window "$highway_window" \
        --skip "$skip" \
        --skip_hidden "$skip_hidden" \
        --lstnet_cnn_hidden "$lstnet_cnn_hidden" \
        --des 'ARBaseline' \
        --seed "$seed" \
        --itr 1
    done
  done
done

echo "AR baseline experiments finished. Results are appended to result_long_term_forecast.txt"
