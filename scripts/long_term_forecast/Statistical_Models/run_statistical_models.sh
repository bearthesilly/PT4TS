#!/bin/bash
# ============================================================================
# Statistical Multivariate Time Series Forecasting Models
# ============================================================================
# Models: VAR, GPVAR, DynamicFactorModel, BVAR, StateSpaceModel
#
# These are non-learning statistical baselines that fit per-sample at
# inference time. The training loop runs 1 epoch (no backprop), computes
# validation/test metrics, and saves a checkpoint for framework consistency.
#
# Usage:
#   bash scripts/long_term_forecast/Statistical_Models/run_statistical_models.sh
#
# To run a single model, set MODEL_LIST before sourcing, e.g.:
#   MODEL_LIST="VAR" bash scripts/.../run_statistical_models.sh
# ============================================================================

export CUDA_VISIBLE_DEVICES=0
seed=2021

# ---------- Configurable knobs ----------
MODEL_LIST="${MODEL_LIST:-VAR BVAR GPVAR DynamicFactorModel StateSpaceModel}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"

# ---------- Dataset catalogue ----------
# Format:  DATA_KEY  ROOT_PATH  DATA_PATH  ENC_IN  DATA_LOADER
datasets=(
    "ETTh1      ./dataset/ETT-small/    ETTh1.csv           7   ETTh1"
    "ETTh2      ./dataset/ETT-small/    ETTh2.csv           7   ETTh2"
    "ETTm1      ./dataset/ETT-small/    ETTm1.csv           7   ETTm1"
    "ETTm2      ./dataset/ETT-small/    ETTm2.csv           7   ETTm2"
    "weather    ./dataset/weather/      weather.csv         21  custom"
    "exchange   ./dataset/exchange_rate/ exchange_rate.csv   8   custom"
    "electricity ./dataset/electricity/ electricity.csv     321 custom"
)

# GPVAR is O(n^3) per dimension — cap batch size for high-dim datasets
get_batch_size() {
    local model=$1
    local enc_in=$2
    if [[ "$model" == "GPVAR" ]]; then
        if (( enc_in > 50 )); then
            echo 4
        elif (( enc_in > 10 )); then
            echo 8
        else
            echo 16
        fi
    else
        echo 32
    fi
}

# ---------- Main loop ----------
for ds_line in "${datasets[@]}"; do
    read -r ds_name root_path data_path enc_in data_key <<< "$ds_line"

    for model_name in $MODEL_LIST; do
        for pred_len in $PRED_LENS; do

            batch_size=$(get_batch_size "$model_name" "$enc_in")

            echo "=============================================="
            echo "  Model: $model_name | Dataset: $ds_name"
            echo "  pred_len: $pred_len | batch: $batch_size"
            echo "=============================================="

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
                --d_model 64 \
                --d_ff 128 \
                --e_layers 1 \
                --d_layers 1 \
                --n_heads 1 \
                --des 'Exp' \
                --seed "$seed" \
                --learning_rate 0.001 \
                --batch_size "$batch_size" \
                --train_epochs 1 \
                --itr 1

        done
    done
done

echo ""
echo "All statistical model experiments finished."
echo "Results appended to: result_long_term_forecast.txt"
