#!/bin/bash
# ============================================================
# Noise Robustness Sweep: Prior vs Vanilla across noise levels
# Usage: bash scripts/toy_experiment/noise_robustness_all.sh
# ============================================================
export CUDA_VISIBLE_DEVICES=0
seed=2025

# Step 1: Generate all datasets
echo "========================================"
echo "Step 1: Generating datasets..."
echo "========================================"
python toy_experiment_related/generate_noise_sweep.py

# Common training args (shared across all experiments)
BASE="--task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/syn_data/ \
  --data toy --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --e_layers 2 --d_layers 1 --factor 3 \
  --d_model 64 --d_ff 128 --top_k 5 --des Exp \
  --seed $seed --learning_rate 0.001 --itr 1 \
  --patch_len 1 --n_heads 1 --train_epochs 10"

# ============================================================
# Step 2: Run LAG experiments
# ============================================================
echo ""
echo "========================================"
echo "Step 2: LAG experiments"
echo "========================================"

for NOISE in 0p05 0p15 0p30 0p50; do
  DATA_PATH="lag_8_150_n${NOISE}.npy"
  MODEL_ID="lag_noise_${NOISE}"
  ENC=6
  COMMON="$BASE --data_path $DATA_PATH --model_id $MODEL_ID --enc_in $ENC --dec_in $ENC --c_out $ENC"

  echo "--- Lag noise=$NOISE: PT_syn_lag (prior) ---"
  python -u run.py $COMMON --model PT_syn_lag

  echo "--- Lag noise=$NOISE: PT_forecast_v15 (vanilla) ---"
  python -u run.py $COMMON --model PT_forecast_v15
done

# ============================================================
# Step 3: Run PERIODICITY experiments
# ============================================================
echo ""
echo "========================================"
echo "Step 3: PERIODICITY experiments"
echo "========================================"

for NOISE in 0p00 0p30 0p60 1p00; do
  DATA_PATH="period_150_n${NOISE}.npy"
  MODEL_ID="period_noise_${NOISE}"
  ENC=10
  COMMON="$BASE --data_path $DATA_PATH --model_id $MODEL_ID --enc_in $ENC --dec_in $ENC --c_out $ENC"

  echo "--- Period noise=$NOISE: PT_syn_period (prior) ---"
  python -u run.py $COMMON --model PT_syn_period

  echo "--- Period noise=$NOISE: PT_forecast_v15 (vanilla) ---"
  python -u run.py $COMMON --model PT_forecast_v15
done

# ============================================================
# Step 4: Run TREND experiments
# ============================================================
echo ""
echo "========================================"
echo "Step 4: TREND experiments"
echo "========================================"

for NOISE in 0p10 0p30 0p50 0p80; do
  DATA_PATH="trend_150_n${NOISE}.npy"
  MODEL_ID="trend_noise_${NOISE}"
  ENC=10
  COMMON="$BASE --data_path $DATA_PATH --model_id $MODEL_ID --enc_in $ENC --dec_in $ENC --c_out $ENC"

  echo "--- Trend noise=$NOISE: PT_syn_trend (prior) ---"
  python -u run.py $COMMON --model PT_syn_trend

  echo "--- Trend noise=$NOISE: PT_forecast_v15 (vanilla) ---"
  python -u run.py $COMMON --model PT_forecast_v15
done

# ============================================================
# Step 5: Analyze results
# ============================================================
echo ""
echo "========================================"
echo "Step 5: Analyzing results..."
echo "========================================"
python toy_experiment_related/analyze_noise_robustness.py

echo ""
echo "Done! Check noise_robustness_report.txt and noise_robustness_plot.pdf"
