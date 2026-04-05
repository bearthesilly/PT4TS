#!/bin/bash
# ============================================================================
# Trend Prior (PtExplicitHMM) on Real-World Datasets
# Compare: PT_syn_trend (with HMM trend prior) vs PT_forecast_v15 (vanilla)
# Percent: 5% (few-shot) and 100% (full)
# Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange
# pred_len: 96 only (quick validation)
# ============================================================================
export CUDA_VISIBLE_DEVICES=0
seed=2021

PERCENTS="5 100"

run_pair() {
  # Usage: run_pair <extra_args>
  # Runs both PT_syn_trend and PT_forecast_v15 with the same args
  for pct in $PERCENTS; do
    local tag="${MODEL_ID}_p${pct}"

    echo "=== ${tag} / PT_forecast_v15 (vanilla) ==="
    python -u run.py \
      --task_name long_term_forecast --is_training 1 \
      --root_path $ROOT --data_path $DATA_PATH \
      --model_id ${tag} --model PT_forecast_v15 \
      --data $DATA --features M \
      --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
      --e_layers $E_LAYERS --d_layers 1 --factor 3 \
      --enc_in $ENC_IN --dec_in $ENC_IN --c_out $ENC_IN \
      --d_model $D_MODEL --d_ff $D_FF --top_k 5 \
      --des 'Exp' --itr 1 --seed $seed \
      --learning_rate $LR \
      --patch_len $PATCH_LEN --n_heads $N_HEADS \
      --percent $pct

    echo "=== ${tag} / PT_syn_trend (trend prior) ==="
    python -u run.py \
      --task_name long_term_forecast --is_training 1 \
      --root_path $ROOT --data_path $DATA_PATH \
      --model_id ${tag} --model PT_syn_trend \
      --data $DATA --features M \
      --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
      --e_layers $E_LAYERS --d_layers 1 --factor 3 \
      --enc_in $ENC_IN --dec_in $ENC_IN --c_out $ENC_IN \
      --d_model $D_MODEL --d_ff $D_FF --top_k 5 \
      --des 'Exp' --itr 1 --seed $seed \
      --learning_rate $LR \
      --patch_len $PATCH_LEN --n_heads $N_HEADS \
      --percent $pct
  done
}

# ============================================================
# ETTh1
# ============================================================
echo ""; echo "========== ETTh1 =========="
ROOT=./dataset/ETT-small/; DATA_PATH=ETTh1.csv; DATA=ETTh1
ENC_IN=7; SEQ_LEN=96; LABEL_LEN=48; PRED_LEN=96
E_LAYERS=4; D_MODEL=256; D_FF=512; LR=0.0001
PATCH_LEN=2; N_HEADS=2; MODEL_ID=ETTh1_trend_96_96
run_pair

# ============================================================
# ETTh2
# ============================================================
echo ""; echo "========== ETTh2 =========="
ROOT=./dataset/ETT-small/; DATA_PATH=ETTh2.csv; DATA=ETTh2
ENC_IN=7; SEQ_LEN=96; LABEL_LEN=48; PRED_LEN=96
E_LAYERS=4; D_MODEL=256; D_FF=1024; LR=0.000147
PATCH_LEN=2; N_HEADS=4; MODEL_ID=ETTh2_trend_96_96
run_pair

# ============================================================
# ETTm1
# ============================================================
echo ""; echo "========== ETTm1 =========="
ROOT=./dataset/ETT-small/; DATA_PATH=ETTm1.csv; DATA=ETTm1
ENC_IN=7; SEQ_LEN=96; LABEL_LEN=48; PRED_LEN=96
E_LAYERS=3; D_MODEL=256; D_FF=1024; LR=0.00007
PATCH_LEN=3; N_HEADS=8; MODEL_ID=ETTm1_trend_96_96
run_pair

# ============================================================
# ETTm2
# ============================================================
echo ""; echo "========== ETTm2 =========="
ROOT=./dataset/ETT-small/; DATA_PATH=ETTm2.csv; DATA=ETTm2
ENC_IN=7; SEQ_LEN=96; LABEL_LEN=48; PRED_LEN=96
E_LAYERS=3; D_MODEL=256; D_FF=512; LR=0.000397
PATCH_LEN=4; N_HEADS=8; MODEL_ID=ETTm2_trend_96_96
run_pair

# ============================================================
# Weather
# ============================================================
echo ""; echo "========== Weather =========="
ROOT=./dataset/weather/; DATA_PATH=weather.csv; DATA=custom
ENC_IN=21; SEQ_LEN=96; LABEL_LEN=48; PRED_LEN=96
E_LAYERS=2; D_MODEL=256; D_FF=512; LR=0.00027
PATCH_LEN=4; N_HEADS=8; MODEL_ID=weather_trend_96_96
run_pair

# ============================================================
# Exchange
# ============================================================
echo ""; echo "========== Exchange =========="
ROOT=./dataset/exchange_rate/; DATA_PATH=exchange_rate.csv; DATA=custom
ENC_IN=8; SEQ_LEN=96; LABEL_LEN=48; PRED_LEN=96
E_LAYERS=2; D_MODEL=128; D_FF=256; LR=0.00015
PATCH_LEN=4; N_HEADS=4; MODEL_ID=exchange_trend_96_96
run_pair

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "All done! Grep results with:"
echo "  grep -A1 'trend_96_96' result_long_term_forecast.txt"
echo "========================================"
