#!/bin/bash
# ============================================================================
# Statistical models on synthetic datasets
# ============================================================================
# Mirrors syn_periodicity.sh but runs all 5 statistical baselines.
# Adapt DATA_PATH and ENC_IN for your own .npy files.
# ============================================================================

export CUDA_VISIBLE_DEVICES=0
seed=2021

# ---------- Synthetic dataset config ----------
ROOT_PATH=./dataset/syn_data/
DATA_PATH="${DATA_PATH:-periodicity_150.npy}"
ENC_IN="${ENC_IN:-10}"

SEQ_LEN="${SEQ_LEN:-96}"
PRED_LEN="${PRED_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"

for model_name in VAR BVAR GPVAR DynamicFactorModel StateSpaceModel; do

    echo "====== $model_name | $DATA_PATH (enc_in=$ENC_IN) ======"

    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --model_id "syn_${model_name}_sl${SEQ_LEN}_pl${PRED_LEN}" \
        --model "$model_name" \
        --data toy \
        --features M \
        --seq_len "$SEQ_LEN" \
        --label_len "$LABEL_LEN" \
        --pred_len "$PRED_LEN" \
        --enc_in "$ENC_IN" \
        --dec_in "$ENC_IN" \
        --c_out "$ENC_IN" \
        --d_model 64 \
        --d_ff 128 \
        --e_layers 1 \
        --d_layers 1 \
        --n_heads 1 \
        --des 'Exp' \
        --seed "$seed" \
        --learning_rate 0.001 \
        --batch_size 32 \
        --train_epochs 1 \
        --itr 1

done

echo ""
echo "Done. Results in result_long_term_forecast.txt"
