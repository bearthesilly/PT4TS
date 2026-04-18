#!/bin/bash
# ============================================================================
# Periodicity synthetic experiment — DLinear baseline across sample sizes
# N in {150, 1500, 15000}
# ============================================================================
export CUDA_VISIBLE_DEVICES=0
seed=2025
ENC=10

for N in 150 1500 15000; do
  DATA_PATH=periodicity_${N}.npy
  MODEL_ID=syn_period_N${N}_dlinear

  echo ""; echo "========== Periodicity / DLinear / N=${N} =========="
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/syn_data/ --data_path $DATA_PATH \
    --model_id $MODEL_ID --model DLinear \
    --data toy --features M \
    --seq_len 96 --label_len 48 --pred_len 96 \
    --e_layers 2 --d_layers 1 --factor 3 \
    --enc_in $ENC --dec_in $ENC --c_out $ENC \
    --d_model 64 --d_ff 128 --top_k 5 --des Exp \
    --seed $seed --learning_rate 0.001 --itr 1 \
    --patch_len 1 --n_heads 1 --train_epochs 10
done
