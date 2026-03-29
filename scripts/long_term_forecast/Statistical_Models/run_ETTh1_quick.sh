#!/bin/bash
# ============================================================================
# Quick test: all 5 statistical models on ETTh1, pred_len=96
# ============================================================================

# export CUDA_VISIBLE_DEVICES=0
seed=2021

for model_name in VAR BVAR GPVAR DynamicFactorModel StateSpaceModel; do

    echo "====== Running $model_name on ETTh1 (pred_len=96) ======"

    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_${model_name}_96_96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 64 \
        --d_ff 128 \
        --e_layers 1 \
        --d_layers 1 \
        --n_heads 1 \
        --des 'Exp' \
        --seed $seed \
        --learning_rate 0.001 \
        --batch_size 32 \
        --train_epochs 1 \
        --itr 1 \
        --use_gpu False

done

echo ""
echo "Quick test finished. Check result_long_term_forecast.txt for metrics."
