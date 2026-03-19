export CUDA_VISIBLE_DEVICES=0
model_name=PT_syn_lag
seed=2025

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/syn_data/ \
  --data_path lag_8_15000.npy \
  --model_id syn_lag_15000_experiment \
  --model $model_name \
  --data toy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \
  --des 'Exp' \
  --seed $seed \
  --learning_rate 0.001 \
  --itr 1 \
  --patch_len 1 \
  --n_heads 1