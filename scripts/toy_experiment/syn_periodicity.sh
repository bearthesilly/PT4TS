export CUDA_VISIBLE_DEVICES=0
model_name=BVAR
seed=2021

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/syn_data/ \
  --data_path periodicity_150.npy \
  --model_id syn_period_150_experiment \
  --model $model_name \
  --data toy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \
  --des 'Exp' \
  --seed $seed \
  --learning_rate 0.001 \
  --itr 1 \
  --patch_len 1 \
  --n_heads 1 \
  --train_epochs 10