export CUDA_VISIBLE_DEVICES=0
model_name=PatchTST
seed=2021

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/toy_data/ \
#   --data_path toy_sine.csv \
#   --model_id toy_sine_experiment \
#   --model PT \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 3 \
#   --dec_in 3 \
#   --c_out 3 \
#   --des 'Exp_Toy_Sine_Wave' \
#   --d_model 64 \
#   --d_ff 128 \
#   --itr 1 \
#   --train_epochs 5 \
#   --batch_size 32 \
#   --learning_rate 0.0001

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/toy_data/ \
  --data_path toy_complicated.csv \
  --model_id toy_complicated_experiment \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \
  --des 'Exp' \
  --seed $seed \
  --learning_rate 0.001 \
  --itr 1 \
  --patch_len 8 \
  --n_heads 1