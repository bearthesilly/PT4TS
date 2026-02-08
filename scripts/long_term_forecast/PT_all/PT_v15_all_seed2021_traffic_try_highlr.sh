export CUDA_VISIBLE_DEVICES=0
model_name=PT_forecast_v15
seed=2021

# Traffic
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 5 \
  --batch_size 16 \
  --seed $seed \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 12 \
  --n_heads 8

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 5 \
  --batch_size 16 \
  --seed $seed \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 12 \
  --n_heads 8

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 5 \
  --batch_size 16 \
  --seed $seed \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 12 \
  --n_heads 8

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 5 \
  --batch_size 16 \
  --seed $seed \
  --learning_rate 0.005 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 12 \
  --n_heads 8