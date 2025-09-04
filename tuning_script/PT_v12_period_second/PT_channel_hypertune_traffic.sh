export CUDA_VISIBLE_DEVICES=0

model_name=PT_forecast_v12_period_second
seed=2021
# Weather
# python -u run.py \
#   --use_optuna \
#   --n_trials 100 \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Optuna_Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1 \
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --learning_rate 0.001 \
#   --seed $seed \
#   --itr 1 \


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1 \

# # Electricity
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# Traffic
python -u run.py \
  --use_optuna \
  --n_trials 50 \
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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 256 \
  --d_ff 1024 \
  --top_k 5 \
  --batch_size 32 \
  --seed $seed \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --d_model 256\
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1 

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --d_model 256\
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --d_model 256\
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1
# # Exchange
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1 \
# # ETTh1
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --des 'Exp' \
#   --itr 1 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --top_k 5 \


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_192 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --des 'Exp' \
#   --itr 1 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --top_k 5


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --des 'Exp' \
#   --itr 1 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --top_k 5


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --des 'Exp' \
#   --itr 1 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --top_k 5
# # ETTh2
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_96 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_192 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_336 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_720 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1
# # ETTm1
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_96 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_192 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_336 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_720 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1
# # ETTm2
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_96 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_192 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1 

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_336 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_720 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 1024 \
#   --top_k 5 \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1
