export CUDA_VISIBLE_DEVICES=0

model_name=PT_forecast_latent_v7
seed=2021

# # ==================== ETTh1 ====================
# for pred_len in 96 192; do
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_${pred_len} \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 2 \
#   --d_layers 3 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --d_ff 512 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1 \
#   --patch_len 4 \
#   --n_heads 8 \
#   --batch_size 128
# done

# # ==================== ETTh2 ====================
# for pred_len in 96 192; do
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_${pred_len} \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 2 \
#   --d_layers 3 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --d_ff 512 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --itr 1 \
#   --patch_len 4 \
#   --n_heads 8 \
#   --batch_size 128
# done


# ==================== Electricity ====================
for pred_len in 192; do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 3 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --seed $seed \
  --learning_rate 0.001 \
  --batch_size 8 \
  --itr 1 \
  --patch_len 4 \
  --n_heads 8
done

# # ==================== Traffic ====================
# for pred_len in 96 192; do
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_${pred_len} \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 2 \
#   --d_layers 3 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --d_model 256 \
#   --d_ff 512 \
#   --top_k 5 \
#   --des 'Exp' \
#   --seed $seed \
#   --learning_rate 0.001 \
#   --batch_size 4 \
#   --itr 1 \
#   --patch_len 2 \
#   --batch_size 8 \
#   --n_heads 4
# done
