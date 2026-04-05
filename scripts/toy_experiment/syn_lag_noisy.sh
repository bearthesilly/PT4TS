export CUDA_VISIBLE_DEVICES=0
seed=2025

DATA_PATH=lag_8_150_noisy.npy
MODEL_ID=syn_lag_150_noisy_experiment
ENC=6

COMMON="--task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/syn_data/ --data_path $DATA_PATH \
  --model_id $MODEL_ID --data toy --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --e_layers 2 --d_layers 1 --factor 3 \
  --enc_in $ENC --dec_in $ENC --c_out $ENC \
  --d_model 64 --d_ff 128 --top_k 5 --des Exp \
  --seed $seed --learning_rate 0.001 --itr 1 \
  --patch_len 1 --n_heads 1 --train_epochs 10"

# 1) PT + lag prior
python -u run.py $COMMON --model PT_syn_lag

# 2) PT vanilla (no prior)
python -u run.py $COMMON --model PT_forecast_v15

# 3) DLinear
python -u run.py $COMMON --model DLinear

# 4) BVAR
python -u run.py $COMMON --model BVAR
