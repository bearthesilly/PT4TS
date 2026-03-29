export CUDA_VISIBLE_DEVICES=0
seed=2025

# ---------- Generate synthetic data (deterministic, seed=2025) ----------
python -u toy_experiment_related/syn_temporal_decay_generation_v2.py

DATA_PATH=temporal_decay_v2_150.npy
MODEL_ID=syn_temporal_decay_v2_150_experiment
ENC=10

COMMON="--task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/syn_data/ --data_path $DATA_PATH \
  --model_id $MODEL_ID --data toy --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --e_layers 2 --d_layers 1 --factor 3 \
  --enc_in $ENC --dec_in $ENC --c_out $ENC \
  --d_model 64 --d_ff 128 --top_k 5 --des Exp \
  --seed $seed --learning_rate 0.001 --itr 1 \
  --patch_len 1 --n_heads 1 --train_epochs 10"

# 1) PT + temporal decay prior v2
python -u run.py $COMMON --model PT_syn_temporal_decay_v2

# 2) PT vanilla (no prior)
python -u run.py $COMMON --model PT_forecast_v15

# 3) DLinear
python -u run.py $COMMON --model DLinear

# 4) BVAR
python -u run.py $COMMON --model BVAR
