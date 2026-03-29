#!/bin/bash
# ============================================================================
# ETTh2 Small-Sample Experiment — ST-PT with Voltage-Level Channel Group Prior
# Models: BVAR, DLinear, PT_forecast_v15 (vanilla), PT_ETT_prior (with prior)
# Percent: 5%, 10%, 20%, 100%    Pred_len: 96, 192, 336, 720
# ============================================================================
export CUDA_VISIBLE_DEVICES=0
seed=2021

PERCENTS="5 10 20 100"
PRED_LENS="96 192 336 720"

ROOT=./dataset/ETT-small/
DATA_PATH=ETTh2.csv
DATA=ETTh2
ENC_IN=7
SEQ_LEN=96
LABEL_LEN=48

# PT hypers for ETTh2
E_LAYERS=4
D_MODEL=256
D_FF=1024
LR=0.000147
PATCH_LEN=2
N_HEADS=4

for pct in $PERCENTS; do
for pl in $PRED_LENS; do

tag="ETTh2_p${pct}_96_${pl}"

# ---------- BVAR ----------
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $ROOT \
#   --data_path $DATA_PATH \
#   --model_id ${tag} \
#   --model BVAR \
#   --data $DATA \
#   --features M \
#   --seq_len $SEQ_LEN \
#   --label_len $LABEL_LEN \
#   --pred_len $pl \
#   --enc_in $ENC_IN \
#   --dec_in $ENC_IN \
#   --c_out $ENC_IN \
#   --des 'Exp' \
#   --itr 1 \
#   --seed $seed \
#   --percent $pct

# # ---------- DLinear ----------
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $ROOT \
#   --data_path $DATA_PATH \
#   --model_id ${tag} \
#   --model DLinear \
#   --data $DATA \
#   --features M \
#   --seq_len $SEQ_LEN \
#   --label_len $LABEL_LEN \
#   --pred_len $pl \
#   --enc_in $ENC_IN \
#   --dec_in $ENC_IN \
#   --c_out $ENC_IN \
#   --des 'Exp' \
#   --itr 1 \
#   --seed $seed \
#   --percent $pct

# ---------- PT_forecast_v15 (vanilla) ----------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT \
  --data_path $DATA_PATH \
  --model_id ${tag} \
  --model PT_forecast_v15 \
  --data $DATA \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $pl \
  --e_layers $E_LAYERS \
  --d_layers 1 \
  --factor 3 \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --seed $seed \
  --learning_rate $LR \
  --patch_len $PATCH_LEN \
  --n_heads $N_HEADS \
  --percent $pct

# ---------- PT_ETT_prior (channel group + decay + daily periodicity) ----------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT \
  --data_path $DATA_PATH \
  --model_id ${tag} \
  --model PT_ETT_prior \
  --data $DATA \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $pl \
  --e_layers $E_LAYERS \
  --d_layers 1 \
  --factor 3 \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --seed $seed \
  --learning_rate $LR \
  --patch_len $PATCH_LEN \
  --n_heads $N_HEADS \
  --percent $pct \
  --decay_alpha 0.15 \
  --period_len 24 \
  --channel_group_str "0,1|2,3|4,5|6"

done
done
