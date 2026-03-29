#!/bin/bash
# ============================================================================
# Weather Small-Sample Experiment — ST-PT with Physical Variable Grouping Prior
#
# Channel grouping (Jena Climate, 21 vars, 0-indexed after dropping 'date'):
#   Temperature : {1,2,3,19}  — T, Tpot, Tdew, Tlog
#   Moisture    : {4,5,6,7,8,9} — rh, VPmax, VPact, VPdef, sh, H2OC
#   Pressure    : {0,10}      — p, rho  (ideal gas law)
#   Wind        : {11,12,13}  — wv, max.wv, wd
#   Precipitation: {14,15}    — rain, raining
#   Radiation   : {16,17,18}  — SWDR, PAR, max.PAR
#   Target      : {20}        — OT
#
# Temporal prior: decay only (window=96×10min=16h, < 1 day → no periodicity)
#
# Models: BVAR, DLinear, PT_forecast_v15 (vanilla), PT_ETT_prior (with prior)
# Percent: 5%, 10%, 20%, 100%    Pred_len: 96, 192, 336, 720
# ============================================================================
export CUDA_VISIBLE_DEVICES=0
seed=2021

PERCENTS="5 10 20 100"
PRED_LENS="96 192 336 720"

ROOT=./dataset/weather/
DATA_PATH=weather.csv
DATA=custom
ENC_IN=21
SEQ_LEN=96
LABEL_LEN=48

# PT hypers for Weather (from PT_v15_all_seed2021.sh)
E_LAYERS=2
D_MODEL=128
D_FF=256
LR=0.0002
PATCH_LEN=4
N_HEADS=8

# Physical variable grouping for Weather
GROUPS="1,2,3,19|4,5,6,7,8,9|0,10|11,12,13|14,15|16,17,18|20"

for pct in $PERCENTS; do
for pl in $PRED_LENS; do

tag="weather_p${pct}_96_${pl}"

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

# ---------- DLinear ----------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $ROOT \
  --data_path $DATA_PATH \
  --model_id ${tag} \
  --model DLinear \
  --data $DATA \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $pl \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --des 'Exp' \
  --itr 1 \
  --seed $seed \
  --percent $pct

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

# ---------- PT_ETT_prior (channel group + temporal decay) ----------
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
  --channel_group_str "$GROUPS" \
  --period_beta 0.5

done
done
