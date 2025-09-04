export CUDA_VISIBLE_DEVICES=0

python run.py \
--seed 3407 \
--task_name classification \
--is_training 1 \
--root_path ./dataset/Handwriting/ \
--model_id Handwriting \
--model PT \
--data UEA \
--e_layers 2 \
--batch_size 2 \
--d_model 32 \
--d_ff 64 \
--top_k 3 \
--des 'Exp' \
--itr 1 \
--learning_rate 0.001 \
--train_epochs 50 \
--patience 10

python -u run.py \
  --seed 3407 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model PT \
  --data UEA \
  --e_layers 3 \
  --batch_size 2 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 1 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10

python -u run.py \
  --seed 3407 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model PT \
  --data UEA \
  --e_layers 2 \
  --batch_size 2 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 60 \
  --patience 10

python -u run.py \
  --seed 3407 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PEMS-SF/ \
  --model_id PEMS-SF \
  --model PT \
  --data UEA \
  --e_layers 6 \
  --batch_size 2 \
  --d_model 64 \
  --d_ff 64 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10

python -u run.py \
  --seed 3407 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model PT \
  --data UEA \
  --e_layers 3 \
  --batch_size 2 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10

python -u run.py \
  --seed 3407 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model PT \
  --data UEA \
  --e_layers 3 \
  --batch_size 2 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10

python -u run.py \
  --seed 3407 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model PT \
  --data UEA \
  --e_layers 2 \
  --batch_size 2 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 2 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10

python -u run.py \
  --seed 3407 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model PT \
  --data UEA \
  --e_layers 2 \
  --batch_size 2 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10

python -u run.py \
  --seed 3407 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model PT \
  --data UEA \
  --e_layers 2 \
  --batch_size 2 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10

  
python -u run.py \
  --seed 3407 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model PT \
  --data UEA \
  --e_layers 2 \
  --batch_size 4 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --num_kernels 4 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 10