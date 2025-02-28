export CUDA_VISIBLE_DEVICES=6
model_name=SimpleBaseline

python -u run_ca.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 10 \
  --train_epochs 20 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03 \
  --model "$model_name" \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 512 \
  --learning_rate 0.002 \
  --batch_size 16 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv 'bior3.1' \
  --m 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.1 \
  --use_norm 0 \
  --l1_weight 0.005

# python -u run_ca.py \
#   --is_training 1 \
#   --lradj 'TST' \
#   --patience 10 \
#   --train_epochs 20 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS03.npz \
#   --model_id PEMS03 \
#   --model "$model_name" \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 24 \
#   --e_layers 1 \
#   --d_model 256 \
#   --d_ff 512 \
#   --learning_rate 0.002 \
#   --batch_size 16 \
#   --fix_seed 2025 \
#   --use_norm 1 \
#   --wv 'bior3.1' \
#   --m 3 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --des 'Exp' \
#   --itr 3 \
#   --alpha 0.1 \
#   --use_norm 0 \
#   --l1_weight 0.005

# python -u run_ca.py \
#   --is_training 1 \
#   --lradj 'TST' \
#   --patience 10 \
#   --train_epochs 20 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS03.npz \
#   --model_id PEMS03 \
#   --model "$model_name" \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 48 \
#   --e_layers 1 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --learning_rate 0.002 \
#   --batch_size 16 \
#   --fix_seed 2025 \
#   --use_norm 0 \
#   --wv 'bior3.1' \
#   --m 3 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --des 'Exp' \
#   --itr 3 \
#   --alpha 0.1 \
#   --l1_weight 0.005

# python -u run_ca.py \
#   --is_training 1 \
#   --lradj 'TST' \
#   --patience 10 \
#   --train_epochs 20 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS03.npz \
#   --model_id PEMS03 \
#   --model "$model_name" \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 1 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --learning_rate 0.002 \
#   --batch_size 16 \
#   --fix_seed 2025 \
#   --use_norm 0 \
#   --wv 'bior3.1' \
#   --m 3 \
#   --enc_in 358 \
#   --dec_in 358 \
#   --c_out 358 \
#   --des 'Exp' \
#   --itr 3 \
#   --alpha 0.1 \
#   --l1_weight 0.005