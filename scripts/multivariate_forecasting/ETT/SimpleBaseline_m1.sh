 export CUDA_VISIBLE_DEVICES=7
model_name=JAX_SimpleBaseline
# python -u jax_run_ca.py \
#   --is_training 1 \
#   --lradj 'TST' \
#   --patience 3 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1 \
#   --model "$model_name" \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 1 \
#   --d_model 32 \
#   --d_ff 32 \
#   --learning_rate 0.02 \
#   --batch_size 256 \
#   --fix_seed 2025 \
#   --use_norm 1 \
#   --wv 'db1' \
#   --m 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 3 \
#   --alpha 0.1 \
#   --l1_weight 0.005

# python -u jax_run_ca.py \
#   --is_training 1 \
#   --lradj 'TST' \
#   --patience 3 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1 \
#   --model "$model_name" \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --e_layers 1 \
#   --d_model 32 \
#   --d_ff 32 \
#   --learning_rate 0.02 \
#   --batch_size 256 \
#   --fix_seed 2025 \
#   --use_norm 1 \
#   --wv 'db1' \
#   --m 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 3 \
#   --alpha 0.1 \
#   --l1_weight 0.005

# python -u jax_run_ca.py \
#   --is_training 1 \
#   --lradj 'TST' \
#   --patience 3 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1 \
#   --model "$model_name" \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --e_layers 1 \
#   --d_model 32 \
#   --d_ff 32 \
#   --learning_rate 0.02 \
#   --batch_size 256 \
#   --fix_seed 2025 \
#   --use_norm 1 \
#   --wv 'db1' \
#   --m 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 3 \
#   --alpha 0.1 \
#   --l1_weight 0.005

python -u jax_run_ca.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1 \
  --model "$model_name" \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate 0.02 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv 'db1' \
  --m 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.1 \
  --l1_weight 0.005