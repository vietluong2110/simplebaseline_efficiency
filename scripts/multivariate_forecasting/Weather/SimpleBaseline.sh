
model_name=SimpleBaseline

python -u run_ca.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather \
  --model "$model_name" \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 32 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 1 \
  --wv "db4" \
  --m 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \ 
  --des 'Exp' \
  --itr 3 \
  --alpha 0.6 \
  --l1_weight 5e-5
      
  