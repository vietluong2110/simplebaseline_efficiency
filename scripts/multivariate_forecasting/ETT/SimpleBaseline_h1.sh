model_name=SimpleBaseline
python -u run_ca.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \ 
  --model_id ETTh1 \ 
  --model "$model_name" \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
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
  --alpha 0.6 \
  --l1_weight 5e-5

