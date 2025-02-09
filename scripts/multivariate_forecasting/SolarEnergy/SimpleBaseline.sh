model_name=SimpleBaseline
python -u run_ca.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar \
  --model "$model_name" \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.006 \
  --batch_size 256 \
  --fix_seed 2025 \
  --use_norm 0 \
  --wv "db8" \
  --m 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \ 
  --des 'Exp' \
  --itr 3 \
  --use_norm 0 \ 
  --alpha 0.6 \
  --l1_weight 5e-5
        


