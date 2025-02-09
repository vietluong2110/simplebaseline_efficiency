model_name=SimpleBaseline
python -u run_ca.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model "$model_name" \ 
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --d_model 256 \
  --d_ff 1024 \
  --learning_rate 0.009 \
  --batch_size 256 \
  --fix_seed 2025 \ 
  --use_norm 1 \
  --wv "db1" \
  --m 3 \
  --enc_in 321 \
  --dec_in 321 \ 
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --alpha 0.6 \
  --l1_weight 5e-5
        
      
  

