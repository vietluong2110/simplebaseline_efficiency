export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# for seed in 2021 2022 2023 2024 2025; do
#   python -u run.py \
#     --is_training 1 \
#     --root_path ../Time-LLM/dataset/electricity/ \
#     --data_path electricity.csv \
#     --model_id ECL_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --e_layers 1 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des 'Exp' \
#     --d_model 512 \
#     --d_ff 512 \
#     --batch_size 16 \
#     --itr 1 \
#     --learning_rate 0.0005 \
#     --fix_seed $seed >./logs_ECL/'predlen96_seed'$seed.log 
# done

# for seed in 2021 2022 2023 2024 2025; do
#   python -u run.py \
#     --is_training 1 \
#     --root_path ../Time-LLM/dataset/electricity/ \
#     --data_path electricity.csv \
#     --model_id ECL_96_192 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 192 \
#     --e_layers 3 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des 'Exp' \
#     --d_model 512 \
#     --d_ff 512 \
#     --batch_size 16 \
#     --learning_rate 0.0005 \
#     --itr 1 \
#     --fix_seed $seed >./logs_ECL/'predlen192_seed'$seed.log 
# done



# for seed in 2021 2022 2023 2024 2025; do
#   python -u run.py \
#     --is_training 1 \
#     --root_path ../Time-LLM/dataset/electricity/ \
#     --data_path electricity.csv \
#     --model_id ECL_96_336 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 336 \
#     --e_layers 3 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des 'Exp' \
#     --d_model 512 \
#     --d_ff 512 \
#     --batch_size 16 \
#     --learning_rate 0.0005 \
#     --itr 1 \
#     --fix_seed $seed >./logs_ECL/'predlen336_seed'$seed.log 
# done


# for seed in 2021 2022 2023 2024 2025; do
#   python -u run.py \
#     --is_training 1 \
#     --root_path ../Time-LLM/dataset/electricity/ \
#     --data_path electricity.csv \
#     --model_id ECL_96_720 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 720 \
#     --e_layers 3 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des 'Exp' \
#     --d_model 512 \
#     --d_ff 512 \
#     --batch_size 16 \
#     --learning_rate 0.0005 \
#     --itr 1 \
#     --fix_seed $seed >./logs_ECL/'predlen720_seed'$seed.log 
# done


# requires_grad=True, dropout=0., wv='sym8', m=1
seed=2022
python -u run.py \
  --is_training 1 \
  --root_path ../Time-LLM/dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 1024 \
  --batch_size 16 \
  --itr 1 \
  --learning_rate 0.0005 \
  --fix_seed $seed