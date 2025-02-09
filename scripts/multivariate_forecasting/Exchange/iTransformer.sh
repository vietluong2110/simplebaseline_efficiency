export CUDA_VISIBLE_DEVICES=7

model_name=iTransformer

# for seed in 2021 2022 2023 2024 2025; do
#   python -u run.py \
#     --is_training 1 \
#     --root_path ../Time-LLM/dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --e_layers 1 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'Exp' \
#     --d_model 128 \
#     --d_ff 128 \
#     --itr 1 \
#     --learning_rate 0.0005 \
#     --fix_seed $seed >./logs_Exchange/'predlen96_seed'$seed.log 
# done


# for seed in 2021 2022 2023 2024 2025; do
#   python -u run.py \
#     --is_training 1 \
#     --root_path ../Time-LLM/dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_192 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 192 \
#     --e_layers 2 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'Exp' \
#     --d_model 128 \
#     --d_ff 128 \
#     --itr 1 \
#     --learning_rate 0.0005 \
#     --fix_seed $seed >./logs_Exchange/'predlen192_seed'$seed.log 
# done


# for seed in 2021 2022 2023 2024 2025; do
#   python -u run.py \
#     --is_training 1 \
#     --root_path ../Time-LLM/dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_336 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 336 \
#     --e_layers 1 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'Exp' \
#     --itr 1 \
#     --d_model 128 \
#     --d_ff 128 \
#     --train_epochs 2 \
#     --learning_rate 0.0005 \
#     --fix_seed $seed >./logs_Exchange/'predlen336_seed'$seed.log 
# done


# requires_grad=True, dropout=0.5, wv='db1', m=1
for seed in 2021 2022 2023 2024 2025; do
  python -u run.py \
    --is_training 1 \
    --root_path ../Time-LLM/dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_96_720 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 1 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --train_epochs 1 \
    --learning_rate 0.001 \
    --fix_seed $seed >./logs_Exchange/'predlen720_seed'$seed.log 
done
