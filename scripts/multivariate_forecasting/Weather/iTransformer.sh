export CUDA_VISIBLE_DEVICES=7

model_name=iTransformer

# 96  requires_grad=True, dropout=0.1, wv='sym5', m=1
# 336  requires_grad=True, dropout=0.1, wv='sym5', m=1
# 192 requires_grad=True, dropout=0.2, wv='bior3.3', m=1
# num_tokens-25; m=1; grad=True; waveletattention
for seed in 2021 2022 2023 2024 2025; do
  python -u run.py \
    --is_training 1 \
    --root_path ../Time-LLM/dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --d_model 96\
    --d_ff 96\
    --itr 1 \
    --learning_rate 0.0005 \
    --fix_seed $seed 
done



# for seed in 2024 2025; do
#   python -u run.py \
#     --is_training 1 \
#     --root_path ../Time-LLM/dataset/weather/ \
#     --data_path weather.csv \
#     --model_id weather_96_192 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 192 \
#     --e_layers 1 \
#     --enc_in 21 \
#     --dec_in 21 \
#     --c_out 21 \
#     --des 'Exp' \
#     --d_model 512\
#     --d_ff 512\
#     --itr 1 \
#     --learning_rate 0.0005 \
#     --fix_seed $seed >./logs_Weather/'predlen192_seed'$seed.log 
# done


# for seed in 2021 2022 2023 2024 2025; do
#   python -u run.py \
#     --is_training 1 \
#     --root_path ../Time-LLM/dataset/weather/ \
#     --data_path weather.csv \
#     --model_id weather_96_336 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 336 \
#     --e_layers 1 \
#     --enc_in 21 \
#     --dec_in 21 \
#     --c_out 21 \
#     --des 'Exp' \
#     --d_model 512\
#     --d_ff 512\
#     --itr 1 \
#     --learning_rate 0.0005 \
#     --fix_seed $seed >./logs_Weather/'predlen336_seed'$seed.log 
# done


# for seed in 2021 2022 2023 2024 2025; do
#   python -u run.py \
#     --is_training 1 \
#     --root_path ../Time-LLM/dataset/weather/ \
#     --data_path weather.csv \
#     --model_id weather_96_720 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 720 \
#     --e_layers 1 \
#     --enc_in 21 \
#     --dec_in 21 \
#     --c_out 21 \
#     --des 'Exp' \
#     --d_model 512\
#     --d_ff 512\
#     --itr 1 \
#     --learning_rate 0.0005 \
#     --fix_seed $seed >./logs_Weather/'predlen720_seed'$seed.log 
# done
