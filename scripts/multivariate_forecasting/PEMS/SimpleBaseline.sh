export CUDA_VISIBLE_DEVICES=7
model_name=SimpleBaseline

# because the rror in the model/simplebaseline, all the e_layers are indeed ==1
# PEMS03_96_12_256_512_2_bior3.1_None_3_0.1_0.005_0.002_TST_16_2025_0
pred_len=12
for pred_len in 24 48 96; do
    python -u run_ca.py \
    --is_training 1 \
    --lradj 'TST' \
    --patience 10 \
    --train_epochs 20 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_model 256 \
    --d_ff 512 \
    --learning_rate 0.002 \
    --batch_size 16 \
    --fix_seed 2025 \
    --use_norm 0 \
    --wv "bior3.1" \
    --m 3 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --itr 3 \
    --alpha 0.1 \
    --l1_weight 0.005
done

for pred_len in 24 48 96; do
    python -u run_ca.py \
    --is_training 1 \
    --lradj 'TST' \
    --patience 10 \
    --train_epochs 20 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_model 256 \
    --d_ff 1024 \
    --learning_rate 0.002 \
    --batch_size 16 \
    --fix_seed 2025 \
    --use_norm 0 \
    --wv "bior3.1" \
    --m 3 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --itr 3 \
    --alpha 0.1 \
    --l1_weight 0.005
done



# PEMS04_96_12_256_1024_2_bior3.1_None_3_0.1_5e-05_0.002_TST_16_2025_0
for pred_len in 24 48 96; do
    python -u run_ca.py \
    --is_training 1 \
    --lradj 'TST' \
    --patience 10 \
    --train_epochs 20 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_model 256 \
    --d_ff 1024 \
    --learning_rate 0.002 \
    --batch_size 16 \
    --fix_seed 2025 \
    --use_norm 0 \
    --wv "bior3.1" \
    --m 3 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --itr 3 \
    --alpha 0.1 \
    --l1_weight 0.00005
done


# ,PEMS08_96_12_256_512_2_db12_None_3_0.0_0.0_0.001_TST_16_2025_0
for pred_len in 24 48 96; do
    python -u run_ca.py \
    --is_training 1 \
    --lradj 'TST' \
    --patience 10 \
    --train_epochs 20 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_model 256 \
    --d_ff 1024 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --fix_seed 2025 \
    --use_norm 0 \
    --wv "db12" \
    --m 3 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --des 'Exp' \
    --itr 3 \
    --alpha 0 \
    --l1_weight 0
done


for pred_len in 12 24 48 96; do
    python -u run_ca.py \
    --is_training 1 \
    --lradj 'TST' \
    --patience 10 \
    --train_epochs 20 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_model 256 \
    --d_ff 512 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --fix_seed 2025 \
    --use_norm 0 \
    --wv "db12" \
    --m 3 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --des 'Exp' \
    --itr 3 \
    --alpha 0 \
    --l1_weight 0
done


# PEMS07_96_12_256_512_1_db1_None_3_0.1_5e-05_0.002_TST_16_2025_0

for pred_len in 24 48 96; do
    python -u run_ca.py \
    --is_training 1 \
    --lradj 'TST' \
    --patience 10 \
    --train_epochs 20 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_model 256 \
    --d_ff 512 \
    --learning_rate 0.002 \
    --batch_size 16 \
    --fix_seed 2025 \
    --use_norm 0 \
    --wv "db1" \
    --m 3 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --itr 3 \
    --alpha 0.1 \
    --l1_weight 5e-5
done

