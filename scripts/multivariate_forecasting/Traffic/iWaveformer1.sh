export CUDA_VISIBLE_DEVICES=6

model_name=iWaveformer

for pred_len in 96 192 336 720; do
  for e_layers in 2; do
    for wv in 'db1' ; do
      for m in 1; do
        for lr in 0.0005; do
          for batch_size in 32; do
            for fix_seed in 2022; do
              python -u run.py \
                --is_training 1 \
                --fix_seed $fix_seed \
                --root_path ../Time-LLM/dataset/traffic/ \
                --data_path traffic.csv \
                --model_id traffic_96_96 \
                --model $model_name \
                --data custom \
                --features M \
                --seq_len 96 \
                --requires_grad False \
                --pred_len $pred_len \
                --auto_pred_len $pred_len \
                --e_layers $e_layers \
                --enc_in 862 \
                --dec_in 862 \
                --c_out 862 \
                --des 'Exp' \
                --d_model 1024 \
                --d_ff 2048 \
                --learning_rate $lr \
                --batch_size $batch_size \
                --itr 1 \
                --use_norm 1 \
                --train_epochs 20 \
                --lradj 'type2' \
                --wv $wv \
                --m $m 
            done
          done
        done
      done
    done
  done
done



# for pred_len in 96 192 336 720; do
#   for e_layers in 1; do
#     for wv in 'db1'; do
#       for m in 1; do
#         for lr in 0.0005; do
#           for batch_size in 32; do
#             for fix_seed in 2022 2023 2024 2025 2026; do
#               python -u run.py \
#                 --is_training 1 \
#                 --fix_seed $fix_seed \
#                 --root_path ../Time-LLM/dataset/traffic/ \
#                 --data_path traffic.csv \
#                 --model_id traffic_96_96 \
#                 --model $model_name \
#                 --data custom \
#                 --features M \
#                 --seq_len 96 \
#                 --requires_grad False \
#                 --pred_len $pred_len \
#                 --auto_pred_len $pred_len \
#                 --e_layers $e_layers \
#                 --enc_in 862 \
#                 --dec_in 862 \
#                 --c_out 862 \
#                 --des 'Exp' \
#                 --d_model 1024 \
#                 --d_ff 2048 \
#                 --learning_rate $lr \
#                 --batch_size $batch_size \
#                 --itr 1 \
#                 --use_norm 1 \
#                 --wv $wv \
#                 --m $m >./logs_0623/Traffic/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed.log
#             done
#           done
#         done
#       done
#     done
#   done
# done

# for pred_len in 96 192 336 720; do
#   for e_layers in 1; do
#     for wv in 'db1'; do
#       for m in 1; do
#         for lr in 0.0005; do
#           for batch_size in 32; do
#             python -u run.py \
#               --is_training 1 \
#               --root_path ../Time-LLM/dataset/traffic/ \
#               --data_path traffic.csv \
#               --model_id traffic_96_96 \
#               --model $model_name \
#               --data custom \
#               --features M \
#               --seq_len 96 \
#               --requires_grad False \
#               --pred_len $pred_len \
#               --e_layers $e_layers \
#               --enc_in 862 \
#               --dec_in 866 \
#               --c_out 862 \
#               --des 'Exp' \
#               --d_model 1024 \
#               --d_ff 2048 \
#               --learning_rate $lr \
#               --batch_size $batch_size \
#               --itr 1 \
#               --use_norm 1 \
#               --wv $wv \
#               --m $m
#           done
#         done
#       done
#     done
#   done
# done

# >./logs_iWaveformer/Traffic/0528/'pred_'$pred_len'_wv_'$wv'm_'$m'e_layers_'$e_layers.log 
# >./logs_iWaveformer/Traffic/0605/'pred'$pred_len'_wv'$wv'_m'$m'_e_layers'$e_layers'_lr'$lr'_bs'$batch_size.log 