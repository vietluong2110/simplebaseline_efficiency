export CUDA_VISIBLE_DEVICES=5

model_name=iWaveformer

for pred_len in 96; do
  for e_layers in 2; do
    for wv in 'db12'; do
      for m in 2; do
        for lr in 0.0005; do
          for batch_size in 16; do
            for fix_seed in 2023; do
              for d_model in 256; do
                python -u run_l1s.py \
                  --is_training 1 \
                  --root_path ../Time-LLM/dataset/PEMS/ \
                  --data_path PEMS08.npz \
                  --model_id PEMS08_96_12 \
                  --model $model_name \
                  --data PEMS \
                  --features M \
                  --seq_len 96 \
                  --requires_grad False \
                  --pred_len $pred_len \
                  --auto_pred_len $pred_len \
                  --fix_seed $fix_seed \
                  --e_layers $e_layers \
                  --enc_in 170 \
                  --dec_in 170 \
                  --c_out 170 \
                  --des 'Exp' \
                  --d_model $d_model \
                  --d_ff 1024 \
                  --learning_rate $lr \
                  --batch_size $batch_size \
                  --itr 1 \
                  --use_norm 0 \
                  --wv $wv \
                  --m $m >./logs_0918_mse/PEMS08/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm'$d_model.log
              done
            done
          done
        done
      done
    done
  done
done

# for pred_len in 48 96; do
#   for e_layers in 2; do
#     for wv in 'db12'; do
#       for m in 2; do
#         for lr in 0.0005; do
#           for batch_size in 16; do
#             for fix_seed in 2022 2023 2024 2025 2026; do
#               python -u run.py \
#                 --is_training 1 \
#                 --root_path ../Time-LLM/dataset/PEMS/ \
#                 --data_path PEMS08.npz \
#                 --model_id PEMS08_96_12 \
#                 --model $model_name \
#                 --data PEMS \
#                 --features M \
#                 --seq_len 96 \
#                 --requires_grad False \
#                 --pred_len $pred_len \
#                 --auto_pred_len $pred_len \
#                 --fix_seed $fix_seed \
#                 --e_layers $e_layers \
#                 --enc_in 170 \
#                 --dec_in 170 \
#                 --c_out 170 \
#                 --des 'Exp' \
#                 --d_model 256 \
#                 --d_ff 1024 \
#                 --learning_rate $lr \
#                 --batch_size $batch_size \
#                 --itr 1 \
#                 --use_norm 0 \
#                 --wv $wv \
#                 --m $m >./logs_0623/PEMS08/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed.log
#             done
#           done
#         done
#       done
#     done
#   done
# done



# export CUDA_VISIBLE_DEVICES=0

# model_name=iWaveformer

# for pred_len in 12 24; do
#   for e_layers in 1; do
#     for wv in 'db1'; do
#       for m in 1; do
#         for lr in 0.0005; do
#           for batch_size in 16; do
#             for fix_seed in 2022 2023 2024 2025 2026; do
#               python -u run.py \
#                 --is_training 1 \
#                 --root_path ../Time-LLM/dataset/PEMS/ \
#                 --data_path PEMS08.npz \
#                 --model_id PEMS08_96_12 \
#                 --model $model_name \
#                 --data PEMS \
#                 --features M \
#                 --seq_len 96 \
#                 --requires_grad False \
#                 --pred_len $pred_len \
#                 --auto_pred_len $pred_len \
#                 --fix_seed $fix_seed \
#                 --e_layers $e_layers \
#                 --enc_in 170 \
#                 --dec_in 170 \
#                 --c_out 170 \
#                 --des 'Exp' \
#                 --d_model 256 \
#                 --d_ff 1024 \
#                 --learning_rate $lr \
#                 --batch_size $batch_size \
#                 --itr 1 \
#                 --use_norm 1 \
#                 --wv $wv \
#                 --m $m >./logs_0623/PEMS08/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed.log
#             done
#           done
#         done
#       done
#     done
#   done
# done