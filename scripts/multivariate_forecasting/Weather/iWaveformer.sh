export CUDA_VISIBLE_DEVICES=5

model_name=iWaveformer

# for pred_len in 96 336 192 720; do
#   for e_layers in 2; do
#     for wv in 'db4'; do
#       for m in 3 1; do
#         for lr in 0.001 0.003 0.006; do
#           for batch_size in 128; do
#             for fix_seed in 2025; do
#               for d_model in 48; do
#                 for d_ff in 48; do
#                   python -u run_l1s.py \
#                     --is_training 1 \
#                     --lradj 'type1' \
#                     --fix_seed $fix_seed \
#                     --root_path ../../Waveformer/dataset/weather/ \
#                     --data_path weather.csv \
#                     --model_id weather_96_96 \
#                     --model $model_name \
#                     --data custom \
#                     --features M \
#                     --seq_len 96 \
#                     --requires_grad True \
#                     --pred_len $pred_len \
#                     --auto_pred_len $pred_len \
#                     --e_layers $e_layers \
#                     --enc_in 21 \
#                     --dec_in 21 \
#                     --c_out 21 \
#                     --des 'Exp' \
#                     --d_model $d_model \
#                     --d_ff $d_ff \
#                     --learning_rate $lr \
#                     --batch_size $batch_size \
#                     --itr 3 \
#                     --use_norm 1 \
#                     --wv $wv \
#                     --m $m   >./logs_0927/Weather/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm'$d_model'_dff'$d_ff.log
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done


for pred_len in 96 336 192 720; do
  for e_layers in 2; do
    for wv in 'db4'; do
      for m in 3 1; do
        for lr in 0.001 0.003 0.006; do
          for batch_size in 128; do
            for fix_seed in 2025; do
              for d_model in 256; do
                for d_ff in 1024; do
                  python -u run_l1s.py \
                    --is_training 1 \
                    --lradj 'type1' \
                    --fix_seed $fix_seed \
                    --root_path ../../Waveformer/dataset/weather/ \
                    --data_path weather.csv \
                    --model_id weather_96_96 \
                    --model $model_name \
                    --data custom \
                    --features M \
                    --seq_len 96 \
                    --requires_grad True \
                    --pred_len $pred_len \
                    --auto_pred_len $pred_len \
                    --e_layers $e_layers \
                    --enc_in 21 \
                    --dec_in 21 \
                    --c_out 21 \
                    --des 'Exp' \
                    --d_model $d_model \
                    --d_ff $d_ff \
                    --learning_rate $lr \
                    --batch_size $batch_size \
                    --itr 3 \
                    --use_norm 1 \
                    --wv $wv \
                    --m $m  >./logs_0927/Weather/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm'$d_model'_dff'$d_ff.log
                done
              done
            done
          done
        done
      done
    done
  done
done
