export CUDA_VISIBLE_DEVICES=7

model_name=iWaveformer

# do not need to normaliza the signal

# for pred_len in 96 192 336 720; do
#   for e_layers in 1 2; do
#     for wv in 'db8'; do
#       for m in 1; do
#         for lr in 0.006 0.003; do
#           for batch_size in 64; do
#             for fix_seed in 2025; do
#               python -u run_l1s.py \
#                 --is_training 1 \
#                 --root_path ../../Waveformer/dataset/Solar/ \
#                 --data_path solar_AL.txt \
#                 --model_id solar_96_96 \
#                 --model $model_name \
#                 --data Solar \
#                 --features M \
#                 --seq_len 96 \
#                 --requires_grad True \
#                 --pred_len $pred_len \
#                 --auto_pred_len $pred_len \
#                 --fix_seed $fix_seed \
#                 --e_layers $e_layers \
#                 --enc_in 137 \
#                 --dec_in 137 \
#                 --c_out 137 \
#                 --des 'Exp' \
#                 --d_model 96 \
#                 --d_ff 96 \
#                 --learning_rate $lr \
#                 --batch_size $batch_size \
#                 --itr 3 \
#                 --use_norm 0 \
#                 --wv $wv \
#                 --m $m >./logs_0927/Solar/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm96_dff96'.log
#             done
#           done
#         done
#       done
#     done
#   done
# done


for pred_len in 96 192 336 720; do
  for e_layers in 1 2; do
    for wv in 'db1' 'db8'; do
      for m in 1 3; do
        for lr in 0.0005 0.001 0.003; do
          for batch_size in 32; do
            for fix_seed in 2025; do
              python -u run_l1s.py \
                --is_training 1 \
                --root_path ../../Waveformer/dataset/Solar/ \
                --data_path solar_AL.txt \
                --model_id solar_96_96 \
                --model $model_name \
                --data Solar \
                --features M \
                --seq_len 96 \
                --requires_grad True \
                --pred_len $pred_len \
                --auto_pred_len $pred_len \
                --fix_seed $fix_seed \
                --e_layers $e_layers \
                --enc_in 137 \
                --dec_in 137 \
                --c_out 137 \
                --des 'Exp' \
                --d_model 512 \
                --d_ff 1024 \
                --learning_rate $lr \
                --batch_size $batch_size \
                --itr 3 \
                --use_norm 0 \
                --wv $wv \
                --m $m >./logs_0927/Solar/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm512_dff1024'.log
            done
          done
        done
      done
    done
  done
done



# >./logs_iWaveformer/Traffic/0528/'pred_'$pred_len'_wv_'$wv'm_'$m'e_layers_'$e_layers.log 
# >./logs_iWaveformer/Traffic/0605/'pred'$pred_len'_wv'$wv'_m'$m'_e_layers'$e_layers'_lr'$lr'_bs'$batch_size.log 