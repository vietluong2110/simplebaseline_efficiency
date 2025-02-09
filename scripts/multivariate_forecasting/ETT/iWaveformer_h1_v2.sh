export CUDA_VISIBLE_DEVICES=7
model_name=iWaveformer2


# # 96: wv='bior3.3, m=1; wv='db1', m=3
# for pred_len in 720; do
#   for e_layers in 2; do
#     for wv in 'coif5'; do
#       for m in 1; do
#         for lr in 0.001; do
#           for batch_size in 256; do
#             for use_norm in 1; do
#               for fix_seed in 2025; do
#                 for d_model in 256; do
#                   for d_ff in 256; do
#                     python -u run_ca.py \
#                       --is_training 1 \
#                       --patience 3 \
#                       --root_path ../../Waveformer/dataset/ETT-small/ \
#                       --data_path ETTh1.csv \
#                       --model_id AAAAA_ETTh1 \
#                       --model $model_name \
#                       --data ETTh1 \
#                       --features M \
#                       --seq_len 96 \
#                       --requires_grad True \
#                       --pred_len $pred_len \
#                       --auto_pred_len $pred_len \
#                       --e_layers $e_layers \
#                       --enc_in 7 \
#                       --dec_in 7 \
#                       --c_out 7 \
#                       --des 'Exp' \
#                       --d_model $d_model \
#                       --d_ff $d_ff \
#                       --learning_rate $lr \
#                       --batch_size $batch_size \
#                       --itr 3 \
#                       --fix_seed $fix_seed \
#                       --use_norm $use_norm \
#                       --wv $wv \
#                       --m $m 
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done





for pred_len in 96; do
  for e_layers in 1 1; do
    for wv in  'db1' ; do
      for m in 1; do
        for lr in 0.0005; do
          for batch_size in 16; do
            for use_norm in 1; do
              for fix_seed in 2025; do
                for d_model in 512; do
                  for d_ff in 512; do
                    python -u run_ca.py \
                      --is_training 1 \
                      --lradj 'type1' \
                      --patience 3 \
                      --root_path ../../Waveformer/dataset/ETT-small/ \
                      --data_path ETTh1.csv \
                      --model_id AAAAA_ETTh1 \
                      --model $model_name \
                      --data ETTh1 \
                      --features M \
                      --seq_len 192 \
                      --requires_grad True \
                      --pred_len $pred_len \
                      --auto_pred_len $pred_len \
                      --e_layers $e_layers \
                      --enc_in 7 \
                      --dec_in 7 \
                      --c_out 7 \
                      --des 'Exp' \
                      --d_model $d_model \
                      --d_ff $d_ff \
                      --learning_rate $lr \
                      --batch_size $batch_size \
                      --itr 3 \
                      --fix_seed $fix_seed \
                      --use_norm $use_norm \
                      --wv $wv \
                      --m $m 
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

# >./logs_0930/ETTh1/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm'$d_model'_dff'$d_ff.log