export CUDA_VISIBLE_DEVICES=4

model_name=iWaveformer

for pred_len in 192 336 720; do
  for e_layers in 1 2; do
    for wv in 'db1'  'bior3.3' 'coif5'; do
      for m in 1 3; do
        for lr in 0.0005 0.001 0.003; do
          for batch_size in 128; do
            for use_norm in 1; do
              for fix_seed in 2025; do
                for d_model in 256; do
                  for d_ff in 256; do
                    python -u run_l1s.py \
                      --is_training 1 \
                      --lradj 'type1' \
                      --patience 3 \
                      --root_path ../../Waveformer/dataset/ETT-small/ \
                      --data_path ETTm2.csv \
                      --model_id ETTm2_96_96 \
                      --model $model_name \
                      --data ETTm2 \
                      --features M \
                      --seq_len 96 \
                      --requires_grad True \
                      --pred_len $pred_len \
                      --auto_pred_len $pred_len \
                      --e_layers $e_layers \
                      --enc_in 7 \
                      --dec_in 7 \
                      --c_out 7 \
                      --des 'Exp' \
                      --d_model $d_model \
                      --d_ff $d_model \
                      --learning_rate $lr \
                      --batch_size $batch_size \
                      --itr 3 \
                      --fix_seed $fix_seed \
                      --use_norm $use_norm \
                      --wv $wv \
                      --m $m >./logs_0927/ETTm2/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm'$d_model'_dff'$d_ff.log
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
