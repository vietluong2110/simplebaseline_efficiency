export CUDA_VISIBLE_DEVICES=6

model_name=iWaveformer


for pred_len in 96 192 336 720; do
  for e_layers in 1 4; do
    for wv in 'db1' ; do
      for m in 1 3; do
        for lr in 0.006 0.001 0.003; do
          for batch_size in 32; do
            for fix_seed in 2025; do
              for d_model in 512; do
                for d_ff in 1024; do
                  python -u run_l1s.py \
                    --is_training 1 \
                    --fix_seed $fix_seed \
                    --root_path ../../Waveformer/dataset/traffic/ \
                    --data_path traffic.csv \
                    --model_id traffic_96_96 \
                    --model $model_name \
                    --data custom \
                    --features M \
                    --seq_len 96 \
                    --requires_grad True \
                    --pred_len $pred_len \
                    --auto_pred_len $pred_len \
                    --e_layers $e_layers \
                    --enc_in 862 \
                    --dec_in 862 \
                    --c_out 862 \
                    --des 'Exp' \
                    --d_model $d_model \
                    --d_ff $d_ff \
                    --learning_rate $lr \
                    --batch_size $batch_size \
                    --itr 3 \
                    --use_norm 1 \
                    --train_epochs 20 \
                    --wv $wv \
                    --m $m >./logs_0927/Traffic/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm'$d_model'_dff'$d_ff.log
                done
              done
            done
          done
        done
      done
    done
  done
done
