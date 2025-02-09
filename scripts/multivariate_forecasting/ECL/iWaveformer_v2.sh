export CUDA_VISIBLE_DEVICES=4

model_name=iWaveformer2

for pred_len in 336 720; do
  for e_layers in 1; do
    for wv in 'db1'; do
      for m in 1; do
        for lr in 0.001; do
          for batch_size in 16; do
            for fix_seed in 2025; do
              python -u run_ca.py \
                --is_training 1 \
                --root_path ../../Waveformer/dataset/electricity/ \
                --data_path electricity.csv \
                --model_id ECL_96_96 \
                --model $model_name \
                --data custom \
                --features M \
                --seq_len 96 \
                --requires_grad True \
                --pred_len $pred_len \
                --auto_pred_len $pred_len \
                --fix_seed $fix_seed \
                --e_layers $e_layers \
                --enc_in 321 \
                --dec_in 321 \
                --c_out 321 \
                --des 'Exp' \
                --d_model 256 \
                --d_ff 1024 \
                --learning_rate $lr \
                --batch_size $batch_size \
                --itr 3 \
                --use_norm 1 \
                --wv $wv \
                --m $m >./logs_0930/ECL/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm256_dff1024'.log
            done
          done
        done
      done
    done
  done
done



for pred_len in 336 720 96 192; do
  for e_layers in 2; do
    for wv in 'db1'; do
      for m in 3; do
        for lr in 0.003; do
          for batch_size in 16; do
            for fix_seed in 2025; do
              python -u run_ca.py \
                --is_training 1 \
                --root_path ../../Waveformer/dataset/electricity/ \
                --data_path electricity.csv \
                --model_id ECL_96_96 \
                --model $model_name \
                --data custom \
                --features M \
                --seq_len 96 \
                --requires_grad True \
                --pred_len $pred_len \
                --auto_pred_len $pred_len \
                --fix_seed $fix_seed \
                --e_layers $e_layers \
                --enc_in 321 \
                --dec_in 321 \
                --c_out 321 \
                --des 'Exp' \
                --d_model 256 \
                --d_ff 1024 \
                --learning_rate $lr \
                --batch_size $batch_size \
                --itr 3 \
                --use_norm 1 \
                --wv $wv \
                --m $m >./logs_0930/ECL/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm256_dff1024'.log
            done
          done
        done
      done
    done
  done
done
