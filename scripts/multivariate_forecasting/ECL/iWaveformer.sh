export CUDA_VISIBLE_DEVICES=3

model_name=iWaveformer

for pred_len in 192 336 720; do
  for e_layers in 1 2; do
    for wv in 'db1' 'bior3.3' 'db8'; do
      for m in 1 3; do
        for lr in 0.0005 0.001 0.003; do
          for batch_size in 16 64; do
            for fix_seed in 2025; do
              python -u run_l1s.py \
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
                --m $m >./logs_0927/ECL/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm256_dff1024'.log
            done
          done
        done
      done
    done
  done
done


for pred_len in 96; do
  for e_layers in 2; do
    for wv in 'db1' 'bior3.3' 'db8'; do
      for m in 1 3; do
        for lr in 0.0005 0.001 0.003; do
          for batch_size in 16 64; do
            for fix_seed in 2025; do
              python -u run_l1s.py \
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
                --m $m >./logs_0927/ECL/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm256_dff1024'.log
            done
          done
        done
      done
    done
  done
done
# >./logs_iWaveformer/Traffic/0528/'pred_'$pred_len'_wv_'$wv'm_'$m'e_layers_'$e_layers.log 
# >./logs_iWaveformer/Traffic/0605/'pred'$pred_len'_wv'$wv'_m'$m'_e_layers'$e_layers'_lr'$lr'_bs'$batch_size.log 