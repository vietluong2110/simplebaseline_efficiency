export CUDA_VISIBLE_DEVICES=3

model_name=iWaveformer


for pred_len in 720 192; do
  for e_layers in 2; do
    for wv in 'db4'; do
      for m in 1; do
        for lr in 0.001; do
          for batch_size in 128; do
            for fix_seed in 2022 2023 2024 2025 2026; do
              for d_model in 256; do
                for d_ff in 512 1024; do
                  python -u run_l1s.py \
                    --is_training 1 \
                    --lradj 'type1' \
                    --fix_seed $fix_seed \
                    --root_path ../Time-LLM/dataset/weather/ \
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
                    --itr 1 \
                    --use_norm 1 \
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


# >./logs_0918_mse/Weather/'pl'$pred_len'_el'$e_layers'_wv'$wv'_m'$m'_lr'$lr'_bs'$batch_size'_seed'$fix_seed'_dm'$d_model'_df'$d_ff.log
