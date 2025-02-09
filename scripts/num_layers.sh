export CUDA_VISIBLE_DEVICES=6
model_name=SimpleBaseline


for pred_len in 96; do
  for e_layers in 1; do
    for wv in 'db4'; do
      for m in 1; do
        for kernel_size in None; do
          for lr in 0.01; do
            for batch_size in 256; do
              for use_norm in 1; do
                for fix_seed in 2025; do
                  for d_model in 32; do
                    for d_ff in 32; do
                      for alpha in 0.3; do
                        for l1_weight in 5e-5; do
                          python -u run_ca.py \
                            --is_training 1 \
                            --lradj 'TST' \
                            --patience 3 \
                            --alpha $alpha \
                            --l1_weight $l1_weight \
                            --root_path ./dataset/weather/ \
                            --data_path weather.csv \
                            --model_id Weather \
                            --model $model_name \
                            --data custom \
                            --features M \
                            --seq_len 96 \
                            --pred_len $pred_len \
                            --e_layers $e_layers \
                            --d_model $d_model \
                            --d_ff $d_ff \
                            --learning_rate $lr \
                            --batch_size $batch_size \
                            --fix_seed $fix_seed \
                            --use_norm $use_norm \
                            --wv $wv \
                            --m $m \
                            --enc_in 21 \
                            --dec_in 21 \
                            --c_out 21 \
                            --des 'Exp' \
                            --itr 3 
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
    done
  done
done







