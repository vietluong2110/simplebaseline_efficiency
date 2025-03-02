export CUDA_VISIBLE_DEVICES=4
export JAX_TRACEBACK_FILTERING=off
model_name=JAX_SimpleBaseline

for pred_len in 96; do
  for e_layers in 1; do
    for wv in 'db1'; do
      for m in 1; do
        for kernel_size in None; do
          for lr in 0.01; do
            for batch_size in 256; do
              for use_norm in 1; do
                for fix_seed in 2025; do
                  for d_model in 512; do
                    for d_ff in 512; do
                      for alpha in 1; do
                        for l1_weight in 5e-5; do
                          python -u jax_run_ca.py \
                            --is_training 1 \
                            --lradj 'TST' \
                            --patience 3 \
                            --root_path ./dataset/ETT-small/ \
                            --data_path ETTh1.csv \
                            --model_id ETTh1 \
                            --model $model_name \
                            --data ETTh1 \
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
                            --enc_in 7 \
                            --dec_in 7 \
                            --c_out 7 \
                            --des 'Exp' \
                            --itr 1 \
                            --exp_name jax \
                            # --benchmark True
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