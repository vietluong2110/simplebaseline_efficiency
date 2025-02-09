export CUDA_VISIBLE_DEVICES=7

model_name=iWaveformer2


for pred_len in 96 192 336 720; do
  for e_layers in 1; do
    for wv in 'db1' ; do
      for m in 1; do
        for lr in 0.0005; do
          for batch_size in 16; do
            for fix_seed in 2025; do
              for d_model in 48; do
                for d_ff in 48; do
                  python -u run_ca.py \
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
