export CUDA_VISIBLE_DEVICES=1

model_name=iWaveformer2

for pred_len in 336; do
  for e_layers in 2; do
    for wv in 'bior3.3'; do
      for m in 1; do
        for lr in 0.0005; do
          for batch_size in 64; do
            for fix_seed in 2021; do
              for d_model in 128; do
                python -u run_ca.py \
                  --train_epochs 20 \
                  --patience 5 \
                  --is_training 1 \
                  --fix_seed $fix_seed \
                  --root_path ../../Waveformer/dataset/exchange_rate/ \
                  --data_path exchange_rate.csv \
                  --model_id Exchange_96_720 \
                  --model $model_name \
                  --data custom \
                  --features M \
                  --seq_len 96 \
                  --requires_grad True \
                  --pred_len $pred_len \
                  --auto_pred_len $pred_len \
                  --e_layers $e_layers \
                  --enc_in 8 \
                  --dec_in 8 \
                  --c_out 8 \
                  --des 'Exp' \
                  --d_model $d_model \
                  --d_ff $d_model \
                  --learning_rate $lr \
                  --batch_size $batch_size \
                  --itr 3 \
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
