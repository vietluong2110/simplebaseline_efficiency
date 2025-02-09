export CUDA_VISIBLE_DEVICES=1

model_name=iWaveformer


for pred_len in 720; do
  for e_layers in 1; do
    for wv in 'bior1.3'; do
      for m in 3; do
        for lr in 0.0005; do
          for batch_size in 128; do
            for use_norm in 1; do
              for fix_seed in 2025 2023 2026 2024 2022; do
                for d_model in 512; do
                  python -u run.py \
                    --is_training 1 \
                    --train_epochs 10 \
                    --root_path ../Time-LLM/dataset/ETT-small/ \
                    --data_path ETTm1.csv \
                    --model_id ETTm1_96_96 \
                    --model $model_name \
                    --data ETTm1 \
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
                    --itr 1 \
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
