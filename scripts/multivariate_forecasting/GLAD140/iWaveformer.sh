export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=iWaveformer

for pred_len in 3600; do
  for e_layers in 1; do
    for wv in 'db6'; do
      for m in 3; do
        for lr in 0.0005; do
          for batch_size in 32; do
            for fix_seed in 2024; do
              python -u run.py \
                --is_training 1 \
                --root_path ../Time-LLM/dataset/GLAD140/ \
                --data_path glad_sample1.csv \
                --model_id GLAD140_300_1800 \
                --model $model_name \
                --data glad \
                --features M \
                --seq_len 600 \
                --requires_grad False \
                --pred_len $pred_len \
                --auto_pred_len $pred_len \
                --fix_seed $fix_seed \
                --e_layers $e_layers \
                --enc_in 16 \
                --dec_in 16 \
                --c_out 16 \
                --des 'Exp' \
                --d_model 256 \
                --d_ff 1024 \
                --learning_rate $lr \
                --batch_size $batch_size \
                --itr 1 \
                --use_norm 1 \
                --wv $wv \
                --m $m  >./logs_glad/results.log
            done
          done
        done
      done
    done
  done
done
# >./logs_iWaveformer/Traffic/0528/'pred_'$pred_len'_wv_'$wv'm_'$m'e_layers_'$e_layers.log 
# >./logs_iWaveformer/Traffic/0605/'pred'$pred_len'_wv'$wv'_m'$m'_e_layers'$e_layers'_lr'$lr'_bs'$batch_size.log 