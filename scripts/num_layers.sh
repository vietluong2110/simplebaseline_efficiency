export CUDA_VISIBLE_DEVICES=6
model_name=SimpleBaseline


# # ETTm1_96_192_32_32_1_db1_None_3_0.1_0.005_0.02_TST_256_2025_1
# # ETTm1_96_336_32_32_1_db1_None_1_0.1_0.005_0.02_TST_256_2025_1
# # ETTm1_96_720_32_32_1_db1_None_1_0.1_0.005_0.02_TST_256_2025_1
# for pred_len in 720 336 192; do
#   for e_layers in 1 2 3 4; do
#     for wv in 'db1'; do
#       for m in 1; do
#         for kernel_size in None; do
#           for lr in 0.02; do
#             for batch_size in 256; do
#               for use_norm in 1; do
#                 for fix_seed in 2025; do
#                   for d_model in 32; do
#                     for d_ff in 32; do
#                       for alpha in 0.1; do
#                         for l1_weight in 0.005; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/ETT-small/ \
#                             --data_path ETTm1.csv \
#                             --model_id ETTm1 \
#                             --model $model_name \
#                             --data ETTm1 \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 7 \
#                             --dec_in 7 \
#                             --c_out 7 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done


# # ETTm1_96_96_32_32_1_db1_None_3_0.1_0.005_0.02_TST_256_2025_1
# for pred_len in 96; do
#   for e_layers in 1 2 3 4; do
#     for wv in 'db1'; do
#       for m in 3; do
#         for kernel_size in None; do
#           for lr in 0.02; do
#             for batch_size in 256; do
#               for use_norm in 1; do
#                 for fix_seed in 2025; do
#                   for d_model in 32; do
#                     for d_ff in 32; do
#                       for alpha in 0.1; do
#                         for l1_weight in 0.005; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/ETT-small/ \
#                             --data_path ETTm1.csv \
#                             --model_id ETTm1 \
#                             --model $model_name \
#                             --data ETTm1 \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 7 \
#                             --dec_in 7 \
#                             --c_out 7 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done








# export CUDA_VISIBLE_DEVICES=6
# model_name=SimpleBaseline


# # Solar_96_96_128_256_1_db8_None_3_0.0_0.005_0.01_TST_256_2025_0
# for pred_len in 96; do
#   for e_layers in 1 2 3 4; do
#     for wv in 'db8'; do
#       for m in 3; do
#         for kernel_size in None; do
#           for lr in 0.01; do
#             for batch_size in 256; do
#               for use_norm in 0; do
#                 for fix_seed in 2025; do
#                   for d_model in 128; do
#                     for d_ff in 256; do
#                       for alpha in 0; do
#                         for l1_weight in 0.005; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/Solar/ \
#                             --data_path solar_AL.txt \
#                             --model_id Solar \
#                             --model $model_name \
#                             --data Solar \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 137 \
#                             --dec_in 137 \
#                             --c_out 137 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done


# # Solar_96_192_128_256_1_db8_None_1_0.0_0.005_0.003_TST_256_2025_0
# for pred_len in 192; do
#   for e_layers in 1 2 3 4; do
#     for wv in 'db8'; do
#       for m in 1; do
#         for kernel_size in None; do
#           for lr in 0.003; do
#             for batch_size in 256; do
#               for use_norm in 0; do
#                 for fix_seed in 2025; do
#                   for d_model in 128; do
#                     for d_ff in 256; do
#                       for alpha in 0; do
#                         for l1_weight in 0.005; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/Solar/ \
#                             --data_path solar_AL.txt \
#                             --model_id Solar \
#                             --model $model_name \
#                             --data Solar \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 137 \
#                             --dec_in 137 \
#                             --c_out 137 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done


# # Solar_96_336_128_256_1_db8_None_1_0.1_0.005_0.003_TST_256_2025_0
# for pred_len in 336; do
#   for e_layers in 1 2 3 4; do
#     for wv in 'db8'; do
#       for m in 1; do
#         for kernel_size in None; do
#           for lr in 0.003; do
#             for batch_size in 256; do
#               for use_norm in 0; do
#                 for fix_seed in 2025; do
#                   for d_model in 128; do
#                     for d_ff in 256; do
#                       for alpha in 0.1; do
#                         for l1_weight in 0.005; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/Solar/ \
#                             --data_path solar_AL.txt \
#                             --model_id Solar \
#                             --model $model_name \
#                             --data Solar \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 137 \
#                             --dec_in 137 \
#                             --c_out 137 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done

# # Solar_96_720_128_256_1_db8_None_1_0.0_0.005_0.009_TST_256_2025_0
# for pred_len in 720; do
#   for e_layers in 1 2 3 4; do
#     for wv in 'db8'; do
#       for m in 1; do
#         for kernel_size in None; do
#           for lr in 0.009; do
#             for batch_size in 256; do
#               for use_norm in 0; do
#                 for fix_seed in 2025; do
#                   for d_model in 128; do
#                     for d_ff in 256; do
#                       for alpha in 0; do
#                         for l1_weight in 0.005; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/Solar/ \
#                             --data_path solar_AL.txt \
#                             --model_id Solar \
#                             --model $model_name \
#                             --data Solar \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 137 \
#                             --dec_in 137 \
#                             --c_out 137 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done






export CUDA_VISIBLE_DEVICES=6
model_name=SimpleBaseline


# Weather_96_96_32_32_2_db4_None_1_0.3_5e-05_0.01_TST_256_2025_1
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



# # Weather_96_192_32_32_2_db4_None_1_0.3_0.0_0.009_TST_256_2025_1
# for pred_len in 192; do
#   for e_layers in 2 3 4 1; do
#     for wv in 'db4'; do
#       for m in 1; do
#         for kernel_size in None; do
#           for lr in 0.009; do
#             for batch_size in 256; do
#               for use_norm in 1; do
#                 for fix_seed in 2025; do
#                   for d_model in 32; do
#                     for d_ff in 32; do
#                       for alpha in 0.3; do
#                         for l1_weight in 0; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/weather/ \
#                             --data_path weather.csv \
#                             --model_id Weather \
#                             --model $model_name \
#                             --data custom \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 21 \
#                             --dec_in 21 \
#                             --c_out 21 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done


# # Weather_96_336_32_32_2_db4_None_3_1.0_5e-05_0.009_TST_256_2025_1
# for pred_len in 336; do
#   for e_layers in 2 3 4 1; do
#     for wv in 'db4'; do
#       for m in 3; do
#         for kernel_size in None; do
#           for lr in 0.009; do
#             for batch_size in 256; do
#               for use_norm in 1; do
#                 for fix_seed in 2025; do
#                   for d_model in 32; do
#                     for d_ff in 32; do
#                       for alpha in 1.0; do
#                         for l1_weight in 5e-5; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/weather/ \
#                             --data_path weather.csv \
#                             --model_id Weather \
#                             --model $model_name \
#                             --data custom \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 21 \
#                             --dec_in 21 \
#                             --c_out 21 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done

# # Weather_96_720_32_32_2_db4_None_1_0.9_0.005_0.02_TST_256_2025_1
# for pred_len in 720; do
#   for e_layers in 2 3 4 1; do
#     for wv in 'db4'; do
#       for m in 1; do
#         for kernel_size in None; do
#           for lr in 0.02; do
#             for batch_size in 256; do
#               for use_norm in 1; do
#                 for fix_seed in 2025; do
#                   for d_model in 32; do
#                     for d_ff in 32; do
#                       for alpha in 0.9; do
#                         for l1_weight in 0.005; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/weather/ \
#                             --data_path weather.csv \
#                             --model_id Weather \
#                             --model $model_name \
#                             --data custom \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 21 \
#                             --dec_in 21 \
#                             --c_out 21 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done


export CUDA_VISIBLE_DEVICES=6
model_name=SimpleBaseline

# # ETTh1_96_720_32_32_1_db1_None_1_0.9_0.0005_0.009_TST_256_2025_1
# for pred_len in 720; do
#   for e_layers in 1 2 3 4; do
#     for wv in 'db1'; do
#       for m in 1; do
#         for kernel_size in None; do
#           for lr in 0.009; do
#             for batch_size in 256; do
#               for use_norm in 1; do
#                 for fix_seed in 2025; do
#                   for d_model in 32; do
#                     for d_ff in 32; do
#                       for alpha in 0.9; do
#                         for l1_weight in 5e-4; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/ETT-small/ \
#                             --data_path ETTh1.csv \
#                             --model_id ETTh1 \
#                             --model $model_name \
#                             --data ETTh1 \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 7 \
#                             --dec_in 7 \
#                             --c_out 7 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done

# # ETTh1_96_192_32_32_1_db1_None_3_1.0_5e-05_0.02_TST_256_2025_1
# for pred_len in 192; do
#   for e_layers in 1 2 3 4; do
#     for wv in 'db1'; do
#       for m in 3; do
#         for kernel_size in None; do
#           for lr in 0.02; do
#             for batch_size in 256; do
#               for use_norm in 1; do
#                 for fix_seed in 2025; do
#                   for d_model in 32; do
#                     for d_ff in 32; do
#                       for alpha in 1.0; do
#                         for l1_weight in 5e-5; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/ETT-small/ \
#                             --data_path ETTh1.csv \
#                             --model_id ETTh1 \
#                             --model $model_name \
#                             --data ETTh1 \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 7 \
#                             --dec_in 7 \
#                             --c_out 7 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done

# # ETTh1_96_96_32_32_1_db1_None_3_0.3_0.0005_0.02_TST_256_2025_1
# for pred_len in 96; do
#   for e_layers in 1 2 3 4; do
#     for wv in 'db1'; do
#       for m in 3; do
#         for kernel_size in None; do
#           for lr in 0.02; do
#             for batch_size in 256; do
#               for use_norm in 1; do
#                 for fix_seed in 2025; do
#                   for d_model in 32; do
#                     for d_ff in 32; do
#                       for alpha in 0.3; do
#                         for l1_weight in 5e-4; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/ETT-small/ \
#                             --data_path ETTh1.csv \
#                             --model_id ETTh1 \
#                             --model $model_name \
#                             --data ETTh1 \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 7 \
#                             --dec_in 7 \
#                             --c_out 7 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done



# # ETTh1_96_336_64_64_2_db1_None_3_0.0_0.0_0.002_TST_256_2025_1
# for pred_len in 336; do
#   for e_layers in 2 3 4 1; do
#     for wv in 'db1'; do
#       for m in 3; do
#         for kernel_size in None; do
#           for lr in 0.002; do
#             for batch_size in 256; do
#               for use_norm in 1; do
#                 for fix_seed in 2025; do
#                   for d_model in 64; do
#                     for d_ff in 64; do
#                       for alpha in 0; do
#                         for l1_weight in 0; do
#                           python -u run_ca.py \
#                             --is_training 1 \
#                             --lradj 'TST' \
#                             --patience 3 \
#                             --alpha $alpha \
#                             --l1_weight $l1_weight \
#                             --root_path ./dataset/ETT-small/ \
#                             --data_path ETTh1.csv \
#                             --model_id ETTh1 \
#                             --model $model_name \
#                             --data ETTh1 \
#                             --features M \
#                             --seq_len 96 \
#                             --pred_len $pred_len \
#                             --e_layers $e_layers \
#                             --d_model $d_model \
#                             --d_ff $d_ff \
#                             --learning_rate $lr \
#                             --batch_size $batch_size \
#                             --fix_seed $fix_seed \
#                             --use_norm $use_norm \
#                             --wv $wv \
#                             --m $m \
#                             --enc_in 7 \
#                             --dec_in 7 \
#                             --c_out 7 \
#                             --des 'Exp' \
#                             --itr 3 
#                         done
#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done