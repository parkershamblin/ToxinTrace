#!/usr/bin/env bash
# R1: base Pythia-160M from scratch (15 h on an RTX 4090). Use MICRO_BATCH=4 if other programs use the GPU.
source "$(dirname "$0")/env.sh"
run torchrun --nproc-per-node=1 --master_port=29500 train.py --model_name $MODEL \
  --learning_rate 6e-4 --weight_decay 4e-4 --warmup_ratio 0.01 --max_grad_norm 1 \
  --train_batch_size ${MICRO_BATCH:-8} --eval_batch_size 8 --num_train_epochs 4 --logging_steps 25 --seed 1004 \
  --train_indices_path "$IDX/train_indices.npy" --valid_indices_path "$IDX/valid_indices.npy" \
  --save_dir "$TT_ROOT/models" --save_id baseline
