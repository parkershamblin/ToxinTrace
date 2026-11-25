#!/usr/bin/env bash
# R1: EK-FAC factors (10 h), differential token scores (4.4 h) and the 2% toxic-token mask (5 min, ~15 GB RAM)
source "$(dirname "$0")/env.sh"
BASE=$(base_run)
run python fit_factors.py --model_name $MODEL --checkpoint_dir "$BASE/final" --output_dir "$BASE" \
  --train_indices_path "$IDX/train_indices.npy" --factor_strategy ekfac --use_half_precision --train_batch_size 4
run python compute_scores.py --model_name $MODEL --checkpoint_dir "$BASE/final" --factors_path "$BASE" \
  --train_indices_path "$IDX/train_indices.npy" --query_dataset RTP \
  --toxic_query_indices_path ../data/RTP/query_indices/toxic_indices.npy \
  --nontoxic_query_indices_path ../data/RTP/query_indices/nontoxic_indices.npy \
  --save_dir RTP --factor_strategy ekfac --factors_name ekfac_half --use_half_precision \
  --query_batch_size 4 --train_batch_size 4
run python build_toxic_token_mask.py --model_name $MODEL --train_indices_path "$IDX/train_indices.npy" \
  --scores_path "$BASE/if_results/scores_RTP/ekfac_half/pairwise_scores.safetensors" \
  --window 1 --toxicity_threshold 0.99 --max_tokens 20000000 --inspection_idx 0
