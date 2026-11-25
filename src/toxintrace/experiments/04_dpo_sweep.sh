#!/usr/bin/env bash
# DPO on both models with a checkpoint every 100 steps, then score steps 0-1,000 (perplexity, LAMBADA, RTP subset).
# Needs the Lee et al. (2024) preference pairs, downloaded by hand: see the README.
source "$(dirname "$0")/env.sh"
for N in baseline ifguide; do
  START=$(latest_with "$TT_ROOT/models/training/$MODEL/$N")/final
  run python toxintrace/dpo.py --model_name $MODEL --checkpoint_dir "$START" \
    --data_dir "$TT_ROOT/data/dpo_toxic/toxicity_pairwise" --save_dir "$TT_ROOT/models/dpo_sweep" --save_id $N \
    --max_steps 3000 --sweep_save_steps 100 --eval_every_examples 400
  RUN=$(latest_with "$TT_ROOT/models/dpo_sweep/dpo/$MODEL/$N" checkpoint-100)
  run bash toxintrace/score_sweep.sh $N "$START" "$RUN" 1000
done
