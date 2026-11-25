#!/usr/bin/env bash
# R2: does IF-Guide's BOLD toxicity depend on the trailing space in every BOLD prompt? (~12 min per model)
source "$(dirname "$0")/env.sh"
export IFGUIDE_STRIP_BOLD=1
for N in baseline ifguide; do
  run python run_toxicity_eval.py --model_name $MODEL --dataset BOLD --batch_size 128 --decoding_defense none \
    --checkpoint_dir "$(latest_with "$TT_ROOT/models/training/$MODEL/$N")/final" \
    --output_dir "$TT_ROOT/results_boldstrip" --save_dir $N --save_outputs
done
