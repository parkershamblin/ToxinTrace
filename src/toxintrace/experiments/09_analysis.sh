#!/usr/bin/env bash
# Bootstrap intervals, severity thresholds, BOLD first-word analysis and mask positions (CPU, ~5 min)
source "$(dirname "$0")/env.sh"
run python toxintrace/analysis.py --results_dir "$OUT" --strip_dir "$TT_ROOT/results_boldstrip" \
  --mask_path $MASK --out_dir "$OUT/analysis"
