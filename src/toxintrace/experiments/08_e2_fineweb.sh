#!/usr/bin/env bash
# E2: 400M GPT-NeoX tokens from FineWeb CC-MAIN-2025-26 (2.2 GB download, ~20 min), then its Detoxify
# toxicity rate against OpenWebText (~10 min). Influence scores, mask and fine-tunes: not run yet.
source "$(dirname "$0")/env.sh"
if [ ! -d "$TT_ROOT/data/fineweb_subset" ]; then
  run python toxintrace/make_fineweb_subset.py --output_dir "$TT_ROOT/data/fineweb_subset" \
    --index_dir "$TT_ROOT/data/fineweb_subset_indices" --cache_dir "$TT_ROOT/data/fineweb_raw"
fi
TT_ROOT="$TT_ROOT" run python toxintrace/compare_toxicity.py --out_dir "$OUT/e2_toxicity"
