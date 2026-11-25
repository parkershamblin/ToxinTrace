#!/usr/bin/env bash
# E2, part 1: 400M GPT-NeoX tokens from FineWeb CC-MAIN-2025-26 (2.2 GB download, ~20 min). Later parts not run yet.
source "$(dirname "$0")/env.sh"
run python toxintrace/make_fineweb_subset.py --output_dir "$TT_ROOT/data/fineweb_subset" \
  --index_dir "$TT_ROOT/data/fineweb_subset_indices" --cache_dir "$TT_ROOT/data/fineweb_raw"
