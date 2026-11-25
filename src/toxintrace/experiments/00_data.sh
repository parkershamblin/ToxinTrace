#!/usr/bin/env bash
# Compact copy of the paper's 1B-token OpenWebText subset (~33 GB download, ~150 GB temporary disk, ~1 h)
source "$(dirname "$0")/env.sh"
run python toxintrace/make_owt_subset.py --output_dir "$TT_ROOT/data/owt_subset" --index_dir "$IDX"
