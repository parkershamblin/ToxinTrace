#!/usr/bin/env bash
# E2: 400M GPT-NeoX tokens from FineWeb CC-MAIN-2025-26 (2.2 GB download, ~20 min), its Detoxify toxicity rate
# against OpenWebText (~10 min), EK-FAC factors and scores on it (~6 h on an RTX 4090), the 2% mask (8M tokens,
# ~3 min) and a first look at it. The Pythia-410M fine-tunes are not run yet.
source "$(dirname "$0")/env.sh"
FW=$TT_ROOT/data/fineweb_subset
FWIDX=$TT_ROOT/data/fineweb_subset_indices/train_indices.npy
E2=$TT_ROOT/models/e2_fineweb          # own folder: the OpenWebText factors and scores are never touched
if [ ! -d "$FW" ]; then
  run python toxintrace/make_fineweb_subset.py --output_dir "$FW" \
    --index_dir "$TT_ROOT/data/fineweb_subset_indices" --cache_dir "$TT_ROOT/data/fineweb_raw"
fi
TT_ROOT="$TT_ROOT" run python toxintrace/compare_toxicity.py --out_dir "$OUT/e2_toxicity"

BASE=$(base_run)                       # scored with the OpenWebText-trained base Pythia-160M
export IFGUIDE_OWT_PATH=$FW
run python fit_factors.py --model_name $MODEL --checkpoint_dir "$BASE/final" --output_dir "$E2" \
  --train_indices_path "$FWIDX" --factor_strategy ekfac --use_half_precision --train_batch_size 4
run python compute_scores.py --model_name $MODEL --checkpoint_dir "$BASE/final" --factors_path "$E2" \
  --train_indices_path "$FWIDX" --query_dataset RTP \
  --toxic_query_indices_path ../data/RTP/query_indices/toxic_indices.npy \
  --nontoxic_query_indices_path ../data/RTP/query_indices/nontoxic_indices.npy \
  --save_dir RTP --factor_strategy ekfac --factors_name ekfac_half --use_half_precision \
  --query_batch_size 4 --train_batch_size 4
run python build_toxic_token_mask.py --model_name $MODEL --train_indices_path "$FWIDX" \
  --scores_path "$E2/if_results/scores_RTP/ekfac_half/pairwise_scores.safetensors" \
  --window 1 --toxicity_threshold 0.99 --max_tokens 8000000 --save_id fineweb --inspection_idx 0
TT_ROOT="$TT_ROOT" run python toxintrace/inspect_mask.py --out_dir "$OUT/e2_mask"
