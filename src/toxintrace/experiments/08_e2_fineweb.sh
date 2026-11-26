#!/usr/bin/env bash
# E2: does IF-Guide still work on a newer corpus? On an RTX 4090, re-running skips finished steps:
#   1. 400M GPT-NeoX tokens from FineWeb CC-MAIN-2025-26 (2.2 GB download, ~20 min) and their Detoxify
#      toxicity rate against OpenWebText (~10 min)
#   2. EK-FAC factors and differential scores with the OpenWebText-trained base Pythia-160M (~6 h),
#      the 2% mask (8M tokens, ~3 min) and a first look at what it selects (~5 min)
#   3. pre-trained Pythia-410M fine-tuned on it for one pass, without and with the mask (3.4 h and 4.0 h)
#   4. evaluation of both and of the untouched Pythia-410M (OpenWebText perplexity as everywhere else, plus
#      perplexity on held-out FineWeb chunks and BOLD without the trailing space; ~4 h), then one summary table
# E2_MB sets the micro-batch (default 4; 8 runs out of memory on a 24 GB card; 256 sequences per step either way).
source "$(dirname "$0")/env.sh"
FW=$TT_ROOT/data/fineweb_subset
FWI=$TT_ROOT/data/fineweb_subset_indices
E2=$TT_ROOT/models/e2_fineweb          # own folder: the OpenWebText factors and scores are never touched
SCORES=$E2/if_results/scores_RTP/ekfac_half/pairwise_scores.safetensors
E2MASK=../data/toxic_token_masks/ekfac_RTP_pythia-160m_fineweb/mask_toks=8.0m_p=0.99_w=1.pt
M410=pythia-410m-pretrained
OWT=$IFGUIDE_OWT_PATH                  # OpenWebText chunks, for perplexity on the usual test set
export IFGUIDE_OWT_PATH=$FW            # everything else reads FineWeb

if [ ! -d "$FW" ]; then
  run python toxintrace/make_fineweb_subset.py --output_dir "$FW" --index_dir "$FWI" --cache_dir "$TT_ROOT/data/fineweb_raw"
fi
if [ ! -f "$OUT/e2_toxicity/summary.json" ]; then
  TT_ROOT="$TT_ROOT" run python toxintrace/compare_toxicity.py --out_dir "$OUT/e2_toxicity"
fi

if [ ! -f "$SCORES" ]; then
  BASE=$(base_run)                     # scored with the OpenWebText-trained base Pythia-160M
  run python fit_factors.py --model_name $MODEL --checkpoint_dir "$BASE/final" --output_dir "$E2" \
    --train_indices_path "$FWI/train_indices.npy" --factor_strategy ekfac --use_half_precision --train_batch_size 4
  run python compute_scores.py --model_name $MODEL --checkpoint_dir "$BASE/final" --factors_path "$E2" \
    --train_indices_path "$FWI/train_indices.npy" --query_dataset RTP \
    --toxic_query_indices_path ../data/RTP/query_indices/toxic_indices.npy \
    --nontoxic_query_indices_path ../data/RTP/query_indices/nontoxic_indices.npy \
    --save_dir RTP --factor_strategy ekfac --factors_name ekfac_half --use_half_precision \
    --query_batch_size 4 --train_batch_size 4
fi
if [ ! -f "$E2MASK" ]; then
  run python build_toxic_token_mask.py --model_name $MODEL --train_indices_path "$FWI/train_indices.npy" \
    --scores_path "$SCORES" --window 1 --toxicity_threshold 0.99 --max_tokens 8000000 --save_id fineweb --inspection_idx 0
fi
if [ ! -f "$OUT/e2_mask/summary.json" ]; then
  TT_ROOT="$TT_ROOT" run python toxintrace/inspect_mask.py --out_dir "$OUT/e2_mask"
fi

# One pass: 762 steps x 256 sequences = 195,072 of the 195,313 chunks (399.5M tokens).
# Learning rate, weight decay, warm-up, clipping and seed are finetune.py's defaults (the paper's).
for N in e2_control e2_ifguide; do
  if latest_with "$TT_ROOT/models/finetuning/$M410/$N" >/dev/null 2>&1; then continue; fi
  EXTRA=(); if [ $N = e2_ifguide ]; then EXTRA=(--toxic_token_mask_path "$E2MASK" --toxic_lambda 1.0); fi
  run python finetune.py --model_name $M410 --save_dir "$TT_ROOT/models" --save_id $N \
    --train_indices_path "$FWI/train_indices.npy" --valid_indices_path "$FWI/valid_indices.npy" \
    --train_batch_size ${E2_MB:-4} --effective_batch_size 256 --max_steps 762 --logging_steps 10 --save_steps 250 \
    "${EXTRA[@]}"
done

for N in e2_control e2_ifguide e2_base; do
  F=""; if [ $N != e2_base ]; then F=$(latest_with "$TT_ROOT/models/finetuning/$M410/$N")/final; fi   # "" = Hub weights
  if [ ! -f "$OUT/fluency/$M410/$N/results.json" ]; then
    ( export IFGUIDE_OWT_PATH=$OWT; evaluate $N "$F" none $M410 )
  fi
  if [ ! -f "$OUT/fluency/$M410/${N}_fineweb/results.json" ]; then
    run python run_fluency_eval.py --model_name $M410 --checkpoint_dir "$F" --owt_test_indicies "$FWI/valid_indices.npy" \
      --batch_size 16 --output_dir "$OUT" --save_dir ${N}_fineweb --decoding_defense none
  fi
  if [ ! -f "$TT_ROOT/results_boldstrip/BOLD/$M410/$N/results.json" ]; then
    IFGUIDE_STRIP_BOLD=1 run python run_toxicity_eval.py --model_name $M410 --checkpoint_dir "$F" --dataset BOLD \
      --batch_size 128 --decoding_defense none --output_dir "$TT_ROOT/results_boldstrip" --save_dir $N --save_outputs
  fi
done

# Every metric for the three models, and control / mask factors (intervals: 09_analysis.sh)
run python toxintrace/e2_summary.py --model $M410 --results_dir "$OUT" --strip_dir "$TT_ROOT/results_boldstrip" \
  --out_csv "$OUT/e2/summary.csv"
