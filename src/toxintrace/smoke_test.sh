#!/usr/bin/env bash
# ToxinTrace: end-to-end smoke test of the IF-Guide pipeline on tiny models (project guide, Phase 2).
# Every stage runs the authors' real scripts at toy scale; the run stops at the first failure.
# Usage (from src/):  bash toxintrace/smoke_test.sh [first_stage]
set -o pipefail
FIRST=${1:-0}

TT_VENV=${TT_VENV:-$HOME/venvs/ifguide}; if [ -f "$TT_VENV/bin/activate" ]; then source "$TT_VENV/bin/activate"; fi
export IFGUIDE_OWT_PATH=${IFGUIDE_OWT_PATH:-$HOME/toxintrace/data/owt_smoke}
IDX=$HOME/toxintrace/data/smoke_indices
S=${TT_SMOKE:-$HOME/toxintrace/smoke}
LOG=$S/logs
M=pythia-14m-smoke
PORT=29511
mkdir -p "$LOG"
# evaluation results from an earlier smoke run would be reused (or rejected if their paths differ)
if [ "$FIRST" -le 7 ]; then rm -rf "$S/results"; fi

run() {
    local stage=$1 name=$2; shift 2
    if [ "$stage" -lt "$FIRST" ]; then echo "--- skip stage $stage: $name"; return; fi
    echo "=== [$(date +%T)] stage $stage: $name"
    local start=$SECONDS
    if "$@" > "$LOG/$stage-$name.log" 2>&1; then
        echo "    ok in $((SECONDS - start))s"
    else
        echo "    FAILED after $((SECONDS - start))s; last lines of $LOG/$stage-$name.log:"
        tail -n 30 "$LOG/$stage-$name.log"
        exit 1
    fi
}
latest() { local d; d=$(ls -d "$1"/*/ 2>/dev/null | sort | tail -n 1); echo "${d%/}"; }   # newest timestamped run

TRAIN_ARGS=(--model_name $M --save_dir $S/models
            --train_indices_path $IDX/owt_train.npy --valid_indices_path $IDX/owt_valid.npy
            --learning_rate 6e-4 --weight_decay 4e-4 --warmup_ratio 0.01 --max_grad_norm 1
            --train_batch_size 4 --eval_batch_size 4 --effective_batch_size 16
            --logging_steps 10 --save_steps 50 --seed 1004)

# --- data
run 0 data python toxintrace/make_smoke_data.py --output_dir "$IFGUIDE_OWT_PATH" --index_dir "$IDX"

# --- base model, influence factors, token scores, toxic-token mask
run 1 train-base torchrun --nproc-per-node=1 --master_port=$PORT train.py "${TRAIN_ARGS[@]}" \
    --num_train_epochs 1 --save_id baseline
BASE=$(latest $S/models/training/$M/baseline)

run 2 fit-factors python fit_factors.py --model_name $M --checkpoint_dir $BASE/final --output_dir $BASE \
    --train_indices_path $IDX/owt_train.npy --factor_strategy ekfac --use_half_precision --train_batch_size 4

run 3 compute-scores python compute_scores.py --model_name $M --checkpoint_dir $BASE/final --factors_path $BASE \
    --train_indices_path $IDX/owt_train.npy --query_dataset RTP \
    --toxic_query_indices_path $IDX/rtp_query_toxic.npy --nontoxic_query_indices_path $IDX/rtp_query_nontoxic.npy \
    --save_dir RTP --factor_strategy ekfac --factors_name ekfac_half --use_half_precision \
    --query_batch_size 4 --train_batch_size 4
SCORES=$BASE/if_results/scores_RTP/ekfac_half/pairwise_scores.safetensors

# 2% of the 1600 x 2048 smoke training tokens
run 4 build-mask python build_toxic_token_mask.py --model_name $M --train_indices_path $IDX/owt_train.npy \
    --scores_path $SCORES --window 1 --toxicity_threshold 0.99 --max_tokens 65536 \
    --output_root $S/masks --inspection_idx 0
MASK=$S/masks/ekfac_RTP_$M/mask_toks=0.1m_p=0.99_w=1.pt

# --- training with the IF-Guide penalty, from scratch and as a fine-tune
run 5 train-ifguide torchrun --nproc-per-node=1 --master_port=$PORT train.py "${TRAIN_ARGS[@]}" \
    --num_train_epochs 1 --save_id ifguide --toxic_token_mask_path $MASK --toxic_lambda 1.0
IFG=$(latest $S/models/training/$M/ifguide)

run 6 finetune-ifguide torchrun --nproc-per-node=1 --master_port=$PORT finetune.py --model_name $M \
    --checkpoint_dir $BASE/final --save_dir $S/models --save_id ft-ifguide \
    --train_indices_path $IDX/owt_train.npy --valid_indices_path $IDX/owt_valid.npy \
    --train_batch_size 4 --eval_batch_size 4 --effective_batch_size 16 --max_steps 5 --logging_steps 1 \
    --toxic_token_mask_path $MASK --toxic_lambda 1.0

# --- evaluation
tox_eval() {
    local defense=$1 tag=$2 n=$3; shift 3
    for D in "$@"; do
        python run_toxicity_eval.py --model_name $M --checkpoint_dir $IFG/final --dataset $D \
            --num_examples $n --num_return_sequences 2 --batch_size 32 \
            --output_dir $S/results --save_dir $tag --decoding_defense $defense --save_outputs || return 1
    done
}
implicit_eval() {
    for D in RTP AttaQ BOLD; do
        python run_implicit_toxicity_eval.py --outputs_file_path $S/results/$D/$M/ifguide/outputs.json \
            --dataset $D --batch_size 64 || return 1
    done
}
fluency_eval() {
    python run_fluency_eval.py --model_name $M --checkpoint_dir $IFG/final \
        --owt_test_indicies $IDX/owt_test.npy --lambada_test_indicies $IDX/lambada_test.npy \
        --batch_size 8 --output_dir $S/results --save_dir $2 --decoding_defense $1
}
rad_download() {
    # The README's storage.googleapis.com/rad_release bucket no longer exists (404 NoSuchBucket);
    # the RAD authors now host the reward models on the Hugging Face Hub (hk/rad_rms).
    local d=utils/rad/reward_modeling/saved_models
    rm -f utils/rad/reward_modeling/saved_models.zip
    if [ -f $d/gpt2_toxicity/pytorch_model.bin ]; then echo "reward model already present"; return 0; fi
    python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('hk/rad_rms', 'gpt2_toxicity/pytorch_model.bin', local_dir='$d'))" \
        && du -sh $d/gpt2_toxicity
}

run 7 toxicity-eval tox_eval none ifguide 20 RTP AttaQ BOLD
run 8 implicit-eval implicit_eval
run 9 fluency-eval fluency_eval none ifguide
run 10 rad-download rad_download
run 11 rad-toxicity-eval tox_eval rad ifguide-rad 10 RTP
run 12 rad-fluency-eval fluency_eval rad ifguide-rad

# --- summary
python - "$S" "$M" "$MASK" <<'EOF'
import json, os, sys, torch
S, M, MASK = sys.argv[1:]
mask = torch.load(MASK, weights_only=True)
print(f"mask: {int(mask.sum())} of {mask.numel()} tokens selected ({100 * mask.float().mean():.2f}%)")
for D in ("RTP", "AttaQ", "BOLD"):
    r = json.load(open(f"{S}/results/{D}/{M}/ifguide/results.json"))["all_prompts"]
    i = json.load(open(f"{S}/results/{D}/{M}/ifguide/implicit_results.json"))["all_prompts"]
    print(f"{D:5s} EMT {r['maximum_toxicity']['mean']:.3f}  TP {r['toxicity_probability']:.3f}  "
          f"implicit EMT {i['maximum_implicit_toxicity']['mean']:.3f}")
r = json.load(open(f"{S}/results/RTP/{M}/ifguide-rad/results.json"))["all_prompts"]
print(f"RTP+RAD EMT {r['maximum_toxicity']['mean']:.3f}  TP {r['toxicity_probability']:.3f}")
print("fluency:", json.load(open(f"{S}/results/fluency/{M}/ifguide/results.json")))
print("fluency+RAD:", json.load(open(f"{S}/results/fluency/{M}/ifguide-rad/results.json")))
EOF
echo "=== smoke test finished $(date +%T)"
