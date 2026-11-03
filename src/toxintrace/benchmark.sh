#!/usr/bin/env bash
# ToxinTrace: speed benchmarks on the real Pythia-160M workloads (project guide, Phase 3.2).
# Short runs of the authors' scripts at the paper's settings, then projected hours for the full runs.
# Usage (from src/):  bash toxintrace/benchmark.sh
set -eo pipefail
source ~/venvs/ifguide/bin/activate
export IFGUIDE_OWT_PATH=${IFGUIDE_OWT_PATH:-$HOME/toxintrace/data/owt_subset}
# Under WSL the driver lets PyTorch reserve past the 24 GB card into system RAM instead of failing, so
# RAD's fragmented cache spilled over PCIe and ran ~10x slower. Expandable segments avoid the fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
OWT_IDX=$HOME/toxintrace/data/owt_subset_indices
B=$HOME/toxintrace/bench
M=pythia-160m
mkdir -p $B/idx $B/logs
: > $B/timings.txt

python - "$B" <<'EOF'
import sys, numpy as np
B = sys.argv[1]
np.save(f"{B}/idx/train_7680.npy", np.arange(7680))          # 30 optimizer steps of 256 x 2048 tokens
np.save(f"{B}/idx/valid_64.npy", np.arange(488282, 488282 + 64))
for n in (256, 1024):
    np.save(f"{B}/idx/train_{n}.npy", np.arange(n))
EOF

# bench <name> <command...>: wall time and peak GPU memory of one command
bench() {
    local name=$1; shift
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $B/logs/$name.mem &
    local smi=$! start=$SECONDS
    if ! "$@" > $B/logs/$name.log 2>&1; then
        kill $smi; echo "FAILED: $name (see $B/logs/$name.log)"; tail -n 20 $B/logs/$name.log; exit 1
    fi
    kill $smi
    local secs=$((SECONDS - start)) peak=$(sort -n $B/logs/$name.mem | tail -1)
    echo "$name $secs $peak" >> $B/timings.txt
    echo "=== $name: ${secs}s, peak GPU memory ${peak} MiB"
}

bench train torchrun --nproc-per-node=1 --master_port=29512 train.py --model_name $M --save_dir $B/models --save_id bench \
    --train_indices_path $B/idx/train_7680.npy --valid_indices_path $B/idx/valid_64.npy \
    --learning_rate 6e-4 --weight_decay 4e-4 --warmup_ratio 0.01 --max_grad_norm 1 \
    --train_batch_size 8 --eval_batch_size 8 --num_train_epochs 1 --logging_steps 5 --save_steps 1000 --seed 1004
CKPT=$(ls -d $B/models/training/$M/bench/*/ | sort | tail -n 1)final
grep "train_runtime" $(dirname $CKPT)/log.out | tail -n 1

for n in 256 1024; do
    bench fit_$n python fit_factors.py --model_name $M --checkpoint_dir $CKPT --output_dir $B/if_$n \
        --train_indices_path $B/idx/train_$n.npy --factor_strategy ekfac --use_half_precision --train_batch_size 4
    bench scores_$n python compute_scores.py --model_name $M --checkpoint_dir $CKPT --factors_path $B/if_$n \
        --train_indices_path $B/idx/train_$n.npy --query_dataset RTP \
        --toxic_query_indices_path ../data/RTP/query_indices/toxic_indices.npy \
        --nontoxic_query_indices_path ../data/RTP/query_indices/nontoxic_indices.npy \
        --save_dir RTP --factor_strategy ekfac --factors_name ekfac_half --use_half_precision \
        --query_batch_size 4 --train_batch_size 4
done

bench eval_rtp200 python run_toxicity_eval.py --model_name $M --checkpoint_dir $CKPT --dataset RTP \
    --num_examples 200 --num_return_sequences 25 --batch_size 128 --output_dir $B/results --save_dir bench --save_outputs
bench implicit_rtp200 python run_implicit_toxicity_eval.py --outputs_file_path $B/results/RTP/$M/bench/outputs.json \
    --dataset RTP --batch_size 128
bench rad_rtp40 python run_toxicity_eval.py --model_name $M --checkpoint_dir $CKPT --dataset RTP \
    --num_examples 40 --num_return_sequences 25 --batch_size 64 --output_dir $B/results --save_dir bench-rad \
    --decoding_defense rad --save_outputs     # batch 128 needs ~30 GB with RAD's reward-model caches
bench fluency_full python run_fluency_eval.py --model_name $M --checkpoint_dir $CKPT \
    --owt_test_indicies $OWT_IDX/test_indices.npy --batch_size 16 --output_dir $B/results --save_dir bench

# Projections for the full runs
python - "$B" "$CKPT" <<'EOF'
import ast, re, sys
B, CKPT = sys.argv[1:]
t = {l.split()[0]: (int(l.split()[1]), int(l.split()[2])) for l in open(f"{B}/timings.txt")}
s = lambda k: t[k][0]

def loop_s(name, desc):
    """Seconds one Kronfluence loop took, from the 'time spent' of its last tqdm line."""
    text = open(f"{B}/logs/{name}.log").read().replace("\r", "\n")
    spent = re.findall(rf"{re.escape(desc)} \[(\d+)/\1\] 100%[^\n]*time spent: ([\d:]+)", text)[-1][1]
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(spent.split(":"))))
run = [ast.literal_eval(l) for l in open(CKPT.rsplit("/", 1)[0] + "/log.out") if "train_runtime" in l][-1]
TRAIN, VALID, PROMPTS = 488_282, 24_415, 10_000 + 1_402 + 7_201      # chunks; RTP+AttaQ+BOLD prompts
steps = round(TRAIN * 4 / 256)                                        # 4 epochs at 256 x 2048 tokens per step
step_s = run["train_runtime"] / 30
fwd_chunk_s = s("fluency_full") / 4_883 * 0.8                         # forward-only cost per 2048-token chunk (approx.)
evals_s = (steps // 500) * VALID * fwd_chunk_s                        # train.py evaluates every 500 steps
# Per-chunk costs come from the 1,024-chunk loops; wall-clock differences between the two sizes
# are swamped by run-to-run noise in the ~24 min query-gradient pass.
fit_chunk = (loop_s("fit_1024", "Fitting covariance matrices") + loop_s("fit_1024", "Fitting Lambda matrices")) / 1024
fit_fixed = s("fit_1024") - fit_chunk * 1024
score_chunk = loop_s("scores_1024", "Computing pairwise scores (training gradient)") / 1024
score_fixed = s("scores_1024") - score_chunk * 1024
h = lambda x: x / 3600
print(f"\nTraining: {step_s:.1f} s/step, {256 * 2048 / step_s / 1e3:.0f}k tokens/s, peak {t['train'][1]} MiB")
print(f"  full pre-training ({steps} steps + {steps // 500} validation passes): {h(steps * step_s + evals_s):.1f} h")
print(f"EK-FAC factors: {fit_chunk * 1e3:.0f} ms/chunk + {fit_fixed:.0f} s fixed -> {h(fit_fixed + fit_chunk * TRAIN):.1f} h for 488,282 chunks (peak {t['fit_1024'][1]} MiB)")
print(f"Influence scores: {score_chunk * 1e3:.0f} ms/chunk + {score_fixed:.0f} s fixed (query gradients) -> {h(score_fixed + score_chunk * TRAIN):.1f} h (peak {t['scores_1024'][1]} MiB)")
gen = s("eval_rtp200") / 200; imp = s("implicit_rtp200") / 200; rad = s("rad_rtp40") / 40
print(f"Explicit eval: {gen:.2f} s/prompt (25 samples) -> {h(gen * PROMPTS):.1f} h per model for RTP+AttaQ+BOLD")
print(f"Implicit eval: {imp:.2f} s/prompt -> {h(imp * PROMPTS):.1f} h per model")
print(f"Fluency eval (full OWT test + LAMBADA): {s('fluency_full') / 60:.1f} min per model")
print(f"RAD eval: {rad:.2f} s/prompt -> {h(rad * PROMPTS):.1f} h per model")
EOF
