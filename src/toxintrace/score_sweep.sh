#!/usr/bin/env bash
# Usage: score_sweep.sh <name> <start_model_dir> <sweep_run_dir>
# Scores the start model (step 0) and every checkpoint-N of a DPO sweep, in step order:
# OpenWebText perplexity + LAMBADA, and RTP toxicity on a 1,000-prompt subset (500 toxic + 500 non-toxic).
# Stops early once perplexity is >15% above the start model for two checkpoints in a row.
NAME=$1; START=$2; RUN=$3
source ~/venvs/ifguide/bin/activate
export IFGUIDE_OWT_PATH=$HOME/toxintrace/data/owt_subset
OUT=$HOME/toxintrace/results_sweep
cd ~/toxintrace/IF-Guide/src
score() {  # <tag> <ckpt>
  python run_fluency_eval.py --model_name pythia-160m --checkpoint_dir $2 \
    --owt_test_indicies $HOME/toxintrace/data/owt_subset_indices/test_indices.npy --batch_size 16 \
    --output_dir $OUT --save_dir $1 --decoding_defense none > /dev/null 2>&1
  echo "=== $1 fluency exit $?"
  python run_toxicity_eval.py --model_name pythia-160m --checkpoint_dir $2 --dataset RTP --num_examples 1000 \
    --batch_size 128 --output_dir $OUT --save_dir $1 --decoding_defense none > /dev/null 2>&1
  echo "=== $1 RTP exit $?"
}
ppl() { python -c "import json;print(json.load(open('$OUT/fluency/pythia-160m/$1/results.json'))['openwebtext']['perplexity'])"; }
score ${NAME}_0000 $START
BASE=$(ppl ${NAME}_0000); echo "=== $NAME start ppl $BASE $(date)"
over=0
for C in $(ls -d $RUN/checkpoint-* | sort -t- -k2 -n); do
  S=$(printf "%04d" ${C##*-})
  score ${NAME}_$S $C
  P=$(ppl ${NAME}_$S); echo "=== $NAME step $S ppl $P $(date)"
  if python -c "import sys; sys.exit(0 if $P > 1.15*$BASE else 1)"; then over=$((over+1)); else over=0; fi
  if [ $over -ge 2 ]; then echo "=== $NAME stop: ppl >15% over start twice"; break; fi
done
