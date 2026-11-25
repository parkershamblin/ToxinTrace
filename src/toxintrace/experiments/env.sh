# Shared settings for every ToxinTrace experiment script (sourced, not run).
#   TT_ROOT  data, models, results and logs        (default: ~/toxintrace)
#   TT_VENV  Python environment, if it exists      (default: ~/venvs/ifguide)
#   DRY_RUN=1  print the commands instead of running them
set -eo pipefail
TT_ROOT=${TT_ROOT:-$HOME/toxintrace}
TT_VENV=${TT_VENV:-$HOME/venvs/ifguide}
if [ -f "$TT_VENV/bin/activate" ]; then source "$TT_VENV/bin/activate"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IFGUIDE_OWT_PATH=${IFGUIDE_OWT_PATH:-$TT_ROOT/data/owt_subset}
IDX=$TT_ROOT/data/owt_subset_indices
OUT=$TT_ROOT/results
MODEL=pythia-160m
MASK=../data/toxic_token_masks/ekfac_RTP_pythia-160m/mask_toks=20.0m_p=0.99_w=1.pt
cd "$(dirname "${BASH_SOURCE[0]}")/../.."          # every script runs from src/

run() { echo "+ $*"; if [ -z "$DRY_RUN" ]; then "$@"; fi; }
# newest run folder under $1 that contains $2 (default: final)
latest_with() {
  local d
  for d in $(ls -d "$1"/*/ 2>/dev/null | sort -r); do
    if [ -e "${d}${2:-final}" ]; then echo "${d%/}"; return 0; fi
  done
  if [ -n "$DRY_RUN" ]; then echo "$1/<timestamp>"; return 0; fi
  echo "no run with ${2:-final} under $1; run the earlier step first" >&2; return 1
}
base_run()    { latest_with "$TT_ROOT/models/training/$MODEL/baseline"; }
ifguide_run() { latest_with "$TT_ROOT/models/training/$MODEL/ifguide"; }

# full evaluation of one model: evaluate <name> <checkpoint_dir> [none|rad] [model_name] [test_indices]
evaluate() {
  local name=$1 ckpt=$2 defense=${3:-none} model=${4:-$MODEL} test=${5:-$IDX/test_indices.npy}
  local bs=128; if [ "$defense" = rad ]; then bs=32; fi      # RAD at 64 exceeds 24 GB
  local D
  for D in RTP AttaQ BOLD; do
    run python run_toxicity_eval.py --model_name $model --checkpoint_dir "$ckpt" --dataset $D \
      --batch_size $bs --output_dir "$OUT" --save_dir $name --decoding_defense $defense --save_outputs
    run python run_implicit_toxicity_eval.py --outputs_file_path "$OUT/$D/$model/$name/outputs.json" \
      --dataset $D --batch_size 128
  done
  run python run_fluency_eval.py --model_name $model --checkpoint_dir "$ckpt" \
    --owt_test_indicies "$test" --batch_size 16 --output_dir "$OUT" --save_dir $name --decoding_defense $defense
}
