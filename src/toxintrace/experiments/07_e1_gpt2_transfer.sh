#!/usr/bin/env bash
# E1: move the Pythia mask to GPT-2's tokenizer (8 min, CPU), fine-tune GPT-2 without and with it
# (~2.3 h each), and evaluate both (~35 min each)
source "$(dirname "$0")/env.sh"
G=gpt2-pretrained; D=$TT_ROOT/data/e1_gpt2; T=$TT_ROOT/data/e1_gpt2_test
run python toxintrace/transfer_mask.py --mask_path $MASK --output_dir "$D" --num_proc 8 \
  --train_indices_path "$IDX/train_indices.npy"
run python -c "import numpy as np; np.save('$D/train_indices.npy', np.arange(390656)); np.save('$D/valid_indices.npy', np.arange(256))"
# GPT-2 blocks of the OpenWebText test chunks (perplexity); transfer_mask.py needs a mask, so give it an empty one
run python -c "import torch; torch.save(torch.zeros(4883, 2048, dtype=torch.bool), '$TT_ROOT/zero_mask_test.pt')"
run python toxintrace/transfer_mask.py --mask_path "$TT_ROOT/zero_mask_test.pt" --output_dir "$T" \
  --train_indices_path "$IDX/test_indices.npy" --num_blocks 9700 --num_proc 8
run python -c "import numpy as np; np.save('$T/test_indices.npy', np.arange(9700))"
COMMON=(--model_name $G --save_dir "$TT_ROOT/models" --train_indices_path "$D/train_indices.npy"
        --valid_indices_path "$D/valid_indices.npy" --train_batch_size 8 --effective_batch_size 256
        --max_steps 1526 --logging_steps 25 --save_steps 500)
export IFGUIDE_TOKENIZED_PATH=$D/dataset
run python finetune.py "${COMMON[@]}" --save_id e1_control
run python finetune.py "${COMMON[@]}" --save_id e1_transfer --toxic_token_mask_path "$D/mask.pt" --toxic_lambda 1.0
export IFGUIDE_TOKENIZED_PATH=$T/dataset
for N in e1_control e1_transfer; do
  F=$(latest_with "$TT_ROOT/models/finetuning/$G/$N")/final
  # checkpoints are saved with transposed Linear layers; write a standard GPT-2 copy
  if [ ! -d "${F}_conv1d" ]; then run python toxintrace/to_conv1d.py "$F" "${F}_conv1d"; fi
  evaluate $N "${F}_conv1d" none $G "$T/test_indices.npy"
done
run python run_fluency_eval.py --model_name $G --owt_test_indicies "$T/test_indices.npy" --batch_size 16 \
  --output_dir "$OUT" --save_dir gpt2_base --decoding_defense none
