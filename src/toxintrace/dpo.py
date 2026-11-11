"""
ToxinTrace: DPO baseline (IF-Guide paper, App. C.3) with TRL's DPOTrainer.

The paper uses the toxicity preference pairs of Lee et al. (2024) and "the exact hyperparameters
reported in their work". Their config (github.com/ajyl/dpo_toxic, toxicity/train_dpo/config):
beta 0.1, lr 1e-6, RMSprop, batch 4, no gradient accumulation, max_grad_norm 10, 150 linear warmup
steps then constant lr, prompts and responses truncated to 64 tokens, up to 5 epochs, the last 64
pairs held out, evaluation every 100 examples (25 steps) with early stopping after 30 evaluations
without a lower validation loss, keeping the best checkpoint.

Their pairs: prompt_text (prompt), unpert_gen_text (preferred, non-toxic) and pert_gen_text
(rejected, PPLM-steered toxic), in every *.jsonl file under --data_dir.

Run from src/:  python toxintrace/dpo.py --model_name pythia-160m --checkpoint_dir <BASE>/final --save_id baseline
"""
import os
import sys
import json
import time
import logging
import argparse

import torch
from datasets import Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.models import construct_hf_model, get_registry_config  # noqa: E402
from utils.training import LogToFileCallback  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="DPO fine-tuning on the Lee et al. (2024) toxicity pairs.")
    parser.add_argument("--model_name", type=str, required=True, help="Registry model name (see utils/registry.yaml).")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Model to fine-tune (e.g. <BASE>/final).")
    parser.add_argument("--data_dir", type=str, default=os.path.expanduser("~/toxintrace/data/dpo_toxic/toxicity_pairwise"),
                        help="Directory holding Lee et al.'s *.jsonl pair files.")
    parser.add_argument("--save_dir", type=str, default=os.path.expanduser("~/toxintrace/models"))
    parser.add_argument("--save_id", type=str, required=True, help="Run name, e.g. baseline or ifguide.")
    parser.add_argument("--seed", type=int, default=42, help="Lee et al.'s seed.")

    # Lee et al. (2024) hyperparameters
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_prompt_length", type=int, default=64)
    parser.add_argument("--max_completion_length", type=int, default=64)
    parser.add_argument("--valid_size", type=int, default=64)
    parser.add_argument("--eval_every_examples", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)

    # ToxinTrace: smoke-test overrides
    parser.add_argument("--max_pairs", type=int, default=None, help="Use only the first N pairs (smoke tests).")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after N steps (smoke tests).")
    parser.add_argument("--no_cuda", action="store_true", default=False)
    return parser.parse_args()


def load_pairs(data_dir: str, max_pairs: int = None) -> list:
    # Same records and held-out tail as Lee et al.; files are sorted so the split is reproducible
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".jsonl"))
    if not files:
        raise ValueError(f"No .jsonl files in {data_dir}")
    pairs = []
    for name in files:
        with open(os.path.join(data_dir, name)) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    pairs.append({
                        "prompt": row["prompt_text"],
                        "chosen": row["unpert_gen_text"],
                        "rejected": row["pert_gen_text"],
                    })
    logging.info(f"Loaded {len(pairs)} pairs from {len(files)} files in {data_dir}")
    return pairs[:max_pairs] if max_pairs else pairs


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # as in Lee et al.

    args.save_dir = os.path.join(args.save_dir, "dpo", args.model_name, args.save_id, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    config = get_registry_config(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['path'], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = load_pairs(args.data_dir, args.max_pairs)
    # Lee et al. tokenize prompts with truncation=True (right side), i.e. keep the FIRST max_prompt_length tokens;
    # TRL's DPOTrainer would keep the LAST ones, so cut prompts here to match
    long_prompts = 0
    for p in pairs:
        ids = tokenizer(p["prompt"]).input_ids
        if len(ids) > args.max_prompt_length:
            long_prompts += 1
            p["prompt"] = tokenizer.decode(ids[:args.max_prompt_length])
    train_dataset = Dataset.from_list(pairs[:-args.valid_size])
    eval_dataset = Dataset.from_list(pairs[-args.valid_size:])
    logging.info(f"Train pairs: {len(train_dataset)}, held-out pairs: {len(eval_dataset)}, "
                 f"prompts cut to their first {args.max_prompt_length} tokens: {long_prompts}")

    # The registry loads bf16 weights; DPO runs in fp32 (tf32 matmuls), as Lee et al. did, so pass the dtype explicitly
    model = construct_hf_model(registry_config=config, checkpoint_dir=args.checkpoint_dir).to(torch.float32)
    ref_model = construct_hf_model(registry_config=config, checkpoint_dir=args.checkpoint_dir).to(torch.float32)

    # Lee et al.: RMSprop, linear warmup to the base lr, then constant
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / (args.warmup_steps + 1)))

    eval_steps = max(1, args.eval_every_examples // args.train_batch_size)
    training_args = DPOConfig(
        output_dir=args.save_dir,
        beta=args.beta,
        loss_type="sigmoid",
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_length=args.max_prompt_length + args.max_completion_length,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=eval_steps,
        bf16=False,
        seed=args.seed,
        report_to="none",
        use_cpu=args.no_cuda,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.patience),
            LogToFileCallback(os.path.join(args.save_dir, "log.out")),
        ],
    )
    trainer.train()

    logging.info(f"Best checkpoint: {trainer.state.best_model_checkpoint} (eval_loss {trainer.state.best_metric})")
    trainer.save_model(os.path.join(args.save_dir, "final"))
