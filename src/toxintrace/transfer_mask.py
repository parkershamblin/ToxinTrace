"""
ToxinTrace E1: move a GPT-NeoX (Pythia) toxic-token mask onto GPT-2's tokenizer.

For each 2,048-token training chunk, in the order of --train_indices_path (the mask's row order):
  1. map each GPT-NeoX token to its exact bytes (both tokenizers are byte-level BPE) and mark the
     bytes of masked tokens;
  2. decode the bytes to text (invalid UTF-8 at chunk edges is kept 1 byte = 1 char, then replaced
     by U+FFFD so the char -> byte map stays exact);
  3. re-tokenize the text with GPT-2 (return_offsets_mapping=True) and mask every GPT-2 token whose
     bytes overlap a masked byte.
The GPT-2 tokens are concatenated and regrouped into --block_size blocks until --num_blocks blocks.

Outputs (under --output_dir):
  dataset/       HF dataset with input_ids, attention_mask, labels (same columns as the OWT subset)
  mask.pt        bool tensor [num_blocks, block_size], True = penalized token
  stats.json     coverage: masked-byte recall, byte inflation, share of tokens masked, etc.

Run from src/:  python toxintrace/transfer_mask.py --mask_path <mask .pt> --output_dir ~/toxintrace/data/owt_gpt2_1024
"""
import os
import re
import sys
import json
import logging
import argparse
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset
from tokenizers import processors
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.models import get_registry_config  # noqa: E402
from utils.datasets import get_tokenized_openwebtext  # noqa: E402

SURROGATES = re.compile("[\udc80-\udcff]")

# Source data set in the parent before forking; the target tokenizer is set in each worker by init_worker
SRC = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Transfer a GPT-NeoX toxic-token mask to the GPT-2 tokenizer.")
    parser.add_argument("--source_model_name", type=str, default="pythia-160m")
    parser.add_argument("--target_tokenizer", type=str, default="gpt2")
    parser.add_argument("--train_indices_path", type=str, default=os.path.expanduser("~/toxintrace/data/owt_subset_indices/train_indices.npy"))
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--num_blocks", type=int, default=1526 * 256, help="1,526 steps x 256 blocks ~ 400M tokens.")
    parser.add_argument("--chunks_per_task", type=int, default=256)
    parser.add_argument("--num_proc", type=int, default=8)
    return parser.parse_args()


def build_byte_table(tokenizer) -> list:
    """Exact bytes of every GPT-NeoX token id."""
    byte_decoder = {u: b for b, u in bytes_to_unicode().items()}
    table, special = [], 0
    for tok_str in tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))):
        if tok_str is not None and all(c in byte_decoder for c in tok_str):
            table.append(bytes(byte_decoder[c] for c in tok_str))
        else:
            # Added/special tokens are stored as plain text
            table.append((tok_str or "").encode("utf-8"))
            special += 1
    logging.info(f"Byte table: {len(table)} ids, {special} added/special tokens")
    return table


def init_worker(target_tokenizer):
    # dataset, mask and byte table are inherited from the parent through fork (the mask is ~1 GB)
    tok = AutoTokenizer.from_pretrained(target_tokenizer, use_fast=True)
    # Keep the leading space inside each token's offsets so a masked " word" maps to all of its bytes
    tok.backend_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    SRC["tok"] = tok


def char_byte_starts(text: str) -> np.ndarray:
    """Byte offset of every char (plus the end) in the UTF-8 / surrogate-escaped encoding of text."""
    cps = np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)
    lens = np.where(cps < 0x80, 1, np.where(cps < 0x800, 2, np.where(cps < 0x10000, 3, 4)))
    lens = np.where((cps >= 0xDC80) & (cps <= 0xDCFF), 1, lens)  # escaped invalid byte
    return np.concatenate([[0], np.cumsum(lens)])


def transfer_rows(rows: list) -> list:
    ds, mask, table, byte_len, tok = SRC["dataset"], SRC["mask"], SRC["bytes"], SRC["byte_len"], SRC["tok"]
    texts, masked_bytes = [], []
    for r in rows:
        ids = np.asarray(ds[r]["input_ids"], dtype=np.int64)
        raw = b"".join(table[i] for i in ids)
        masked_bytes.append(np.repeat(mask[r].numpy(), byte_len[ids]))
        texts.append(raw.decode("utf-8", "surrogateescape"))

    enc = tok([SURROGATES.sub("�", t) for t in texts], return_offsets_mapping=True, add_special_tokens=False)
    out = []
    for text, mb, ids, offs in zip(texts, masked_bytes, enc["input_ids"], enc["offset_mapping"]):
        starts = char_byte_starts(text)
        csum = np.concatenate([[0], np.cumsum(mb)])
        offs = np.asarray(offs, dtype=np.int64).reshape(-1, 2)
        b0, b1 = starts[offs[:, 0]], starts[offs[:, 1]]
        tgt_mask = (csum[b1] - csum[b0]) > 0
        covered = np.zeros(len(mb) + 1, dtype=np.int64)
        np.add.at(covered, b0[tgt_mask], 1)
        np.add.at(covered, b1[tgt_mask], -1)
        covered = np.cumsum(covered)[:-1] > 0
        out.append((
            np.asarray(ids, dtype=np.int32), tgt_mask,
            int(mb.sum()), int((mb & covered).sum()), int(covered.sum()), len(mb),
            int(SURROGATES.search(text) is not None),
        ))
    return out


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    os.makedirs(args.output_dir, exist_ok=True)

    config = get_registry_config(args.source_model_name)
    src_tok = AutoTokenizer.from_pretrained(config['tokenizer']['path'], use_fast=True)
    dataset = get_tokenized_openwebtext(config, np.load(args.train_indices_path)).select_columns(["input_ids"])
    mask = torch.load(args.mask_path, map_location="cpu", weights_only=True)
    if mask.shape[0] != len(dataset):
        raise ValueError(f"Mask rows {mask.shape[0]} != dataset rows {len(dataset)}")
    table = build_byte_table(src_tok)
    SRC.update(dataset=dataset, mask=mask, bytes=table, byte_len=np.array([len(b) for b in table], dtype=np.int64))

    need = args.num_blocks * args.block_size
    ids_parts, mask_parts = [], []
    totals = dict(chunks=0, src_tokens=0, src_masked_tokens=0, masked_bytes=0, recalled_bytes=0,
                  bytes_in_masked_targets=0, total_bytes=0, chunks_with_invalid_utf8=0)
    have = 0
    tasks = [list(range(i, min(i + args.chunks_per_task, len(dataset)))) for i in range(0, len(dataset), args.chunks_per_task)]
    # Work in rounds with pool.map so the pool is never shut down with tasks in flight (terminate() can hang)
    pool = Pool(args.num_proc, initializer=init_worker, initargs=(args.target_tokenizer,))
    round_size = args.num_proc * 4
    for start in range(0, len(tasks), round_size):
        round_tasks = tasks[start:start + round_size]
        for task, results in zip(round_tasks, pool.map(transfer_rows, round_tasks)):
            for r, (ids, tmask, mbytes, recalled, inflated, nbytes, bad) in zip(task, results):
                ids_parts.append(ids)
                mask_parts.append(tmask)
                have += len(ids)
                totals["chunks"] += 1
                totals["src_tokens"] += mask.shape[1]
                totals["src_masked_tokens"] += int(mask[r].sum())
                totals["masked_bytes"] += mbytes
                totals["recalled_bytes"] += recalled
                totals["bytes_in_masked_targets"] += inflated
                totals["total_bytes"] += nbytes
                totals["chunks_with_invalid_utf8"] += bad
            if totals["chunks"] % (args.chunks_per_task * 100) == 0:
                logging.info(f"{totals['chunks']} chunks -> {have / 1e6:.1f}M target tokens")
            if have >= need:
                break
        if have >= need:
            break
    pool.close()
    pool.join()
    if have < need:
        raise ValueError(f"Only {have} target tokens available, need {need}")

    all_ids = np.concatenate(ids_parts)[:need]
    all_mask = torch.from_numpy(np.concatenate(mask_parts)[:need].reshape(args.num_blocks, args.block_size))
    torch.save(all_mask, os.path.join(args.output_dir, "mask.pt"))

    flat = pa.array(all_ids, type=pa.int32())
    table_out = pa.table({
        "input_ids": pa.FixedSizeListArray.from_arrays(flat, args.block_size),
        "attention_mask": pa.FixedSizeListArray.from_arrays(pa.array(np.ones(need, dtype=np.int8)), args.block_size),
        "labels": pa.FixedSizeListArray.from_arrays(pa.array(all_ids.astype(np.int64)), args.block_size),
    })
    Dataset(table_out).save_to_disk(os.path.join(args.output_dir, "dataset"))

    stats = dict(totals)
    stats.update(
        target_tokens=int(need),
        target_masked_tokens=int(all_mask.sum()),
        target_masked_share=float(all_mask.float().mean()),
        src_masked_share=totals["src_masked_tokens"] / totals["src_tokens"],
        masked_byte_recall=totals["recalled_bytes"] / max(1, totals["masked_bytes"]),
        byte_inflation=totals["bytes_in_masked_targets"] / max(1, totals["masked_bytes"]),
        target_tokens_per_src_token=have / totals["src_tokens"],
    )
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(json.dumps(stats, indent=2))
