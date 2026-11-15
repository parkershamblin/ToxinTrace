"""
ToxinTrace (E2): a 400M-token GPT-NeoX-tokenized subset of a recent FineWeb dump, in the same format
as the OpenWebText chunks the paper trains on.

Documents are read in file order from one Common Crawl dump of HuggingFaceFW/fineweb (its parquet
shards are downloaded one by one, at a pinned dataset revision, until there are enough tokens), tokenized
with the Pythia tokenizer and concatenated with no separator into 2,048-token chunks, exactly like
`tokenize_openwebtext` in utils/datasets.py (the leftover tail is dropped). The first
`--num_train_chunks` chunks are the training split and the next `--num_valid_chunks` the validation
split. Columns and dtypes match ~/toxintrace/data/owt_subset (input_ids int32, attention_mask int8,
labels int64), so `export IFGUIDE_OWT_PATH=<output_dir>` with the written index files works with
train.py, finetune.py, fit_factors.py, compute_scores.py and build_toxic_token_mask.py unchanged.

Also writes docs.jsonl (one line per document used: FineWeb id, URL, first token offset, token
count) to trace top-ranked chunks back to their pages, and stats.json.

Run from `src/`:
    python toxintrace/make_fineweb_subset.py --output_dir ~/toxintrace/data/fineweb_subset \
        --index_dir ~/toxintrace/data/fineweb_subset_indices
"""
import os
import json
import time
import argparse
import hashlib
import logging

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

FINEWEB = "HuggingFaceFW/fineweb"
TOKENIZER = "EleutherAI/pythia-70m-deduped"    # registry 'pythia-tok'
BLOCK = 2048


def parse_args():
    p = argparse.ArgumentParser(description="Build a GPT-NeoX-tokenized FineWeb subset in OpenWebText-chunk format.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--index_dir", type=str, required=True)
    p.add_argument("--dump", type=str, default="CC-MAIN-2025-26", help="FineWeb config (Common Crawl dump).")
    p.add_argument("--num_train_chunks", type=int, default=195_313, help="195,313 x 2,048 = 400.0M tokens.")
    p.add_argument("--num_valid_chunks", type=int, default=2_048)
    p.add_argument("--batch_docs", type=int, default=2_000, help="Documents per tokenizer call.")
    p.add_argument("--cache_dir", type=str, default=None, help="Where parquet shards are downloaded (default: HF cache).")
    return p.parse_args()


def fingerprint(dset, rows) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(np.asarray(dset[int(r)]["input_ids"], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    out = os.path.expanduser(args.output_dir)
    idx_dir = os.path.expanduser(args.index_dir)
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; delete it or pick another --output_dir")
    os.makedirs(idx_dir, exist_ok=True)

    n_chunks = args.num_train_chunks + args.num_valid_chunks
    need = n_chunks * BLOCK
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    tok.model_max_length = int(1e12)   # we chunk ourselves; silence the length warning

    # Download the dump's parquet shards one at a time, in file order, with the hub's resumable
    # downloader (HF streaming kept stalling on read timeouts), and read only the shards needed.
    api = HfApi()
    revision = api.dataset_info(FINEWEB).sha
    files = sorted(s.rfilename for s in api.dataset_info(FINEWEB, revision=revision).siblings
                   if s.rfilename.startswith(f"data/{args.dump}/") and s.rfilename.endswith(".parquet"))
    used_files = []
    def stream_docs():
        for f in files:
            for attempt in range(10):
                try:
                    path = hf_hub_download(FINEWEB, f, repo_type="dataset", revision=revision, cache_dir=args.cache_dir)
                    break
                except Exception as e:     # network hiccup: the download resumes from the partial file
                    logging.warning(f"download of {f} failed ({e!r}); retry {attempt + 1}/10 in 30 s")
                    time.sleep(30)
            else:
                raise RuntimeError(f"could not download {f}")
            used_files.append(f)
            logging.info(f"reading {f}")
            for rb in pq.ParquetFile(path).iter_batches(batch_size=args.batch_docs, columns=["text", "id", "url"]):
                yield from rb.to_pylist()
    stream = stream_docs()
    buf = np.empty(need, dtype=np.int32)
    filled = 0
    docs_path = os.path.join(idx_dir, "docs.jsonl")
    n_docs = 0
    t0 = time.time()
    with open(docs_path, "w") as docs_f:
        batch = []
        def flush(batch):
            nonlocal filled, n_docs
            enc = tok([d["text"] for d in batch], add_special_tokens=False)["input_ids"]
            for d, ids in zip(batch, enc):
                if filled >= need:
                    return
                take = min(len(ids), need - filled)
                buf[filled:filled + take] = ids[:take]
                docs_f.write(json.dumps({"id": d["id"], "url": d["url"], "start": filled, "n_tokens": take}) + "\n")
                filled += take
                n_docs += 1
        for d in stream:
            batch.append(d)
            if len(batch) == args.batch_docs:
                flush(batch)
                batch = []
                logging.info(f"{filled / 1e6:.1f}M / {need / 1e6:.1f}M tokens, {n_docs} docs, {time.time() - t0:.0f} s")
                if filled >= need:
                    break
        if batch and filled < need:
            flush(batch)
    if filled < need:
        raise RuntimeError(f"Dump ran out at {filled} tokens, need {need}")

    chunks = buf.reshape(n_chunks, BLOCK)
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(Value("int64")),
    })
    ones = np.ones(BLOCK, dtype=np.int8)
    def gen():
        for row in chunks:
            yield {"input_ids": row, "attention_mask": ones, "labels": row.astype(np.int64)}
    ds = Dataset.from_generator(gen, features=features)
    ds.save_to_disk(out, num_proc=8)
    ds = load_from_disk(out)

    # check: chunks decode back to the concatenated documents
    if ds[0]["input_ids"] != chunks[0].tolist() or ds[len(ds) - 1]["labels"] != chunks[-1].tolist():
        raise RuntimeError("saved rows do not match the token buffer")

    np.save(os.path.join(idx_dir, "train_indices.npy"), np.arange(args.num_train_chunks))
    np.save(os.path.join(idx_dir, "valid_indices.npy"), np.arange(args.num_train_chunks, n_chunks))
    rows = [0, 1, args.num_train_chunks - 1, len(ds) // 2, len(ds) - 1]
    stats = {
        "dump": args.dump, "revision": revision, "parquet_files": used_files, "tokenizer": TOKENIZER, "block_size": BLOCK,
        "train_chunks": args.num_train_chunks, "valid_chunks": args.num_valid_chunks,
        "train_tokens": args.num_train_chunks * BLOCK, "documents": n_docs,
        "mean_tokens_per_doc": filled / n_docs, "seconds": round(time.time() - t0),
        "fingerprint_rows": rows, "fingerprint": fingerprint(ds, rows),
    }
    json.dump(stats, open(os.path.join(idx_dir, "stats.json"), "w"), indent=2)
    logging.info(json.dumps(stats))
    logging.info(f"Sample of chunk 0: {tok.decode(chunks[0][:120])!r}")


if __name__ == "__main__":
    main()
    # The abandoned streaming iterator's worker threads crash interpreter shutdown
    # ("PyGILState_Release ... finalizing"); everything is saved by now, so exit directly.
    logging.shutdown()
    os._exit(0)
