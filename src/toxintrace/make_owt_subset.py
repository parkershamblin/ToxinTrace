"""
ToxinTrace: build a compact copy of the authors' 1B-token OpenWebText subset.

The paper's training set is 488,282 chunks picked by index from a 4.4M-chunk tokenized dataset
(~33 GB download, ~120 GB once prepared). This keeps only the train, valid and test chunks, in
their original index order, and writes remapped index files (0..N-1). Toxic-token masks are
indexed by position in the training set, so masks built on the compact copy line up with the paper's.

Run from `src/`:
    python toxintrace/make_owt_subset.py --output_dir ~/toxintrace/data/owt_subset --index_dir ~/toxintrace/data/owt_subset_indices
then use `export IFGUIDE_OWT_PATH=~/toxintrace/data/owt_subset` and pass the remapped index files.
"""
import os
import argparse
import hashlib
import logging
import numpy as np

from datasets import load_dataset, load_from_disk

OWT_HUB_PATH = "ztcoalson/openwebtext-gptneox-2048-chunks"
SPLITS = ("train", "valid", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a compact copy of the 1B-token OpenWebText subset.")

    parser.add_argument("--output_dir", type=str, required=True, help="Where to `save_to_disk` the subset.")
    parser.add_argument("--index_dir", type=str, required=True, help="Where to write the remapped index files.")
    parser.add_argument("--num_proc", type=int, default=8, help="Processes for download/preparation and saving.")
    parser.add_argument("--num_checks", type=int, default=200, help="Random rows compared against the full dataset.")

    return parser.parse_args()


def fingerprint(dset, rows) -> str:
    """Short hash of a few rows' token ids, to compare copies built on different machines."""
    h = hashlib.sha256()
    for r in rows:
        h.update(np.asarray(dset[int(r)]["input_ids"], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args.output_dir = os.path.expanduser(args.output_dir)
    args.index_dir = os.path.expanduser(args.index_dir)
    os.makedirs(args.index_dir, exist_ok=True)

    # The authors' index files (paths relative to `src/`)
    indices = {s: np.load(f"../data/openwebtext/{s}_indices.npy") for s in SPLITS}
    all_indices = np.concatenate([indices[s] for s in SPLITS])
    overlap = len(all_indices) - len(np.unique(all_indices))
    logging.info(f"Index files: " + ", ".join(f"{s}={len(indices[s])}" for s in SPLITS) + f"; overlapping indices: {overlap}")

    if os.path.isdir(args.output_dir):
        logging.info(f"{args.output_dir} already exists, keeping it.")
        subset = load_from_disk(args.output_dir)
    else:
        full = load_dataset(OWT_HUB_PATH, split="train", num_proc=args.num_proc)
        logging.info(f"Full dataset: {len(full)} chunks, columns {full.column_names}")

        full.select(all_indices).save_to_disk(args.output_dir, num_proc=args.num_proc)
        subset = load_from_disk(args.output_dir)

        rng = np.random.default_rng(0)
        for k in rng.choice(len(all_indices), args.num_checks, replace=False):
            if subset[int(k)]["input_ids"] != full[int(all_indices[k])]["input_ids"]:
                raise RuntimeError(f"Row {k} of the subset does not match row {all_indices[k]} of the full dataset")
        logging.info(f"Spot check passed: {args.num_checks} random rows match the full dataset")

    if len(subset) != len(all_indices):
        raise RuntimeError(f"Subset has {len(subset)} rows, expected {len(all_indices)}")

    start = 0
    for s in SPLITS:
        n = len(indices[s])
        np.save(os.path.join(args.index_dir, f"{s}_indices.npy"), np.arange(start, start + n))
        logging.info(f"{s}_indices.npy: {start}..{start + n - 1}")
        start += n

    rows = [0, 1, len(indices["train"]) - 1, len(subset) // 2, len(subset) - 1]
    logging.info(f"Subset: {len(subset)} chunks ({len(subset) * 2048 / 1e9:.3f}B tokens); "
                 f"fingerprint of rows {rows}: {fingerprint(subset, rows)}")


if __name__ == "__main__":
    main()
