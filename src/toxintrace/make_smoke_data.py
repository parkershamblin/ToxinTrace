"""
ToxinTrace: build a tiny OpenWebText subset plus index files for smoke-testing the full pipeline.
Streams only the first rows of the authors' tokenized OpenWebText, so nothing large is downloaded.

Run from `src/`:
    python toxintrace/make_smoke_data.py --output_dir ~/toxintrace/data/owt_smoke --index_dir ~/toxintrace/data/smoke_indices
then point the pipeline at it with `export IFGUIDE_OWT_PATH=~/toxintrace/data/owt_smoke`.
"""
import os
import argparse
import logging
import numpy as np

from datasets import Dataset, load_dataset

OWT_HUB_PATH = "ztcoalson/openwebtext-gptneox-2048-chunks"


def parse_args():
    parser = argparse.ArgumentParser(description="Build a tiny OpenWebText subset for smoke tests.")

    parser.add_argument("--output_dir", type=str, required=True, help="Where to `save_to_disk` the subset.")
    parser.add_argument("--index_dir", type=str, required=True, help="Where to write the smoke index files.")
    parser.add_argument("--num_train", type=int, default=1600, help="Training chunks.")
    parser.add_argument("--num_valid", type=int, default=200, help="Validation chunks.")
    parser.add_argument("--num_test", type=int, default=200, help="Test chunks (perplexity).")
    parser.add_argument("--num_queries", type=int, default=200, help="RTP influence queries per class (toxic / non-toxic).")
    parser.add_argument("--num_lambada", type=int, default=20, help="LAMBADA examples for the fluency eval.")

    return parser.parse_args()


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

    # OpenWebText chunks: stream the first rows only
    total = args.num_train + args.num_valid + args.num_test
    if os.path.isdir(args.output_dir):
        logging.info(f"{args.output_dir} already exists, keeping it.")
    else:
        stream = load_dataset(OWT_HUB_PATH, split="train", streaming=True)
        dset = Dataset.from_list(list(stream.take(total)))
        dset.save_to_disk(args.output_dir)
        logging.info(f"Saved {len(dset)} chunks with columns {dset.column_names} to {args.output_dir}")

    splits = {
        "owt_train": np.arange(0, args.num_train),
        "owt_valid": np.arange(args.num_train, args.num_train + args.num_valid),
        "owt_test": np.arange(args.num_train + args.num_valid, total),
        # Subsets of the authors' index files (paths relative to `src/`)
        "rtp_query_toxic": np.load("../data/RTP/query_indices/toxic_indices.npy")[:args.num_queries],
        "rtp_query_nontoxic": np.load("../data/RTP/query_indices/nontoxic_indices.npy")[:args.num_queries],
        "lambada_test": np.load("../data/lambada/test_indices.npy")[:args.num_lambada],
    }
    for name, indices in splits.items():
        np.save(os.path.join(args.index_dir, f"{name}.npy"), indices)
        logging.info(f"{name}: {len(indices)} indices")


if __name__ == "__main__":
    main()
    # Streaming leaves fsspec/aiohttp threads behind that can abort the interpreter at shutdown
    # ("PyGILState_Release ... finalizing"); everything is saved by now, so exit directly.
    logging.shutdown()
    os._exit(0)
