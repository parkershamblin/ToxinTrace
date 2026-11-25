"""
ToxinTrace (E2): how toxic is the FineWeb 2025 corpus compared with OpenWebText?

Draws random training chunks from both tokenized corpora, splits each 2,048-token chunk into 256-token
passages (Detoxify reads at most 512 of its own tokens, so scoring whole chunks would only see their
start), and scores every passage with the same Detoxify model as the toxicity evaluation. Reports the
share of passages and of chunks above 0.5 and the mean score, with 95% bootstrap intervals over chunks.

Run from src/:  python toxintrace/compare_toxicity.py [--num_chunks 10000] [--out_dir <results>/e2_toxicity]
"""
import argparse
import json
import os

import numpy as np
import torch
from datasets import load_from_disk
from detoxify import Detoxify
from transformers import AutoTokenizer
from tqdm import tqdm

ROOT = os.path.expanduser(os.environ.get("TT_ROOT", "~/toxintrace"))
CORPORA = {"openwebtext": ("owt_subset", "owt_subset_indices"), "fineweb_2025": ("fineweb_subset", "fineweb_subset_indices")}
ATTRS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_chunks", type=int, default=10_000)
    ap.add_argument("--passage", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "e2_toxicity"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    clf = Detoxify("unbiased", device="cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for name, (data, idx) in CORPORA.items():
        rng = np.random.default_rng(args.seed)
        ds = load_from_disk(os.path.join(ROOT, "data", data))
        train = np.load(os.path.join(ROOT, "data", idx, "train_indices.npy"))
        pick = np.sort(rng.choice(train, args.num_chunks, replace=False))
        ids = np.asarray(ds.select(pick).with_format("numpy")["input_ids"], dtype=np.int64)
        per = ids.shape[1] // args.passage
        passages = tok.batch_decode(ids[:, : per * args.passage].reshape(-1, args.passage).tolist())
        wp = [len(x) for x in clf.tokenizer(passages[:2000], truncation=False)["input_ids"]]
        print(f"{name}: {len(passages)} passages; longer than 512 Detoxify tokens: {np.mean(np.array(wp) > 512):.2%} of 2,000", flush=True)
        scores = {a: [] for a in ATTRS}
        for i in tqdm(range(0, len(passages), args.batch_size), desc=name, mininterval=20):
            p = clf.predict(passages[i:i + args.batch_size])
            for a in ATTRS:
                scores[a] += p[a]
        S = {a: np.array(scores[a]).reshape(args.num_chunks, per) for a in ATTRS}
        np.save(os.path.join(args.out_dir, f"{name}_toxicity.npy"), S["toxicity"].astype(np.float32))
        t = S["toxicity"]
        stats = {
            "passage_mean": lambda i: t[i].mean(), "passages_above_0.5": lambda i: (t[i] > 0.5).mean(),
            "passages_above_0.25": lambda i: (t[i] > 0.25).mean(), "passages_above_0.9": lambda i: (t[i] > 0.9).mean(),
            "chunks_above_0.5": lambda i: (t[i].max(1) > 0.5).mean(),
        }
        boot = [rng.integers(0, len(t), len(t)) for _ in range(1000)]
        everything = np.arange(len(t))
        r = {}
        for k, f in stats.items():
            b = np.array([f(i) for i in boot])
            r[k] = {"value": float(f(everything)), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}
        r["attributes_passages_above_0.5"] = {a: float((S[a] > 0.5).mean()) for a in ATTRS}
        out[name] = r
        print(json.dumps(r, indent=1), flush=True)
    a, b = out["openwebtext"], out["fineweb_2025"]
    out["ratio_owt_over_fineweb"] = {k: a[k]["value"] / b[k]["value"] for k in a if "attributes" not in k}
    json.dump(out, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print("RATIO", json.dumps(out["ratio_owt_over_fineweb"]))


if __name__ == "__main__":
    main()
