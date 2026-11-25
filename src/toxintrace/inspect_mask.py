"""
ToxinTrace (E2): what did the 2% mask select in FineWeb 2025, next to the OpenWebText mask?

For each corpus: how many chunks are flagged and where, how toxic the most-flagged chunks are (Detoxify on
256-token passages), and how well flags line up with toxic passages on the 10,000 random chunks scored by
compare_toxicity.py. For FineWeb also: the most-flagged documents and sites, with their URLs.

Run from src/ after build_toxic_token_mask.py and compare_toxicity.py:
  python toxintrace/inspect_mask.py [--top_chunks 1000] [--out_dir <results>/e2_mask]
"""
import argparse
import collections
import json
import os
from urllib.parse import urlparse

import numpy as np
import torch
from datasets import load_from_disk
from detoxify import Detoxify
from transformers import AutoTokenizer

ROOT = os.path.expanduser(os.environ.get("TT_ROOT", "~/toxintrace"))
MASKS = "../data/toxic_token_masks"
CORPORA = {
    "openwebtext": ("owt_subset", f"{MASKS}/ekfac_RTP_pythia-160m/mask_toks=20.0m_p=0.99_w=1.pt"),
    "fineweb_2025": ("fineweb_subset", f"{MASKS}/ekfac_RTP_pythia-160m_fineweb/mask_toks=8.0m_p=0.99_w=1.pt"),
}
P = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_chunks", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "e2_mask"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    clf = Detoxify("unbiased", device="cuda" if torch.cuda.is_available() else "cpu")
    out = {}

    for name, (data, mask_path) in CORPORA.items():
        ds = load_from_disk(os.path.join(ROOT, "data", data)).with_format("numpy")
        train = np.load(os.path.join(ROOT, "data", f"{data}_indices", "train_indices.npy"))
        mask = torch.load(mask_path, map_location="cpu", weights_only=True, mmap=True).numpy()
        assert mask.shape == (len(train), 2048), mask.shape
        per_row = mask.sum(1)
        r = {
            "chunks": int(len(mask)), "flagged_tokens": int(per_row.sum()), "flagged_share": float(per_row.sum() / mask.size),
            "chunks_with_a_flag": int((per_row > 0).sum()), "median_flags_in_flagged_chunks": float(np.median(per_row[per_row > 0])),
            "max_flags": int(per_row.max()), "share_of_flags_in_first_64_positions": float(mask[:, :64].sum() / per_row.sum()),
        }
        # most-flagged chunks: how toxic are they?
        top = np.argsort(-per_row, kind="stable")[: args.top_chunks]
        ids = np.stack([ds[int(train[i])]["input_ids"] for i in top]).astype(np.int64)
        texts = tok.batch_decode(ids.reshape(-1, P).tolist())
        t = []
        for i in range(0, len(texts), args.batch_size):
            t += clf.predict(texts[i:i + args.batch_size])["toxicity"]
        t = np.array(t).reshape(args.top_chunks, -1)
        r["top_chunks"] = {"n": args.top_chunks, "passages_above_0.5": float((t > 0.5).mean()), "chunks_above_0.5": float((t.max(1) > 0.5).mean()), "mean": float(t.mean())}
        # flags against toxicity on the random chunks of compare_toxicity.py (same seed, same draw)
        tox = np.load(os.path.join(ROOT, "results", "e2_toxicity", f"{name}_toxicity.npy"))
        pick = np.sort(np.random.default_rng(args.seed).choice(train, len(tox), replace=False))
        row_of = {int(c): i for i, c in enumerate(train)}
        rows = np.array([row_of[int(c)] for c in pick])
        f = mask[rows].reshape(len(rows), -1, P).sum(2)          # flagged tokens per passage
        tp = tox > 0.5
        r["random_chunks"] = {
            "n": int(len(rows)), "flagged_share": float(f.sum() / (f.size * P)),
            "flagged_share_in_toxic_passages": float(f[tp].mean() / P), "flagged_share_in_other_passages": float(f[~tp].mean() / P),
            "share_of_flags_in_toxic_passages": float(f[tp].sum() / f.sum()), "toxic_passages_with_a_flag": float((f[tp] > 0).mean()),
            "other_passages_with_a_flag": float((f[~tp] > 0).mean()), "share_of_flags_in_passages_below_0.1": float(f[tox < 0.1].sum() / f.sum()),
        }
        out[name] = r
        print(name, json.dumps(r, indent=1), flush=True)

        # 12 most-flagged chunks, flagged tokens in [[ ]]
        with open(os.path.join(args.out_dir, f"{name}_top_chunks.txt"), "w") as fh:
            for k, i in enumerate(top[:12]):
                m = mask[i]
                a = int(np.argmax(np.convolve(m.astype(int), np.ones(160, int), "valid")))
                words = [("[[%s]]" if m[j] else "%s") % tok.decode([int(ids[k, j])]) for j in range(a, a + 160)]
                fh.write(f"--- rank {k}: row {i}, {int(per_row[i])} flagged, max passage toxicity {t[k].max():.2f}\n{''.join(words)}\n\n")

        if name == "fineweb_2025":
            starts, n_tok, urls = [], [], []
            with open(os.path.join(ROOT, "data", "fineweb_subset_indices", "docs.jsonl")) as fh:
                for line in fh:
                    d = json.loads(line)
                    if d["start"] < mask.size:
                        starts.append(d["start"]); n_tok.append(d["n_tokens"]); urls.append(d["url"])
            starts = np.array(starts)
            assert np.array_equal(train, np.arange(len(train))), "documents map to rows only if rows are in order"
            flags = np.add.reduceat(mask.ravel(), starts, dtype=np.int64)
            n_tok = np.array(n_tok)
            # check the document map: documents are back to back and end where the next one starts (the corpus has no end-of-text tokens)
            cont = starts[1:] == starts[:-1] + n_tok[:-1]
            r["documents_contiguous"] = bool(starts[0] == 0 and cont.all())
            assert r["documents_contiguous"], int((~cont).sum())
            hosts = np.array([urlparse(u).netloc for u in urls])
            hf, ht, hd = collections.Counter(), collections.Counter(), collections.Counter()
            for h, a, b in zip(hosts, flags, n_tok):
                hf[h] += int(a); ht[h] += int(b); hd[h] += 1
            order = np.argsort(-flags)
            total = flags.sum()
            r["documents"] = {
                "with_a_flag": int((flags > 0).sum()), "total": int(len(flags)),
                "share_of_flags_in_top_1pct_documents": float(flags[order[: len(flags) // 100]].sum() / total),
                "hosts_with_a_flag": int(sum(1 for v in hf.values() if v)), "share_of_flags_in_top_100_hosts": float(sum(v for _, v in hf.most_common(100)) / total),
            }
            r["top_hosts"] = [dict(host=h, flags=v, tokens=ht[h], docs=hd[h]) for h, v in hf.most_common(25)]
            r["top_documents"] = [dict(url=urls[i], flags=int(flags[i]), tokens=int(n_tok[i])) for i in order[:40]]
            print(json.dumps({k: r[k] for k in ("documents_contiguous", "documents", "top_hosts", "top_documents")}, indent=1), flush=True)
        del mask

    json.dump(out, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
