"""
ToxinTrace: one table of every metric for a set of runs of one model, with the reduction factor of a
reference run over a compared run (E1, E2). Reads only saved results; no GPU.

Run from src/:
  python toxintrace/e2_summary.py                                  # E2: stock Pythia-410M, control, mask
  python toxintrace/e2_summary.py --model gpt2-pretrained --runs e1_control e1_transfer \
      --ref e1_control --test e1_transfer --out_csv <results>/e1/summary.csv
"""
import argparse
import csv
import json
import os

import numpy as np

ROOT = os.path.expanduser(os.environ.get("TT_ROOT", "~/toxintrace"))
ATTR = ["obscene", "threat", "insult", "identity_attack", "sexual_explicit"]


def get(path, *keys):
    try:
        d = json.load(open(path))
        for k in keys:
            d = d[k]
        return d
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def metrics(results, strip, model, run):
    r = lambda D, f="results.json", root=results: os.path.join(root, D, model, run, f)
    m = {}
    for part, tag in [("all_prompts", ""), ("toxic_prompts", " (toxic prompts)"), ("nontoxic_prompts", " (non-toxic prompts)")]:
        m[f"RTP EMT{tag}"] = get(r("RTP"), part, "maximum_toxicity", "mean")
        m[f"RTP TP{tag}"] = get(r("RTP"), part, "toxicity_probability")
    for D in ["AttaQ", "BOLD"]:
        m[f"{D} EMT"] = get(r(D), "all_prompts", "maximum_toxicity", "mean")
        m[f"{D} TP"] = get(r(D), "all_prompts", "toxicity_probability")
    m["BOLD EMT (no trailing space)"] = get(r("BOLD", root=strip), "all_prompts", "maximum_toxicity", "mean")
    m["BOLD TP (no trailing space)"] = get(r("BOLD", root=strip), "all_prompts", "toxicity_probability")
    for D in ["RTP", "AttaQ", "BOLD"]:
        m[f"Implicit TP, {D}"] = get(r(D, "implicit_results.json"), "all_prompts", "implicit_toxicity_probability")
    for a in ATTR:
        m[f"RTP {a} probability"] = get(r("RTP"), "all_prompts", f"{a}_probability")
    o = get(r("RTP", "outputs.json"))
    m["RTP completions above 0.5"] = None if o is None else float(np.mean([c["toxicity"] > 0.5 for p in o for c in p["outputs"]]))
    f = lambda name: os.path.join(results, "fluency", model, name, "results.json")
    m["OWT perplexity"] = get(f(run), "openwebtext", "perplexity")
    m["LAMBADA accuracy"] = get(f(run), "lambada", "last_token_accuracy")
    m["FineWeb perplexity (held-out chunks)"] = get(f(run + "_fineweb"), "openwebtext", "perplexity")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m-pretrained")
    ap.add_argument("--runs", nargs="+", default=["e2_base", "e2_control", "e2_ifguide"])
    ap.add_argument("--ref", default="e2_control")
    ap.add_argument("--test", default="e2_ifguide")
    ap.add_argument("--results_dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--strip_dir", default=os.path.join(ROOT, "results_boldstrip"))
    ap.add_argument("--out_csv", default=os.path.join(ROOT, "results", "e2", "summary.csv"))
    args = ap.parse_args()
    M = {run: metrics(args.results_dir, args.strip_dir, args.model, run) for run in args.runs}
    rows = []
    for k in M[args.runs[0]]:
        vals = [M[run][k] for run in args.runs]
        a, b = M[args.ref][k], M[args.test][k]
        if a is None or b is None:
            change = ""
        elif "perplexity" in k:
            change = f"{100 * (b / a - 1):+.1f}%"
        elif "accuracy" in k:
            change = f"{b - a:+.3f}"
        else:
            change = f"{a / b:.2f}x" if b > 0 else "inf"
        rows.append([k] + ["" if v is None else f"{v:.4f}" for v in vals] + [change])
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = ["metric"] + args.runs + [f"{args.ref} / {args.test}"]
    with open(args.out_csv, "w", newline="") as fh:
        csv.writer(fh).writerows([header] + rows)
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
