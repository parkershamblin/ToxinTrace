"""
ToxinTrace 4.8: gather every results.json into a CSV, LaTeX tables with the IF-Guide paper's numbers
alongside, the reduction factors with pass/fail against our success rule, and the DPO-sweep figure
(pgfplots, so it compiles with the report and needs no plotting library).

Run from src/:  python toxintrace/make_tables.py [--results_dir ~/toxintrace/results] [--out_dir <results>/tables]
"""
import os
import csv
import json
import argparse

# Paper (Coalson et al., NeurIPS 2025), Pythia-160M. Table 1: RTP + fluency; Table 4 (App. D.1): AttaQ, BOLD.
PAPER = {
    "None":           dict(rtp_emt=0.557, rtp_tp=0.560, tox_emt=0.764, tox_tp=0.801, non_emt=0.350, non_tp=0.319, ppl=25.84, acc=0.450, attaq_emt=0.458, attaq_tp=0.450, bold_emt=0.276, bold_tp=0.217),
    "DPO":            dict(rtp_emt=0.348, rtp_tp=0.330, tox_emt=0.517, tox_tp=0.525, non_emt=0.179, non_tp=0.136, ppl=26.47, acc=0.474, attaq_emt=0.262, attaq_tp=0.233, bold_emt=0.094, bold_tp=0.050),
    "RAD":            dict(rtp_emt=0.118, rtp_tp=0.094, tox_emt=0.202, tox_tp=0.176, non_emt=0.034, non_tp=0.011, ppl=None,  acc=0.457, attaq_emt=0.069, attaq_tp=0.029, bold_emt=0.030, bold_tp=0.013),
    "IF-Guide":       dict(rtp_emt=0.101, rtp_tp=0.054, tox_emt=0.136, tox_tp=0.085, non_emt=0.067, non_tp=0.024, ppl=26.77, acc=0.433, attaq_emt=0.122, attaq_tp=0.066, bold_emt=0.114, bold_tp=0.076),
    "IF-Guide + DPO": dict(rtp_emt=0.077, rtp_tp=0.035, tox_emt=0.101, tox_tp=0.053, non_emt=0.053, non_tp=0.017, ppl=27.27, acc=0.408, attaq_emt=0.097, attaq_tp=0.053, bold_emt=0.106, bold_tp=0.073),
    "IF-Guide + RAD": dict(rtp_emt=0.031, rtp_tp=0.017, tox_emt=0.047, tox_tp=0.030, non_emt=0.015, non_tp=0.004, ppl=None,  acc=0.438, attaq_emt=0.039, attaq_tp=0.012, bold_emt=0.030, bold_tp=0.018),
}
# Paper Table 6 (App. D.3): implicit toxicity probability, ToxiGen-RoBERTa, Pythia-160M (RTP full, AttaQ, BOLD)
for _k, _v in {
    "None": (0.550, 0.539, 0.186),
    "DPO": (0.334, 0.381, 0.048),
    "RAD": (0.249, 0.278, 0.043),
    "IF-Guide": (0.195, 0.291, 0.062),
    "IF-Guide + DPO": (0.187, 0.277, 0.067),
    "IF-Guide + RAD": (0.149, 0.228, 0.024),
}.items():
    PAPER[_k].update(zip(("rtp_itp", "attaq_itp", "bold_itp"), _v))
# Our result folders for each paper row; DPO rows are the sweep picks (ppl increase closest to the paper's +2.4%)
OURS = {
    "None": "baseline", "DPO": "dpo300_baseline", "RAD": "baseline_rad",
    "IF-Guide": "ifguide", "IF-Guide + DPO": "dpo400_ifguide", "IF-Guide + RAD": "ifguide_rad",
}
# Extra rows with no paper counterpart: DPO with Lee et al.'s early stopping, which overtrained
EXTRA = {"DPO, early stopping (overtrained)": "dpo_baseline", "IF-Guide + DPO, early stopping (overtrained)": "dpo_ifguide"}
RAD_ROWS = {"RAD", "IF-Guide + RAD"}  # RAD changes the output distribution, so no perplexity (as in the paper)
METRICS = ["rtp_emt", "rtp_tp", "tox_emt", "tox_tp", "non_emt", "non_tp", "rtp_itp", "ppl", "acc",
           "attaq_emt", "attaq_tp", "attaq_itp", "bold_emt", "bold_tp", "bold_itp"]


def load(path, *keys):
    try:
        with open(path) as f:
            d = json.load(f)
        for k in keys:
            d = d[k]
        return d
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def ours(results_dir, name, rad):
    r = lambda D, f: os.path.join(results_dir, D, "pythia-160m", name, f)
    m = {
        "rtp_emt": load(r("RTP", "results.json"), "all_prompts", "maximum_toxicity", "mean"),
        "rtp_tp": load(r("RTP", "results.json"), "all_prompts", "toxicity_probability"),
        "tox_emt": load(r("RTP", "results.json"), "toxic_prompts", "maximum_toxicity", "mean"),
        "tox_tp": load(r("RTP", "results.json"), "toxic_prompts", "toxicity_probability"),
        "non_emt": load(r("RTP", "results.json"), "nontoxic_prompts", "maximum_toxicity", "mean"),
        "non_tp": load(r("RTP", "results.json"), "nontoxic_prompts", "toxicity_probability"),
        "rtp_itp": load(r("RTP", "implicit_results.json"), "all_prompts", "implicit_toxicity_probability"),
        "ppl": None if rad else load(r("fluency", "results.json"), "openwebtext", "perplexity"),
        "acc": load(r("fluency", "results.json"), "lambada", "last_token_accuracy"),
    }
    for D in ["AttaQ", "BOLD"]:
        k = D.lower()
        m[f"{k}_emt"] = load(r(D, "results.json"), "all_prompts", "maximum_toxicity", "mean")
        m[f"{k}_tp"] = load(r(D, "results.json"), "all_prompts", "toxicity_probability")
        m[f"{k}_itp"] = load(r(D, "implicit_results.json"), "all_prompts", "implicit_toxicity_probability")
    return m


def cell(o, p, digits=3, with_paper=True):
    f = lambda v: "--" if v is None else f"{v:.{digits}f}"
    if not with_paper:
        return f(o)
    return "--" if o is None and p is None else f"{f(o)} ({f(p)})"


def latex_table(rows, cols, caption, label, with_paper=True):
    # Wide tables are scaled to the text width (needs \usepackage{graphicx})
    head = " & ".join(["Defense"] + [c[1] for c in cols])
    wide = len(cols) > 5
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}"]
    lines += [r"\resizebox{\linewidth}{!}{%"] if wide else []
    lines += [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", head + r" \\", r"\midrule"]
    for name, o, p in rows:
        vals = [cell(o.get(k), p.get(k) if p else None, 2 if k == "ppl" else 3, with_paper) for k, _ in cols]
        lines.append(" & ".join([name] + vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"] + (["}"] if wide else [])
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def sweep_rows(sweep_dir):
    rows = []
    fl = os.path.join(sweep_dir, "fluency", "pythia-160m")
    if not os.path.isdir(fl):
        return rows
    for tag in sorted(os.listdir(fl)):
        model, step = tag.rsplit("_", 1)
        p = load(os.path.join(fl, tag, "results.json"), "openwebtext", "perplexity")
        a = load(os.path.join(fl, tag, "results.json"), "lambada", "last_token_accuracy")
        rtp = os.path.join(sweep_dir, "RTP", "pythia-160m", tag, "results.json")
        e = load(rtp, "all_prompts", "maximum_toxicity", "mean")
        t = load(rtp, "all_prompts", "toxicity_probability")
        if None not in (p, a, e, t) and int(step) <= 1000:
            rows.append(dict(model=model, step=int(step), ppl=p, acc=a, emt=e, tp=t))
    for r in rows:
        base = next(x["ppl"] for x in rows if x["model"] == r["model"] and x["step"] == 0)
        r["ppl_change"] = 100 * (r["ppl"] / base - 1)
    return sorted(rows, key=lambda r: (r["model"], r["step"]))


FIG = r"""%% DPO checkpoint sweep: RTP toxicity (1,000-prompt subset) vs. OpenWebText perplexity increase.
%% Needs \usepackage{pgfplots} and \pgfplotsset{compat=1.18}. Data: dpo_sweep.csv (same folder).
\begin{tikzpicture}
\definecolor{s1}{HTML}{2A78D6}\definecolor{s2}{HTML}{EB6834}\definecolor{ink2}{HTML}{52514E}
\begin{axis}[
  width=\linewidth, height=0.62\linewidth,
  xlabel={OpenWebText perplexity increase over the starting model (\%)},
  ylabel={RTP expected max.\ toxicity},
  xmin=-0.5, xmax=15, ymin=0, ymax=0.6,
  axis lines=left, tick style={ink2}, tick label style={font=\footnotesize, text=ink2},
  label style={font=\small}, grid=major, major grid style={ink2!15, line width=0.3pt},
  legend style={draw=none, fill=none, font=\footnotesize, at={(0.98,0.98)}, anchor=north east},
  legend cell align=left,
]
\addplot[s1, line width=1pt, mark=*, mark size=1.6pt] table[x=ppl_change, y=emt, col sep=comma,
  discard if not={model}{baseline}] {DATA};
\addlegendentry{DPO on base}
\addplot[s2, line width=1pt, mark=square*, mark size=1.5pt] table[x=ppl_change, y=emt, col sep=comma,
  discard if not={model}{ifguide}] {DATA};
\addlegendentry{DPO on IF-Guide}
\addplot[forget plot, only marks, mark=*, mark size=3.2pt, draw=s1, fill=white, line width=1pt] coordinates {(PICKB)};
\addplot[forget plot, only marks, mark=square*, mark size=3pt, draw=s2, fill=white, line width=1pt] coordinates {(PICKI)};
\addplot[forget plot, only marks, mark=diamond, mark size=3.2pt, draw=ink2, line width=0.8pt] coordinates {(2.44,0.348) (1.87,0.077)};
\addlegendimage{only marks, mark=*, mark size=3.2pt, draw=ink2, fill=white, line width=1pt}
\addlegendentry{picked checkpoint}
\addlegendimage{only marks, mark=diamond, mark size=3.2pt, draw=ink2, line width=0.8pt}
\addlegendentry{paper (full RTP)}
\node[font=\scriptsize, text=ink2, anchor=west] at (axis cs:LABB) {step SB};
\node[font=\scriptsize, text=ink2, anchor=west] at (axis cs:LABI) {step SI};
\end{axis}
\end{tikzpicture}
"""
# `discard if not` filter for pgfplots tables (defined once, before the figure)
FIG_PREAMBLE = r"""\pgfplotsset{discard if not/.style 2 args={x filter/.append code={
  \edef\tempa{\thisrow{#1}}\edef\tempb{#2}\ifx\tempa\tempb\else\def\pgfmathresult{inf}\fi}}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.expanduser("~/toxintrace/results"))
    ap.add_argument("--sweep_dir", default=os.path.expanduser("~/toxintrace/results_sweep"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pick_baseline", type=int, default=300)
    ap.add_argument("--pick_ifguide", type=int, default=400)
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.results_dir, "tables")
    os.makedirs(out, exist_ok=True)

    table = [(name, ours(args.results_dir, res, name in RAD_ROWS), PAPER[name]) for name, res in OURS.items()]
    extra = [(name, ours(args.results_dir, res, False), None) for name, res in EXTRA.items()]

    # 1) long CSV
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["defense", "result_name", "metric", "ours", "paper"])
        for (name, o, p), res in zip(table + extra, list(OURS.values()) + list(EXTRA.values())):
            for k in METRICS:
                w.writerow([name, res, k, "" if o.get(k) is None else f"{o[k]:.6g}",
                            "" if not p or p.get(k) is None else p[k]])

    # 2) LaTeX tables: ours (paper)
    rtp_cols = [("rtp_emt", "EMT"), ("rtp_tp", "TP"), ("tox_emt", "Toxic EMT"), ("tox_tp", "Toxic TP"),
                ("non_emt", "Non-tox.\\ EMT"), ("non_tp", "Non-tox.\\ TP"), ("ppl", "PPL"), ("acc", "LAMBADA")]
    with open(os.path.join(out, "table_rtp.tex"), "w") as f:
        f.write(latex_table(table, rtp_cols,
            "Pythia-160M on RealToxicityPrompts and fluency: ours (paper, Table 1). EMT = expected maximum "
            "toxicity, TP = toxicity probability (Detoxify), PPL = OpenWebText perplexity (none under RAD), "
            "LAMBADA = last-token accuracy. DPO rows use the sweep checkpoint whose perplexity increase is "
            "closest to the paper's DPO.", "tab:rtp"))
    ab_cols = [("attaq_emt", "AttaQ EMT"), ("attaq_tp", "AttaQ TP"), ("bold_emt", "BOLD EMT"), ("bold_tp", "BOLD TP")]
    with open(os.path.join(out, "table_attaq_bold.tex"), "w") as f:
        f.write(latex_table(table, ab_cols, "Pythia-160M on AttaQ and BOLD, all prompts: ours (paper, Table 4).", "tab:attaq-bold"))
    imp_cols = [("rtp_itp", "RTP"), ("attaq_itp", "AttaQ"), ("bold_itp", "BOLD")]
    with open(os.path.join(out, "table_implicit.tex"), "w") as f:
        f.write(latex_table(table, imp_cols,
            "Implicit toxicity probability (ToxiGen-RoBERTa), all prompts: ours (paper, Table 6).", "tab:implicit"))

    # 3) reduction factors and success rule: within 25% of the paper's factors, perplexity up < 10%
    o, p = table[0][1], table[0][2]
    oi, pi = table[3][1], table[3][2]
    lines = []
    ok_all = True
    for k, label in [("rtp_emt", "EMT"), ("rtp_tp", "TP")]:
        f_ours, f_paper = o[k] / oi[k], p[k] / pi[k]
        need = 0.75 * f_paper
        ok = f_ours >= need
        ok_all &= ok
        lines.append(f"RTP {label} reduction: ours {f_ours:.2f}x, paper {f_paper:.2f}x, need >= {need:.2f}x -> {'PASS' if ok else 'FAIL'}")
    d_ppl = 100 * (oi["ppl"] / o["ppl"] - 1)
    ok = d_ppl < 10
    ok_all &= ok
    lines.append(f"OWT perplexity increase: ours {d_ppl:+.1f}% (paper {100 * (pi['ppl'] / p['ppl'] - 1):+.1f}%), need < 10% -> {'PASS' if ok else 'FAIL'}")
    lines.append(f"Success rule overall: {'PASS' if ok_all else 'FAIL'}")
    with open(os.path.join(out, "success.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # 4) DPO sweep CSV + pgfplots figure
    rows = sweep_rows(args.sweep_dir)
    if rows:
        with open(os.path.join(out, "dpo_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "step", "ppl", "ppl_change", "acc", "emt", "tp"])
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
        pick = lambda m, s: next(r for r in rows if r["model"] == m and r["step"] == s)
        b, i = pick("baseline", args.pick_baseline), pick("ifguide", args.pick_ifguide)
        fig = (FIG.replace("DATA", "dpo_sweep.csv")
               .replace("PICKB", f"{b['ppl_change']:.3f},{b['emt']:.4f}")
               .replace("PICKI", f"{i['ppl_change']:.3f},{i['emt']:.4f}")
               .replace("LABB", f"{b['ppl_change'] + 0.4:.3f},{b['emt'] + 0.02:.4f}").replace("SB", str(args.pick_baseline))
               .replace("LABI", f"{i['ppl_change'] + 0.4:.3f},{i['emt'] + 0.03:.4f}").replace("SI", str(args.pick_ifguide)))
        with open(os.path.join(out, "fig_dpo_sweep.tex"), "w") as f:
            f.write(FIG_PREAMBLE + fig)
        print(f"sweep: {len(rows)} points")
    print(f"wrote {sorted(os.listdir(out))} to {out}")


if __name__ == "__main__":
    main()
