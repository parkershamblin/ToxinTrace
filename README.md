# ToxinTrace: reproducing and extending IF-Guide

Course project for CAI 5307 (Natural Language Processing), University of South Florida, Fall 2026.
Steven Abdalla, Pritkumar Pagda, Parker Shamblin.

We reproduce the Pythia-160M results of [IF-Guide: Influence Function-Guided Detoxification of LLMs](https://arxiv.org/abs/2506.01790)
(Coalson, Bae, Carlini and Hong, NeurIPS 2025) on one RTX 4090. We then test whether a toxic-token mask
still works after it is moved to a model with a different tokenizer (E1), and whether the same 2020 queries
still find the tokens to penalize in a 2025 web corpus (E2, fine-tuning Pythia-410M).

This repository is the authors' code ([ztcoalson/IF-Guide](https://github.com/ztcoalson/IF-Guide)) plus our own scripts.
**Everything we wrote is in [`src/toxintrace/`](src/toxintrace)**; our changes to the authors' files are listed below.
The authors' README is kept as [`README_IFGUIDE.md`](README_IFGUIDE.md).

## Quick check (about 10 minutes)

This runs every stage of the pipeline on tiny models: data, training, EK-FAC, influence scores, the
toxic-token mask, IF-Guide training, fine-tuning, all evaluations, and RAD. It stops at the first failure.

```bash
git clone https://github.com/parkershamblin/ToxinTrace.git && cd ToxinTrace
python3.10 -m venv ~/venvs/ifguide && source ~/venvs/ifguide/bin/activate
pip install --upgrade pip wheel
pip install -r requirements-toxintrace.txt
cd src && bash toxintrace/smoke_test.sh
```

Needs Linux (or WSL2 Ubuntu 22.04), Python 3.10 and an NVIDIA GPU. The first run downloads a few GB of
models and data; with those cached, all 13 stages took about 8 minutes on an RTX 4090. Logs go to
`~/toxintrace/smoke/logs`. If it fails at stage N, fix the cause and resume with `bash toxintrace/smoke_test.sh N`.

## Full experiments

Run the numbered scripts in order from `src/`. Each one prints its commands; set `DRY_RUN=1` to only print them.

| Script in `toxintrace/experiments/` | What it does | Time on an RTX 4090 |
|---|---|---|
| `00_data.sh` | The paper's 1B-token OpenWebText subset, compacted (33 GB download, 150 GB temporary disk) | 1 h |
| `01_pretrain_base.sh` | R1: base Pythia-160M from scratch | 15 h |
| `02_influence_mask.sh` | R1: EK-FAC factors, differential token scores, 2% toxic-token mask | 15 h |
| `03_pretrain_ifguide.sh` | R1: Pythia-160M from scratch with the IF-Guide loss | 15-20 h |
| `04_dpo_sweep.sh` | DPO on both models, checkpoint sweep and scoring | 3-4 h |
| `05_evaluate.sh` | R1 + R2: RTP, AttaQ, BOLD, implicit toxicity, fluency for 6 systems, then the tables | 16 h (RAD is 6 h per model) |
| `06_bold_trailing_space.sh` | R2: BOLD without the trailing space of its prompts | 0.5 h |
| `07_e1_gpt2_transfer.sh` | E1: mask moved to GPT-2's tokenizer, two fine-tunes, evaluation | 6 h |
| `08_e2_fineweb.sh` | E2: FineWeb 2025 corpus and its toxicity rate against OpenWebText, EK-FAC scores and 2% mask, Pythia-410M fine-tunes without and with the mask, evaluation of those and of stock Pythia-410M | 18 h |
| `09_analysis.sh` | Bootstrap intervals, severity thresholds, BOLD first-word analysis, mask positions (CPU) | 5 min |

```bash
cd src
bash toxintrace/experiments/00_data.sh
bash toxintrace/experiments/01_pretrain_base.sh      # ... and so on
```

Settings are in [`experiments/env.sh`](src/toxintrace/experiments/env.sh): `TT_ROOT` (data, models and results; default `~/toxintrace`) and
`TT_VENV` (default `~/venvs/ifguide`). Run long jobs inside `tmux`.

**DPO preference pairs (manual download).** Get `toxicity_pairwise.zip` from the Google Drive folder linked in
[ajyl/dpo_toxic](https://github.com/ajyl/dpo_toxic) (Lee et al., 2024) and unzip it so that the six `.jsonl` files are in
`$TT_ROOT/data/dpo_toxic/toxicity_pairwise/`. Needed only for `04_dpo_sweep.sh` and the DPO rows of `05_evaluate.sh`.

**RAD reward model.** The original download is gone; `smoke_test.sh` (stage 10) fetches the authors' copy from
`hk/rad_rms` on Hugging Face, or see [`README_IFGUIDE.md`](README_IFGUIDE.md).

Checkpoints, the 1 GB mask and generated outputs are too large for GitHub and are not included. Summary results are in [`results_toxintrace/`](results_toxintrace).

## What to expect

| Pythia-160M on RealToxicityPrompts | EMT | TP | Perplexity |
|---|---|---|---|
| Undefended, ours (paper) | 0.557 (0.557) | 0.558 (0.560) | 25.83 (25.84) |
| IF-Guide, ours (paper) | 0.115 (0.101) | 0.066 (0.054) | 26.84 (26.77) |

| GPT-2, fine-tuned on 400M tokens (E1) | EMT | TP | Perplexity |
|---|---|---|---|
| Control | 0.567 | 0.571 | 20.92 |
| Transferred mask | 0.308 | 0.271 | 21.01 |

| BOLD, IF-Guide's EMT reduction (`06`, `09`) | Prompts as given | Trailing space removed |
|---|---|---|
| Undefended EMT / IF-Guide EMT | 0.285 / 0.186 (1.5x) | 0.218 / 0.042 (5.2x) |

| E2, share of 256-token passages with toxicity above 0.5 | OpenWebText | FineWeb 2025 |
|---|---|---|
| Detoxify, 10,000 random chunks each | 1.70% | 0.56% |

| Pythia-410M, fine-tuned on 400M FineWeb tokens (E2) | EMT | TP | Perplexity (OWT) | Perplexity (FineWeb) |
|---|---|---|---|---|
| Stock, not fine-tuned | 0.615 | 0.624 | 15.81 | 18.83 |
| Control | 0.565 | 0.571 | 16.60 | 16.47 |
| Mask built on FineWeb | 0.159 | 0.119 | 16.92 | 16.63 |

All E2 metrics are in [`results_toxintrace/e2_summary.csv`](results_toxintrace/e2_summary.csv) (made by `e2_summary.py`).

## Our code

| File in `src/toxintrace/` | Purpose |
|---|---|
| `make_smoke_data.py`, `smoke_test.sh` | Tiny data and the end-to-end check |
| `make_owt_subset.py` | Compact, fingerprinted copy of the paper's training subset |
| `benchmark.sh` | Measures training and scoring speed at the paper's settings |
| `dpo.py`, `score_sweep.sh` | DPO (TRL) with Lee et al.'s settings, checkpoint sweep and scoring |
| `transfer_mask.py` | E1: moves a token mask between tokenizers through exact byte spans |
| `to_conv1d.py` | E1: converts fine-tuned GPT-2 checkpoints to a standard layout and verifies them |
| `make_fineweb_subset.py`, `compare_toxicity.py`, `inspect_mask.py` | E2: FineWeb corpus in the same format as the OpenWebText chunks; Detoxify comparison; what the masks select |
| `make_tables.py`, `analysis.py` | Tables, reduction factors, bootstrap intervals and diagnostics |
| `e2_summary.py` | One table of every metric for a set of runs of one model, with reduction factors (E2; also reproduces the E1 table) |
| `experiments/` | The numbered scripts above |

Changes to the authors' files (`git diff main toxintrace`):

- `utils/datasets.py`: `IFGUIDE_OWT_PATH` (local data), `IFGUIDE_TOKENIZED_PATH` (pre-tokenized data for other tokenizers), `IFGUIDE_STRIP_BOLD` (strip BOLD's trailing space).
- `utils/models.py`, `utils/registry.yaml`: new models, architecture overrides, and the Conv1D-to-Linear swap keeps its dtype.
- `run_fluency_eval.py`: position IDs for left-padded LAMBADA on GPT-2 (Pythia's path is unchanged).
- `utils/rad/utils/logits_processor.py`: RAD's per-row loops batched; completions were identical on test prompts.
- `train.py`, `finetune.py`, `fit_factors.py`, `build_toxic_token_mask.py`, `scripts/run_toxicity_eval.sh`: command-line options and small fixes.

## Troubleshooting

- **Very slow under WSL2:** past 24 GB, the Windows driver spills GPU memory into system RAM instead of failing. Close other GPU programs, use `MICRO_BATCH=4` for training, and keep RAD at batch 32.
- **WSL2 stops a background job:** keep one terminal window open while jobs run in `tmux`.
- **`inf` or `nan` after the first step in your own scripts:** Pythia's configs default to float16. Pass `torch_dtype=torch.bfloat16`, as `utils/models.py` does.
- **RAM:** building the mask peaks at about 15 GB; the E1 mask transfer at about 27 GB (it ran on swap).

## Credit

The method and most of the pipeline are by Coalson et al.; please cite their paper. License: see [`LICENSE`](LICENSE).
