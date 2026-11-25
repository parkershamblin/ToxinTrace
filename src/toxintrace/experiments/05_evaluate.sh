#!/usr/bin/env bash
# R1 + R2: RTP, AttaQ, BOLD, implicit toxicity and fluency (~40 min per model, ~6 h per RAD model), then the tables
source "$(dirname "$0")/env.sh"
B=$(base_run)/final; I=$(ifguide_run)/final
S=$TT_ROOT/models/dpo_sweep/dpo/$MODEL
evaluate baseline "$B"
evaluate ifguide "$I"
evaluate dpo300_baseline "$(latest_with "$S/baseline" checkpoint-300)/checkpoint-300"
evaluate dpo400_ifguide "$(latest_with "$S/ifguide" checkpoint-400)/checkpoint-400"
evaluate baseline_rad "$B" rad
evaluate ifguide_rad "$I" rad
run python toxintrace/make_tables.py --results_dir "$OUT" --sweep_dir "$TT_ROOT/results_sweep"
