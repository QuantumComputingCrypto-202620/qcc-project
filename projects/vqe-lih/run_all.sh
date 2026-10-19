#!/usr/bin/env sh
# Regenerate every vqe-lih result from the committed configs (run from the repository root).
# Finished jobs are checkpointed under results/raw/<name>.parts/, so re-running this script
# after an interruption resumes instead of starting over. E1 takes seconds; E2/E3 take hours.
set -eu
run() { uv run --package vqe-lih vqe-lih "$@"; }
C=projects/vqe-lih/configs

run e1 --config "$C/e1_h2.toml"
run e6 --config "$C/e6_shots.toml" --workers 1 &   # independent of the curves
run e2 --config "$C/e2_uccsd.toml"
run e3 --config "$C/e3_hea.toml" --workers 3
wait
run e4 --config "$C/e4_tradeoff.toml"
run e5 --config "$C/e5_optimizers.toml"   # warm starts come from E2/E3
run e7 --config "$C/e7_noise.toml"        # parameters come from E2/E3
run beh2 --config "$C/beh2.toml"
