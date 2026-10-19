#!/usr/bin/env sh
# Regenerate every qae-options result from the committed configs (run from the repository root).
# Finished jobs are checkpointed under results/raw/<name>.parts/, so re-running this script after
# an interruption resumes instead of starting over.
set -eu
run() { uv run --package qae-options qae-options "$@"; }
C=projects/qae-options/configs

run e5 --config "$C/e5_loading.toml"
run e1 --config "$C/e1_validate.toml"
run e2 --config "$C/e2_convergence.toml"
run e3 --config "$C/e3_resources.toml"   # operating points come from E2
run e4 --config "$C/e4_noise.toml"
