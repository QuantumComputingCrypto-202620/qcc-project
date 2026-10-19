# qae-options

Option pricing with quantum amplitude estimation: QAE, IQAE and MLAE compared with Monte Carlo
and quasi-Monte Carlo on a bimodal distribution and a digital payoff. Results and conclusions
are in **[REPORT.md](REPORT.md)**; the design is in [SPEC.md](SPEC.md); repo-wide rules are in
the [root SPEC](../../SPEC.md).

```bash
uv sync --package qae-options
uv run --package qae-options pytest projects/qae-options -m "not slow"   # fast tests
uv run --package qae-options pytest projects/qae-options -m slow         # end-to-end smoke test
uv run --package qae-options qae-options e2 --config projects/qae-options/configs/e2_convergence.toml
projects/qae-options/run_all.sh                                           # every experiment, ~1.5 h
```

| Experiment | Config | Question |
|---|---|---|
| `e1` | `e1_validate.toml` | Correct prices? Lognormal call/digital vs Black-Scholes; barrier (stretch) |
| `e2` | `e2_convergence.toml` | Error vs oracle queries: QAE, IQAE, MLAE, MC, QMC |
| `e3` | `e3_resources.toml` | Qubits, depth, 2q gates at ε = 10⁻³; fills in the claim |
| `e4` | `e4_noise.toml` | Noise frontier on a FakeSherbrooke noise model |
| `e5` | `e5_loading.toml` | Cost of loading the distribution vs n |

Each run writes `results/raw/<name>.json` (gitignored, with config, seed, git SHA, `uv.lock`
hash and versions), `results/summary/<name>*.csv|json` and `results/figures/<name>*.{png,pdf}`.
Finished jobs are checkpointed in `results/raw/<name>.parts/`; delete it to force a recompute.

Library layout (`src/qae_options/`): `distributions` (grids, loader), `payoffs` (A operators),
`reference` (closed forms, quadrature), `classical` (MC, QMC), `estimators` (QAE / IQAE / MLAE
with query counting), `sampling` (seeded SamplerV2), `resources` (circuit cost),
`experiments` (runners), `cli`.
