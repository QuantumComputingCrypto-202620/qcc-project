# vqe-lih

LiH dissociation curve with VQE: UCCSD vs a hardware-efficient ansatz, against exact
diagonalization. Results and conclusions are in **[REPORT.md](REPORT.md)**; the design,
experiments and milestones are in [SPEC.md](SPEC.md); repo-wide rules are in the
[root SPEC](../../SPEC.md).

```bash
uv sync --package vqe-lih
uv run --package vqe-lih pytest projects/vqe-lih -m "not slow"   # fast tests
uv run --package vqe-lih pytest projects/vqe-lih -m slow         # end-to-end smoke test (~3 min)
uv run --package vqe-lih vqe-lih e2 --config projects/vqe-lih/configs/e2_uccsd.toml
projects/vqe-lih/run_all.sh                                       # every experiment, ~3–4 h
```

| Experiment | Config | Question |
|---|---|---|
| `e1` | `e1_h2.toml` | H₂ pipeline validation |
| `e2` | `e2_uccsd.toml` | UCCSD along the LiH curve, (2e, 3o) and (4e, 4o) |
| `e3` | `e3_hea.toml` | Hardware-efficient ansatz, reps 1–4 |
| `e4` | `e4_tradeoff.toml` | Depth vs accuracy; fills in the claim |
| `e5` | `e5_optimizers.toml` | Optimizers × initialization |
| `e6` | `e6_shots.toml` | Shot noise |
| `e7` | `e7_noise.toml` | Noisy simulator + readout mitigation + ZNE |
| `beh2` | `beh2.toml` | BeH₂ stretch goal |

Each run writes `results/raw/<name>.json` (gitignored; full records with config, seed, git SHA,
`uv.lock` hash, parameters and energy histories), `results/summary/<name>*.csv` and
`results/figures/<name>*.{png,pdf}` (committed). Finished jobs are checkpointed in
`results/raw/<name>.parts/`, so an interrupted run resumes; delete that directory to force a
recompute.

Library layout (`src/vqe_lih/`): `molecules` (geometries), `hamiltonian` (driver → active space →
qubits), `reference` (exact, FCI, HF), `ansatze`, `solver` (`run_vqe`), `measurement`
(shot-based estimator), `mitigation` (readout + ZNE), `experiments` (runners), `cli`.
