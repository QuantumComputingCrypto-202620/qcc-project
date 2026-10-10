# qcc-project

Monorepo for individual quantum computing course projects: 8-week, simulator-first studies in
Python with Qiskit. The repo-wide design is in [SPEC.md](SPEC.md).

| Project | Spec | Summary |
|---|---|---|
| [`projects/qae-options`](projects/qae-options/) | [SPEC](projects/qae-options/SPEC.md) | QAE, IQAE and MLAE vs classical Monte Carlo for a digital payoff on a bimodal distribution. |
| [`projects/vqe-lih`](projects/vqe-lih/) | [SPEC](projects/vqe-lih/SPEC.md) | UCCSD vs a hardware-efficient ansatz along the full LiH dissociation curve. |

Shared code lives in [`packages/qcc-common`](packages/qcc-common/): simulator backends
(`statevector_simulator`, `shot_simulator`, `noisy_simulator`) and `circuit_metrics`.

## Quickstart

Everything goes through [uv](https://docs.astral.sh/uv/). Linux or macOS is required
(PySCF has no Windows wheels; use WSL2 on Windows).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync --all-packages                             # create .venv with every member + dev tools
uv run pytest -m "not slow"                        # fast tests for the whole workspace
```

Run a project:

```bash
uv run --package qae-options qae-options e2 --config projects/qae-options/configs/e2_convergence.toml
uv run --package vqe-lih pytest projects/vqe-lih
uv run --package vqe-lih jupyter lab
```

See [SPEC.md §3.4](SPEC.md#34-command-cheat-sheet) for the full command cheat-sheet.
IBM Quantum hardware runs read the token from `$QISKIT_IBM_TOKEN` or a gitignored `.env` file.

## Resolved versions

The workspace has one lockfile (`uv.lock`), so both projects share these versions
(Python 3.12, resolved 2026-10-10):

| Package | Version |
|---|---|
| qiskit | 2.5.2 |
| qiskit-aer | 0.17.2 |
| qiskit-algorithms | 0.4.0 |
| qiskit-nature | 0.8.0 |
| qiskit-ibm-runtime | 0.50.0 |
| pyscf | 2.14.0 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |

Upgrade one package with `uv lock --upgrade-package <pkg>`, then run every member's tests
before committing the lockfile.
