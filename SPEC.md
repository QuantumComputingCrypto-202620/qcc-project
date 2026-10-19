# qcc-project: Repository Specification

This spec covers the whole repository: layout, tooling, shared code, conventions and CI.
Each project has its own spec that defines its science, experiments and schedule:

| Project | Directory | Spec | Summary |
|---|---|---|---|
| QAE option pricing | [`projects/qae-options/`](projects/qae-options/) | [SPEC](projects/qae-options/SPEC.md) | Compares QAE, IQAE and MLAE with classical Monte Carlo for a digital or barrier payoff on a bimodal distribution. |
| VQE LiH dissociation | [`projects/vqe-lih/`](projects/vqe-lih/) | [SPEC](projects/vqe-lih/SPEC.md) | Compares UCCSD and a hardware-efficient ansatz along the full LiH curve, against exact diagonalization. |

---

## 1. Overview and goals

`qcc-project` is a monorepo for individual projects in a quantum computing course, with one
directory per project. Every project is an 8-week, simulator-first study written in Python
with Qiskit.

Principles:

1. **Reproducible.** Every result can be regenerated from three things: a committed lockfile
   (`uv.lock`), a committed config file and a fixed seed.
2. **Honest comparisons.** No project claims quantum advantage. Each project states a
   *verifiable claim* with numeric placeholders and fills them in from measured data, even if
   the result is unfavourable.
3. **Isolated projects.** Projects do not import each other. Each project has its own
   dependencies, tests, configs, results and spec.
4. **Minimal shared code.** Code moves into the shared package only after **two or more
   projects** need it.

## 2. Repository layout

```text
qcc-project/
├── SPEC.md                      # this file: repo-wide spec
├── README.md                    # quickstart: install uv, uv sync, run a project
├── pyproject.toml               # uv workspace root (virtual: no [project] table)
├── uv.lock                      # single lockfile for the whole workspace (committed)
├── .python-version              # Python version used by uv
├── .gitignore
├── .github/
│   └── workflows/ci.yml         # lint + fast tests for every workspace member
├── packages/                    # shared libraries (not projects)
│   └── qcc-common/
│       ├── pyproject.toml       # name = "qcc-common"
│       ├── src/qcc_common/
│       │   ├── __init__.py
│       │   ├── backends.py      # Aer simulators, noise models, optional IBM Runtime
│       │   ├── metrics.py       # transpiled depth, 2q-gate count, qubits, parameters
│       │   ├── results.py       # experiment records (config, seed, git SHA, versions)
│       │   └── plotting.py      # shared matplotlib style, log-log helpers
│       └── tests/
└── projects/                    # one directory per project, fully self-contained
    ├── qae-options/
    │   ├── SPEC.md              # project spec
    │   ├── README.md            # how to run this project
    │   ├── pyproject.toml       # name = "qae-options", import name qae_options
    │   ├── src/qae_options/     # library code + cli.py
    │   ├── configs/             # experiment configs (TOML)
    │   ├── notebooks/           # exploration only
    │   ├── tests/
    │   └── results/
    │       ├── raw/             # gitignored
    │       ├── summary/         # small CSVs, committed
    │       └── figures/         # final plots, committed
    └── vqe-lih/
        ├── SPEC.md
        ├── README.md
        ├── pyproject.toml       # name = "vqe-lih", import name vqe_lih
        ├── src/vqe_lih/
        ├── configs/
        ├── notebooks/
        ├── tests/
        └── results/{raw,summary,figures}/
```

### Separation rules

- `projects/*` may depend on `packages/*`. They must **never** depend on another `projects/*`.
- `packages/*` must never depend on `projects/*`.
- Each project owns its `pyproject.toml`, `README.md`, `SPEC.md`, `tests/`, `configs/` and `results/`.
- Notebooks are for exploration and presenting results. Any function a notebook relies on
  lives in `src/` and is tested.
- Naming: directory and distribution names use kebab-case (`qae-options`); import names use
  snake_case (`qae_options`).

## 3. Tooling: uv workspace

All Python environments, dependencies and commands go through **[uv](https://docs.astral.sh/uv/)**.
`pip install`, `conda` and hand-made virtualenvs are not used. The repo is a
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): every member shares
one `.venv` and one `uv.lock`.

### 3.1 Root `pyproject.toml`

```toml
[tool.uv.workspace]
members = ["packages/*", "projects/*"]

[dependency-groups]
dev = [
    "pytest>=8",
    "ruff>=0.6",
    "ipykernel>=6",
    "jupyterlab>=4",
]

[tool.ruff]
line-length = 100
target-version = "py312"
extend-exclude = ["**/notebooks/**"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "NPY"]

[tool.pytest.ini_options]
# importlib mode lets every member have tests/test_*.py files with the same names.
addopts = ["--import-mode=importlib"]
markers = ["slow: long-running simulation tests (deselect with -m 'not slow')"]
```

The root is a *virtual* workspace root: it has no `[project]` table and is never built or
installed.

### 3.2 Member `pyproject.toml` (pattern)

```toml
[project]
name = "qae-options"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "qcc-common",
    "qiskit-algorithms",
    "scipy",
]

[project.scripts]
qae-options = "qae_options.cli:main"

[tool.uv.sources]
qcc-common = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 3.3 Python version

`.python-version` pins **3.12**. Week 0 confirmed that qiskit, qiskit-aer, qiskit-algorithms,
qiskit-nature and pyscf all have wheels for it; the resolved versions are in the README.

### 3.4 Command cheat-sheet

| Task | Command |
|---|---|
| Install everything (all members + dev group) | `uv sync --all-packages` |
| Install a single project | `uv sync --package vqe-lih` |
| Run a project CLI | `uv run --package qae-options qae-options sweep --config projects/qae-options/configs/e2.toml` |
| Run one project's tests | `uv run --package vqe-lih pytest projects/vqe-lih` |
| Run all fast tests | `uv run pytest -m "not slow"` |
| Add a dependency to a project | `uv add --package qae-options scipy` |
| Add a dev tool repo-wide | `uv add --dev mypy` |
| Add an optional extra | `uv add --package vqe-lih --optional stretch <pkg>` |
| Upgrade one package in the lock | `uv lock --upgrade-package qiskit` |
| Verify the lock is up to date (CI) | `uv lock --check` |
| Start Jupyter | `uv run --package vqe-lih jupyter lab` |

Rules:

- `uv.lock` is always committed. Any change to dependencies is a `uv add` or `uv remove`
  followed by a commit that includes the lockfile.
- `qiskit-ibm-runtime` is a core dependency of `qcc-common`, because the fake IBM backends
  that `noisy_simulator` needs (`FakeSherbrooke`, …) ship only in that package since Qiskit 1.0.
  Installing it does not touch hardware: real-device access still needs an explicit
  `ibm_runtime_session()` call and a token (§5.4).

### 3.5 Single lockfile: implications

A workspace resolves **one** version of every package for all members, so both projects share
one Qiskit version. qiskit-nature (0.7.x) and qiskit-algorithms are community-maintained and
often lag behind Qiskit core, so the Qiskit version is set by what they support.

- Week 0: run `uv lock` with all members and record the resolved versions in the README.
- If the projects ever need incompatible versions:
  - first choice: declare them with [`[tool.uv] conflicts`](https://docs.astral.sh/uv/concepts/projects/config/#conflicting-dependencies);
  - otherwise: remove the project from `members` and give it its own lockfile. It keeps its
    place under `projects/` and still uses `qcc-common` through a path source.

## 4. Shared package: `qcc-common`

This package contains only code that both projects need. Every module is small and tested.

| Module | Responsibility |
|---|---|
| `backends` | `statevector_simulator()`, `shot_simulator(shots, seed)`, `noisy_simulator(fake_backend="FakeSherbrooke", scale=1.0)`. `scale` multiplies every gate and readout error rate and divides T1/T2. `ibm_runtime_session()` reads `QISKIT_IBM_TOKEN` from the environment or `.env`. |
| `metrics` | `circuit_metrics(circuit, basis=..., optimization_level=..., seed_transpiler=...) -> {num_qubits, num_parameters, logical_depth, depth, two_qubit_gates, transpile}`. Transpilation uses one fixed basis ({rz, sx, x, ecr}), optimization level 2, a fixed seed and a linear coupling map, so numbers are comparable across projects; `transpile` records those settings. `logical_depth` is the depth after unrolling to {u, cx} without mapping or optimization. |
| `results` | `ExperimentRecord` (config dict, seed, git SHA, `uv.lock` hash, package versions, timestamp, metrics, payload). It provides `save()`/`load()` to JSON under `results/raw/`. Parquet for large sweeps is added when a project needs it. |
| `plotting` | Shared matplotlib style (`apply_style`), a validated categorical palette with per-series markers (`series_style`), standard figure sizes, and `save_figure` (PNG + PDF). `loglog_with_fit(x, y, yerr)` is planned for qae-options. |

What does **not** belong here: anything about finance or chemistry, algorithm wrappers,
project configs.

## 5. Cross-cutting conventions

### 5.1 Experiments

- Each experiment (E1, E2, …, as defined in the project spec) has a TOML config in
  `projects/<p>/configs/`.
- Every run goes through the project CLI (`[project.scripts]`), so notebooks only *read*
  results and never produce them.
- Every config sets an explicit `seed`. Repetitions use `seed + i`.
- Outputs:
  - `results/raw/`: everything, gitignored.
  - `results/summary/`: small CSVs, committed.
  - `results/figures/`: final figures, committed, both PNG and PDF.

### 5.2 Testing

- `pytest` lives in each member's `tests/`.
- Fast tests (seconds) run in CI. These include analytic checks, normalization checks and small
  circuits.
- Simulations longer than about 30 s are marked `@pytest.mark.slow` and run locally.
- Each project spec lists its required tests.

### 5.3 Code style

- `ruff check` and `ruff format`. Type hints on public functions.
- Docstrings say what the units are (Hartree, Å, price units) wherever relevant.

### 5.4 Secrets

- The IBM Quantum token comes only from the `QISKIT_IBM_TOKEN` environment variable or a local
  `.env` file, which is gitignored.
- Tokens are never written into configs, notebooks or results.

### 5.5 CI (`.github/workflows/ci.yml`)

- Uses `astral-sh/setup-uv`, then runs:
  1. `uv lock --check`
  2. `uv sync --locked --all-packages`
  3. `uv run ruff check`
  4. `uv run ruff format --check`
  5. `uv run pytest -m "not slow"`
- A matrix over the workspace members (`packages/qcc-common`, `projects/qae-options`,
  `projects/vqe-lih`) runs each member's tests in its own job.
- Runs on Ubuntu only, because PySCF has no Windows wheels.

## 6. Adding a new project

1. `uv init --package projects/<name>`. It is picked up automatically by the `projects/*` member glob.
2. Add `qcc-common = { workspace = true }` under `[tool.uv.sources]` if it is needed.
3. Create `SPEC.md` (follow the structure of the existing project specs), `README.md`,
   `configs/`, `tests/` and `results/{raw,summary,figures}/`.
4. Add the project to the Projects table at the top of this file and to the CI matrix.
5. `uv lock`, then commit the lockfile.

## 7. Milestones

### Week 0: shared scaffold (before either project's week 1)

- [x] Root `pyproject.toml`, `.python-version`, `.gitignore` (`.venv/`, `results/raw/`, `.env`, `.ipynb_checkpoints/`)
- [x] `packages/qcc-common` skeleton with `backends` and `metrics`, plus tests
- [x] `uv init --package` for `projects/qae-options` and `projects/vqe-lih`
- [x] `uv lock` resolves qiskit, qiskit-aer, qiskit-algorithms and qiskit-nature + pyscf together; resolved versions recorded in the README
- [x] Smoke test: a Bell circuit on the Aer simulator through `qcc_common.backends`, and `circuit_metrics` on it
- [ ] CI green on an empty test suite
- [x] `README.md` quickstart

**Done when:** a fresh clone runs `uv sync --all-packages && uv run pytest` with everything
passing.

### Weeks 1–8

Each project has its own schedule:

- [QAE options: milestones](projects/qae-options/SPEC.md#8-milestones)
- [VQE LiH: milestones](projects/vqe-lih/SPEC.md#9-milestones)

## 8. Cross-cutting risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| qiskit-nature or qiskit-algorithms lag behind Qiskit core | Lock fails, or deprecated APIs | Pin Qiskit to the line both support (checked in week 0); fall back to `[tool.uv] conflicts` or a separate lock (§3.5). |
| Single lockfile couples the projects | Upgrading one project breaks the other | Upgrade with `uv lock --upgrade-package <pkg>` and run both test suites before committing. |
| PySCF unavailable on Windows | Contributor cannot install `vqe-lih` | Use WSL2 or a Linux/macOS machine; CI runs on Ubuntu. |
| IBM hardware access or queue times | Optional hardware experiments slip | Hardware runs are optional in both projects; noisy simulators from fake backends are the baseline. |
