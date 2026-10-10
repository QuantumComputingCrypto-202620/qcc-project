# vqe-lih

LiH dissociation curve with VQE. The design, experiments and milestones are in [SPEC.md](SPEC.md); repo-wide rules
are in the [root SPEC](../../SPEC.md).

```bash
uv sync --package vqe-lih
uv run --package vqe-lih pytest projects/vqe-lih -m "not slow"
uv run --package vqe-lih vqe-lih e2 --config projects/vqe-lih/configs/e2_uccsd.toml
```

Layout: library code and the CLI in `src/vqe_lih/`, experiment configs in `configs/`,
tests in `tests/`, outputs in `results/` (`raw/` is gitignored; `summary/` and
`figures/` are committed).

**Status:** Week 0 scaffold. No experiments are implemented yet.
