# qae-options

Option pricing with quantum amplitude estimation. The design, experiments and milestones are in [SPEC.md](SPEC.md); repo-wide rules
are in the [root SPEC](../../SPEC.md).

```bash
uv sync --package qae-options
uv run --package qae-options pytest projects/qae-options -m "not slow"
uv run --package qae-options qae-options e2 --config projects/qae-options/configs/e2_convergence.toml
```

Layout: library code and the CLI in `src/qae_options/`, experiment configs in `configs/`,
tests in `tests/`, outputs in `results/` (`raw/` is gitignored; `summary/` and
`figures/` are committed).

**Status:** Week 0 scaffold. No experiments are implemented yet.
