"""Command-line entry point: `vqe-lih <experiment> --config <file.toml>`."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

KNOWN_EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "beh2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vqe-lih", description=__doc__)
    parser.add_argument("experiment", choices=KNOWN_EXPERIMENTS, help="experiment ID from SPEC.md")
    parser.add_argument("--config", type=Path, required=True, help="experiment config (TOML)")
    parser.add_argument(
        "--results-dir", type=Path, default=None, help="override results/ (raw, summary, figures)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # One BLAS thread per process: experiments parallelize over worker processes instead.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from vqe_lih.experiments import RUNNERS, load_config

    config = load_config(args.config)
    if config.get("experiment") not in (None, args.experiment):
        print(
            f"vqe-lih: {args.config} is a config for {config['experiment']!r}, "
            f"not {args.experiment!r}",
            file=sys.stderr,
        )
        return 2
    config["experiment"] = args.experiment
    config.setdefault("name", args.config.stem)
    if args.results_dir is not None:
        config["results_dir"] = str(args.results_dir)
    RUNNERS[args.experiment](config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
