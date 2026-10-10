"""Command-line entry point: `vqe-lih <experiment> --config <file.toml>`."""

import argparse
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path

# Experiment ID -> runner(config). Runners are added to experiments.py as each one is built.
EXPERIMENTS: dict[str, Callable[[dict], None]] = {}
KNOWN_EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5", "e6", "e7")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vqe-lih", description=__doc__)
    parser.add_argument("experiment", choices=KNOWN_EXPERIMENTS, help="experiment ID from SPEC.md")
    parser.add_argument("--config", type=Path, required=True, help="experiment config (TOML)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("rb") as f:
        config = tomllib.load(f)
    runner = EXPERIMENTS.get(args.experiment)
    if runner is None:
        print(f"vqe-lih: experiment {args.experiment!r} is not implemented yet", file=sys.stderr)
        return 2
    runner(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
