"""Experiment records: config, seed and provenance (git SHA, ``uv.lock`` hash, package versions,
timestamp), saved as JSON under a project's ``results/raw/``."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

TRACKED_PACKAGES = (
    "qiskit",
    "qiskit-aer",
    "qiskit-algorithms",
    "qiskit-nature",
    "qiskit-ibm-runtime",
    "pyscf",
    "numpy",
    "scipy",
)


def find_repo_root(start: Path | None = None) -> Path | None:
    """First parent of ``start`` (default: this file) that contains ``uv.lock``."""
    here = (start or Path(__file__)).resolve()
    for directory in (here, *here.parents):
        if (directory / "uv.lock").is_file():
            return directory
    return None


def git_sha(repo: Path | None = None) -> str | None:
    """``HEAD`` commit, with a ``-dirty`` suffix if tracked files have local changes."""
    cwd = repo or find_repo_root()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def uv_lock_sha256(repo: Path | None = None) -> str | None:
    root = repo or find_repo_root()
    lock = root / "uv.lock" if root else None
    if lock is None or not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays, complex numbers, paths and dataclasses for ``json``."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, complex | np.complexfloating):
        if abs(value.imag) > 1e-12:
            return {"real": float(value.real), "imag": float(value.imag)}
        return float(value.real)
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class ExperimentRecord:
    """Everything needed to trace a result back to its inputs."""

    experiment: str
    config: dict[str, Any]
    seed: int | None
    payload: Any
    metrics: dict[str, Any] = field(default_factory=dict)
    git_sha: str | None = field(default_factory=git_sha)
    uv_lock_sha256: str | None = field(default_factory=uv_lock_sha256)
    package_versions: dict[str, str] = field(default_factory=package_versions)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_jsonable(dataclasses.asdict(self)), indent=1))
        return path

    @classmethod
    def load(cls, path: Path | str) -> ExperimentRecord:
        return cls(**json.loads(Path(path).read_text()))
