"""Run experiment jobs in worker processes, checkpointing each finished job.

A job is a JSON-serializable dict. With a checkpoint directory, each job's result is saved to
``<dir>/<hash of the job>.json`` and loaded instead of recomputed on a rerun, so an interrupted
experiment resumes. Inside a job, ``job["_checkpoint"]`` is that path; a long job can store
finer-grained progress next to it (e.g. ``Path(job["_checkpoint"]).with_suffix(".points.jsonl")``).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from qcc_common.results import to_jsonable


def job_key(job: dict) -> str:
    """Stable 16-hex-digit hash of a job's settings."""
    blob = json.dumps(to_jsonable(job), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _checkpointed(args: tuple[Callable, dict]):
    fn, job = args
    path = Path(job["_checkpoint"])
    if path.is_file():
        return json.loads(path.read_text())
    result = to_jsonable(fn(job))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    return result


def parallel_map(
    fn: Callable, jobs: list[dict], workers: int = 1, checkpoint_dir: Path | None = None
) -> list:
    """``[fn(job) for job in jobs]`` over ``workers`` spawned processes (``fn`` must be a
    module-level function). Results keep the order of ``jobs``."""
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        jobs = [
            {**job, "_checkpoint": str(checkpoint_dir / f"{job_key(job)}.json")} for job in jobs
        ]
        args, call = [(fn, job) for job in jobs], _checkpointed
    else:
        args, call = jobs, fn
    if workers <= 1 or len(jobs) <= 1:
        return [call(a) for a in args]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as pool:
        return list(pool.map(call, args))
