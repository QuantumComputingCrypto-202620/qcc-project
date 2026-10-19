"""Experiment runners E1-E7 (SPEC §8), driven by TOML configs.

Every runner writes ``results/raw/<name>.json`` (full records incl. parameters and energy
histories, gitignored), ``results/summary/<name>*.csv`` (committed) and
``results/figures/<name>*.{png,pdf}`` (committed). Energies in Ha, bond lengths in Å.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import sys
import tomllib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qcc_common.metrics import circuit_metrics
from qcc_common.results import ExperimentRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
CHEM_ACC_MHA = 1.6
ANSATZ_SLOTS = {"uccsd": 0, "hea_r1": 1, "hea_r2": 2, "hea_r3": 3, "hea_r4": 4}
OPTIMIZER_SLOTS = {"l_bfgs_b": 0, "cobyla": 1, "spsa": 2}
INIT_SLOTS = {"zeros": 0, "random": 1, "warm_start": 2}
DELTA_E_FLOOR = 1e-9  # Ha; ΔE below this is plotted at the floor on log axes


# ---------------------------------------------------------------------------- config & I/O


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a TOML config; a ``curve = "file.toml"`` key merges that file's bond lengths."""
    path = Path(path)
    with path.open("rb") as f:
        config = tomllib.load(f)
    if "curve" in config:
        with (path.parent / config["curve"]).open("rb") as f:
            curve = tomllib.load(f)
        for key, value in curve.items():
            config.setdefault(key, value)
    config.setdefault("results_dir", str(RESULTS_DIR))
    return config


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _dirs(config: dict) -> tuple[Path, Path, Path]:
    root = Path(config["results_dir"])
    dirs = (root / "raw", root / "summary", root / "figures")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _save(config: dict, rows: list[dict], summary: pd.DataFrame, extra: dict | None = None):
    raw, summ, _ = _dirs(config)
    ExperimentRecord(
        experiment=config["experiment"],
        config=config,
        seed=config.get("seed"),
        payload=rows,
        metrics=extra or {},
    ).save(raw / f"{config['name']}.json")
    summary.to_csv(summ / f"{config['name']}.csv", index=False, float_format="%.10g")


def load_raw_rows(config: dict, name: str) -> list[dict]:
    path = Path(config["results_dir"]) / "raw" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: run the experiment that produces {name!r} first"
        )
    return json.loads(path.read_text())["payload"]


def load_summary(config: dict, name: str) -> pd.DataFrame:
    path = Path(config["results_dir"]) / "summary" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: run the experiment that produces {name!r} first"
        )
    return pd.read_csv(path)


def _parallel_map(fn: Callable, jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(job) for job in jobs]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as pool:
        return list(pool.map(fn, jobs))


def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ("parameters", "history")}


# ----------------------------------------------------------------------- curve (E1, E2, E3)


def _curve_job(job: dict) -> list[dict]:
    """One warm-start sweep for one (active space, ansatz) along the bond-length grid."""
    from vqe_lih.ansatze import AnsatzSpec, build_ansatz, compile_for_simulation
    from vqe_lih.hamiltonian import ActiveSpace, build_hamiltonian
    from vqe_lih.molecules import warm_start_order
    from vqe_lih.reference import exact_energy, fci_energy
    from vqe_lih.solver import initial_point, run_vqe

    space = ActiveSpace.from_dict(job["active_space"])
    spec = AnsatzSpec.from_dict(job["ansatz"])
    opt = job["optimizer"]
    first = job.get("first_point", {})
    seed = int(job["seed"])
    if job.get("warm_start", True):
        order = warm_start_order(job["bond_lengths"], job["equilibrium"])
    else:
        order = [(float(r), None) for r in sorted(job["bond_lengths"])]

    ansatz = metrics = None
    params_at: dict[float, list[float]] = {}
    rows = []
    for r, prev in order:
        ham = build_hamiltonian(job["molecule"], r, job["basis"], space, job["mapper"])
        if ansatz is None:
            circuit = build_ansatz(spec, ham)
            metrics = circuit_metrics(circuit)
            ansatz = compile_for_simulation(circuit)
        exact = exact_energy(ham)
        e_fci = fci_energy(job["molecule"], r, job["basis"])
        n = ansatz.num_parameters

        if prev is None:
            strategy = first.get("init", "zeros")
            restarts = int(first.get("restarts", 1)) if strategy == "random" else 1
            candidates = [
                run_vqe(
                    ham,
                    ansatz,
                    opt["name"],
                    initial_point(strategy, n, seed=seed + i),
                    e_exact=exact.energy,
                    init=strategy,
                    seed=seed + i,
                    maxiter=opt.get("maxiter"),
                )
                for i in range(restarts)
            ]
            record = min(candidates, key=lambda rec: rec.energy)
            restart_energies = [c.energy for c in candidates]
        else:
            record = run_vqe(
                ham,
                ansatz,
                opt["name"],
                initial_point("warm_start", n, warm=params_at[prev]),
                e_exact=exact.energy,
                init="warm_start",
                seed=seed,
                maxiter=opt.get("maxiter"),
            )
            restart_energies = None
        params_at[r] = record.parameters
        denom = ham.hf_energy - exact.energy
        row = {
            "molecule": job["molecule"],
            "active_space": space.label,
            "ansatz": spec.label,
            "bond_length": r,
            "warm_start_from": prev,
            "num_qubits": ham.num_qubits,
            "num_parameters": metrics["num_parameters"],
            "logical_depth": metrics["logical_depth"],
            "depth": metrics["depth"],
            "two_qubit_gates": metrics["two_qubit_gates"],
            "e_exact": exact.energy,
            "e_fci": e_fci,
            "e_hf": ham.hf_energy,
            "e_vqe": record.energy,
            "active_space_error": exact.energy - e_fci,
            "correlation_recovered": (ham.hf_energy - record.energy) / denom if denom else None,
            **record.summary(),
            "restart_energies": restart_energies,
            "parameters": record.parameters,
            "history": record.history,
        }
        rows.append(row)
        log(
            f"[{job['molecule']} {space.label} {spec.label}] R={r:.3f} "
            f"dE={record.delta_e:.2e} Ha nfev={record.nfev} {record.wall_time:.1f}s"
        )
    return sorted(rows, key=lambda row: row["bond_length"])


def run_curve(config: dict) -> pd.DataFrame:
    """E1/E2/E3 (and the BeH2 stretch): VQE along a dissociation curve, warm-started."""
    jobs = []
    for space in config.get("active_spaces", [{"label": "full"}]):
        for i, ansatz in enumerate(config["ansatze"]):
            jobs.append(
                {
                    "molecule": config["molecule"],
                    "basis": config.get("basis", "sto3g"),
                    "mapper": config.get("mapper", "jordan_wigner"),
                    "active_space": space,
                    "ansatz": ansatz,
                    "bond_lengths": config["bond_lengths"],
                    "equilibrium": config.get("equilibrium", config["bond_lengths"][0]),
                    "warm_start": config.get("warm_start", True),
                    "optimizer": config["optimizer"],
                    "first_point": config.get("first_point", {}),
                    "seed": config["seed"] + 1000 * i,
                }
            )
    # Biggest jobs first for better load balance.
    jobs.sort(key=lambda j: -(j["active_space"].get("num_spatial_orbitals") or 99))
    rows = [
        row
        for result in _parallel_map(_curve_job, jobs, config.get("workers", 1))
        for row in result
    ]
    summary = pd.DataFrame([_strip(r) for r in rows]).drop(columns=["restart_energies"])
    _save(config, rows, summary)
    plot_curves(config, summary)
    return summary


# ------------------------------------------------------------------------------- plotting


def _plt():
    import matplotlib.pyplot as plt

    from qcc_common.plotting import apply_style

    apply_style()
    return plt


def _chem_acc_line(ax, label: bool = True) -> None:
    from qcc_common.plotting import MUTED, REFERENCE

    ax.axhline(CHEM_ACC_MHA * 1e-3, color=REFERENCE, linestyle="--", linewidth=1.0)
    if label:
        ax.annotate(
            "chemical accuracy (1.6 mHa)",
            xy=(1.0, CHEM_ACC_MHA * 1e-3),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=MUTED,
        )


def _abs_floor(values) -> np.ndarray:
    return np.maximum(np.abs(np.asarray(values, dtype=float)), DELTA_E_FLOOR)


def _slot(mapping: dict[str, int], key: str, fallback: int) -> int:
    return mapping.get(key, fallback)


def plot_curves(config: dict, summary: pd.DataFrame) -> None:
    from qcc_common.plotting import FIGSIZE_WIDE, REFERENCE, save_figure, series_style

    plt = _plt()
    _, _, figdir = _dirs(config)
    molecule = config["molecule"]
    for space, df in summary.groupby("active_space", sort=False):
        fig, (ax_e, ax_d) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
        ref = df.drop_duplicates("bond_length").sort_values("bond_length")
        ax_e.plot(ref.bond_length, ref.e_hf, color=REFERENCE, linestyle=":", label="Hartree-Fock")
        ax_e.plot(
            ref.bond_length, ref.e_fci, color=REFERENCE, linestyle="--", label="FCI (full space)"
        )
        ax_e.plot(
            ref.bond_length, ref.e_exact, color=REFERENCE, linewidth=2.0, label=f"exact ({space})"
        )
        for k, (label, d) in enumerate(df.groupby("ansatz", sort=False)):
            d = d.sort_values("bond_length")
            style = series_style(_slot(ANSATZ_SLOTS, label, k))
            ax_e.plot(d.bond_length, d.e_vqe, linestyle="none", label=f"VQE {label}", **style)
            ax_d.plot(d.bond_length, _abs_floor(d.delta_e), label=label, **style)
        ax_e.set(
            xlabel="bond length R (Å)", ylabel="energy (Ha)", title=f"{molecule} dissociation curve"
        )
        ax_e.legend()
        ax_d.set_yscale("log")
        _chem_acc_line(ax_d)
        ax_d.set(
            xlabel="bond length R (Å)",
            ylabel="|E_VQE − E_exact| (Ha)",
            title=f"VQE error ({space})",
        )
        ax_d.legend(loc="lower right")
        fig.tight_layout()
        save_figure(fig, figdir / f"{config['name']}_{space}")


# ----------------------------------------------------------------------------- E4 trade-off


def run_tradeoff(config: dict) -> pd.DataFrame:
    """E4: combine the E2 and E3 curves into depth-vs-accuracy and the claim fields."""
    from qcc_common.plotting import FIGSIZE_SINGLE, MUTED, save_figure, series_style

    df = pd.concat([load_summary(config, name) for name in config["inputs"]], ignore_index=True)
    df["abs_delta_e"] = df.delta_e.abs()
    near_r = float(config.get("near_dissociation_min_r", 2.5))
    eq_r = float(config.get("equilibrium_max_r", 2.0))

    table = (
        df.groupby(["active_space", "ansatz"], sort=False)
        .agg(
            num_qubits=("num_qubits", "first"),
            num_parameters=("num_parameters", "first"),
            logical_depth=("logical_depth", "first"),
            depth=("depth", "first"),
            two_qubit_gates=("two_qubit_gates", "first"),
            points=("bond_length", "size"),
            points_chemical_accuracy=("chemical_accuracy", "sum"),
            median_delta_e=("abs_delta_e", "median"),
            max_delta_e=("abs_delta_e", "max"),
            wrong_sector_points=("wrong_sector", "sum"),
            mean_correlation_recovered=("correlation_recovered", "mean"),
        )
        .reset_index()
    )
    near = df[df.bond_length >= near_r].groupby(["active_space", "ansatz"]).chemical_accuracy.mean()
    eq = df[df.bond_length <= eq_r].groupby(["active_space", "ansatz"]).chemical_accuracy.mean()
    table["fraction_chem_acc_r_le_eq"] = [
        eq.get((s, a)) for s, a in zip(table.active_space, table.ansatz, strict=True)
    ]
    table["fraction_chem_acc_r_ge_near"] = [
        near.get((s, a)) for s, a in zip(table.active_space, table.ansatz, strict=True)
    ]

    claims = {}
    for space, t in table.groupby("active_space", sort=False):
        u = t[t.ansatz.str.startswith("uccsd")].iloc[0]
        hea = t[t.ansatz.str.startswith("hea")].sort_values(
            ["points_chemical_accuracy", "depth"], ascending=[False, True]
        )
        h = hea.iloc[0]
        claims[space] = {
            "X": int(u.points_chemical_accuracy),
            "Y": int(u.points),
            "Z": int(u.depth),
            "X_prime": int(h.points_chemical_accuracy),
            "Z_prime": int(h.depth),
            "hea_best": h.ansatz,
            "hea_fraction_near_dissociation": float(h.fraction_chem_acc_r_ge_near),
            "hea_fraction_up_to_eq": float(h.fraction_chem_acc_r_le_eq),
            "hea_fails_near_dissociation": bool(
                h.fraction_chem_acc_r_ge_near < h.fraction_chem_acc_r_le_eq
            ),
            "near_dissociation_min_r": near_r,
            "per_hea_reps": {
                row.ansatz: {"X_prime": int(row.points_chemical_accuracy), "depth": int(row.depth)}
                for row in hea.itertuples()
            },
        }

    _, summ, figdir = _dirs(config)
    table.to_csv(summ / f"{config['name']}.csv", index=False, float_format="%.6g")
    (summ / f"{config['name']}_claim.json").write_text(json.dumps(claims, indent=1))
    ExperimentRecord(experiment="e4", config=config, seed=None, payload=claims, metrics={}).save(
        Path(config["results_dir"]) / "raw" / f"{config['name']}.json"
    )

    plt = _plt()
    for space, t in table.groupby("active_space", sort=False):
        d = df[df.active_space == space]
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        for k, row in enumerate(t.itertuples()):
            vals = d[d.ansatz == row.ansatz].abs_delta_e
            style = series_style(_slot(ANSATZ_SLOTS, row.ansatz, k))
            lo, med, hi = (np.maximum(np.quantile(vals, q), DELTA_E_FLOOR) for q in (0.0, 0.5, 1.0))
            ax.errorbar(
                row.depth,
                med,
                yerr=[[med - lo], [hi - med]],
                capsize=3,
                color=style["color"],
                marker=style["marker"],
                linestyle="none",
                markersize=7,
                markeredgecolor=style["markeredgecolor"],
                label=row.ansatz,
            )
            ax.annotate(
                row.ansatz,
                (row.depth, med),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=MUTED,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        _chem_acc_line(ax)
        ax.set(
            xlabel="transpiled depth (linear coupling, {rz, sx, x, ecr})",
            ylabel="|ΔE| over the curve (Ha)",
            title=f"LiH ({space}): error vs depth — median, min–max over "
            f"{int(t.points.iloc[0])} points",
        )
        ax.legend(loc="lower left")
        save_figure(fig, figdir / f"{config['name']}_error_vs_depth_{space}")

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        for k, (label, dd) in enumerate(d.groupby("ansatz", sort=False)):
            dd = dd.sort_values("bond_length")
            ax.plot(
                dd.bond_length,
                _abs_floor(dd.delta_e),
                label=label,
                **series_style(_slot(ANSATZ_SLOTS, label, k)),
            )
        ax.set_yscale("log")
        _chem_acc_line(ax)
        ax.set(
            xlabel="bond length R (Å)",
            ylabel="|E_VQE − E_exact| (Ha)",
            title=f"LiH ({space}): error vs distance",
        )
        ax.legend(loc="lower right")
        save_figure(fig, figdir / f"{config['name']}_error_vs_distance_{space}")
    log(json.dumps(claims, indent=1))
    return table


# ---------------------------------------------------------------- E5 optimizers and inits


def _neighbour_toward(grid: list[float], r: float, equilibrium: float) -> float:
    """Grid neighbour of ``r`` on the equilibrium side (the point below, for r = equilibrium)."""
    grid = sorted(grid)
    i = grid.index(r)
    if r > equilibrium + 1e-9:
        return grid[i - 1]
    if r < equilibrium - 1e-9:
        return grid[i + 1]
    return grid[i - 1]


def _single_vqe_job(job: dict) -> dict:
    from vqe_lih.ansatze import AnsatzSpec, build_ansatz, compile_for_simulation
    from vqe_lih.hamiltonian import ActiveSpace, build_hamiltonian
    from vqe_lih.reference import exact_energy
    from vqe_lih.solver import initial_point, run_vqe

    ham = build_hamiltonian(
        job["molecule"],
        job["bond_length"],
        job["basis"],
        ActiveSpace.from_dict(job["active_space"]),
    )
    spec = AnsatzSpec.from_dict(job["ansatz"])
    ansatz = compile_for_simulation(build_ansatz(spec, ham))
    x0 = initial_point(job["init"], ansatz.num_parameters, seed=job["seed"], warm=job.get("warm"))
    estimator = None
    if job.get("shots"):
        from vqe_lih.measurement import ShotEstimator

        estimator = ShotEstimator(job["shots"], seed=job["seed"])
    e_exact = exact_energy(ham).energy
    record = run_vqe(
        ham,
        ansatz,
        job["optimizer"],
        x0,
        estimator=estimator,
        e_exact=e_exact,
        init=job["init"],
        seed=job["seed"],
        maxiter=job.get("maxiter"),
    )
    row = {
        "bond_length": job["bond_length"],
        "ansatz": spec.label,
        "num_parameters": ansatz.num_parameters,
        "e_exact": e_exact,
        **record.summary(),
        "estimate_error": record.energy_estimate - e_exact,
        "parameters": record.parameters,
        "history": record.history,
    }
    if estimator is not None:
        row["shots"] = job["shots"]
        row["total_shots"] = estimator.total_shots
    log(
        f"[{job['tag']}] {spec.label} R={job['bond_length']} {job['optimizer']} {job['init']} "
        f"seed={job['seed']} dE={record.delta_e:.2e} nfev={record.nfev}"
    )
    return row


def run_optimizers(config: dict) -> pd.DataFrame:
    """E5: optimizer x initialization grid at representative bond lengths."""
    space = config["active_space"]
    grid = [float(r) for r in config["bond_lengths"]]
    jobs = []
    for ansatz in config["ansatze"]:
        from vqe_lih.ansatze import AnsatzSpec

        label = AnsatzSpec.from_dict(ansatz).label
        source = {
            (row["active_space"], row["ansatz"], round(row["bond_length"], 6)): row["parameters"]
            for row in load_raw_rows(config, ansatz["source"])
        }
        for r, opt in itertools.product(config["points"], config["optimizers"]):
            warm = source[
                (
                    space["label"],
                    label,
                    round(_neighbour_toward(grid, float(r), config["equilibrium"]), 6),
                )
            ]
            inits = [("zeros", config["seed"], None), ("warm_start", config["seed"], warm)]
            inits += [("random", config["seed"] + i, None) for i in range(config["random_starts"])]
            for init, seed, w in inits:
                jobs.append(
                    {
                        "tag": "e5",
                        "molecule": config["molecule"],
                        "basis": config.get("basis", "sto3g"),
                        "active_space": space,
                        "ansatz": ansatz,
                        "bond_length": float(r),
                        "optimizer": opt,
                        "init": init,
                        "seed": seed,
                        "warm": w,
                        "maxiter": config.get("maxiter", {}).get(opt),
                    }
                )
    jobs.sort(key=lambda j: (j["ansatz"]["type"] != "hea", j["optimizer"] != "l_bfgs_b"))
    rows = _parallel_map(_single_vqe_job, jobs, config.get("workers", 1))
    runs = pd.DataFrame([_strip(r) for r in rows])
    runs["abs_delta_e"] = runs.delta_e.abs()
    stats = (
        runs.groupby(["ansatz", "bond_length", "optimizer", "init"])
        .agg(
            runs=("delta_e", "size"),
            median_delta_e=("abs_delta_e", "median"),
            min_delta_e=("abs_delta_e", "min"),
            max_delta_e=("abs_delta_e", "max"),
            fraction_chem_acc=("chemical_accuracy", "mean"),
            median_nfev=("nfev", "median"),
            median_nit=("nit", "median"),
            median_evals_to_chem_acc=("evals_to_chemical_accuracy", "median"),
            wrong_sector_runs=("wrong_sector", "sum"),
        )
        .reset_index()
    )
    _save(config, rows, runs)
    _, summ, _ = _dirs(config)
    stats.to_csv(summ / f"{config['name']}_stats.csv", index=False, float_format="%.6g")
    plot_optimizers(config, runs, rows)
    return stats


def plot_optimizers(config: dict, runs: pd.DataFrame, rows: list[dict]) -> None:
    from qcc_common.plotting import MUTED, SERIES_COLORS, SERIES_MARKERS, save_figure, series_style

    plt = _plt()
    _, _, figdir = _dirs(config)
    points = sorted(runs.bond_length.unique())
    optimizers = list(config["optimizers"])
    inits = ["zeros", "warm_start", "random"]
    for label, d in runs.groupby("ansatz", sort=False):
        fig, axes = plt.subplots(1, len(points), figsize=(3.4 * len(points), 3.8), sharey=True)
        for ax, r in zip(np.atleast_1d(axes), points, strict=True):
            dr = d[d.bond_length == r]
            for i, opt in enumerate(optimizers):
                for j, init in enumerate(inits):
                    vals = _abs_floor(dr[(dr.optimizer == opt) & (dr.init == init)].abs_delta_e)
                    x = i + (j - 1) * 0.25 + np.linspace(-0.05, 0.05, len(vals))
                    slot = INIT_SLOTS[init]
                    ax.scatter(
                        x,
                        vals,
                        s=22,
                        color=SERIES_COLORS[slot],
                        marker=SERIES_MARKERS[slot],
                        edgecolors="#fcfcfb",
                        linewidths=0.6,
                        label=init.replace("_", " ") if (i == 0 and r == points[0]) else None,
                    )
            ax.set_xticks(range(len(optimizers)), [o.upper().replace("_", "-") for o in optimizers])
            ax.set_yscale("log")
            _chem_acc_line(ax, label=r == points[-1])
            ax.set_title(f"R = {r} Å")
        np.atleast_1d(axes)[0].set_ylabel("|ΔE| final (Ha)")
        fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
        fig.suptitle(
            f"E5 — {label}: final error by optimizer and initialization "
            f"({config['random_starts']} random seeds)",
            y=1.10,
            fontsize=10,
            color=MUTED,
        )
        save_figure(fig, figdir / f"{config['name']}_{label}")

    # Convergence histories at the hardest point, zero init (UCCSD) / first random seed (HEA).
    r = max(points)
    fig, axes = plt.subplots(1, runs.ansatz.nunique(), figsize=(10, 3.8), sharey=True)
    for ax, label in zip(np.atleast_1d(axes), runs.ansatz.unique(), strict=True):
        init = "zeros" if label.startswith("uccsd") else "random"
        for opt in optimizers:
            sel = [
                row
                for row in rows
                if row["ansatz"] == label
                and row["bond_length"] == r
                and row["optimizer"] == opt
                and row["init"] == init
            ]
            if not sel:
                continue
            row = min(sel, key=lambda rr: rr["seed"])
            h = _abs_floor(np.asarray(row["history"]) - row["e_exact"])
            ax.plot(
                np.arange(1, len(h) + 1),
                np.minimum.accumulate(h),
                label=opt.upper().replace("_", "-"),
                **{**series_style(OPTIMIZER_SLOTS[opt]), "marker": None},
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="energy evaluations",
            title=f"{label}, R = {r} Å, {init} init",
        )
        _chem_acc_line(ax, label=label == runs.ansatz.unique()[-1])
        ax.legend(loc="lower left")
    np.atleast_1d(axes)[0].set_ylabel("best |ΔE| so far (Ha)")
    save_figure(fig, figdir / f"{config['name']}_histories")


# ------------------------------------------------------------------------------ E6 shots


def run_shots(config: dict) -> pd.DataFrame:
    """E6: VQE with a shot-based estimator; error vs shots per measurement group."""
    from qcc_common.plotting import FIGSIZE_SINGLE, save_figure, series_style

    jobs = [
        {
            "tag": "e6",
            "molecule": config["molecule"],
            "basis": config.get("basis", "sto3g"),
            "active_space": config["active_space"],
            "ansatz": config["ansatz"],
            "bond_length": float(config["point"]),
            "optimizer": opt,
            "init": config.get("init", "zeros"),
            "seed": config["seed"] + i,
            "shots": int(shots),
            "maxiter": config.get("maxiter", {}).get(opt),
        }
        for shots in config["shots"]
        for opt in config["optimizers"]
        for i in range(config["repetitions"])
    ]
    jobs.sort(key=lambda j: -j["shots"])
    rows = _parallel_map(_single_vqe_job, jobs, config.get("workers", 1))
    runs = pd.DataFrame([_strip(r) for r in rows])
    runs["abs_delta_e"] = runs.delta_e.abs()
    stats = (
        runs.groupby(["optimizer", "shots"])
        .agg(
            runs=("delta_e", "size"),
            median_delta_e=("abs_delta_e", "median"),
            p16_delta_e=("abs_delta_e", lambda v: np.quantile(v, 0.16)),
            p84_delta_e=("abs_delta_e", lambda v: np.quantile(v, 0.84)),
            fraction_chem_acc=("chemical_accuracy", "mean"),
            median_abs_estimate_error=("estimate_error", lambda v: np.median(np.abs(v))),
            median_nfev=("nfev", "median"),
            median_total_shots=("total_shots", "median"),
        )
        .reset_index()
    )
    needed = {}
    for opt, s in stats.groupby("optimizer"):
        ok = s[s.median_delta_e < CHEM_ACC_MHA * 1e-3].sort_values("shots")
        needed[opt] = (
            None
            if ok.empty
            else {
                "shots_per_group": int(ok.shots.iloc[0]),
                "median_total_shots": float(ok.median_total_shots.iloc[0]),
            }
        )
    _save(config, rows, runs, extra={"shots_needed_for_chemical_accuracy": needed})
    _, summ, figdir = _dirs(config)
    stats.to_csv(summ / f"{config['name']}_stats.csv", index=False, float_format="%.6g")
    (summ / f"{config['name']}_needed.json").write_text(json.dumps(needed, indent=1))

    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for opt, s in stats.groupby("optimizer", sort=False):
        style = series_style(OPTIMIZER_SLOTS[opt])
        ax.fill_between(
            s.shots,
            _abs_floor(s.p16_delta_e),
            _abs_floor(s.p84_delta_e),
            color=style["color"],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(s.shots, _abs_floor(s.median_delta_e), label=opt.upper().replace("_", "-"), **style)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="shots per measurement group per evaluation",
        ylabel="|ΔE| at final parameters (Ha)",
        title=f"E6 — shot noise, LiH {config['active_space']['label']} UCCSD at "
        f"R = {config['point']} Å (median, 16–84 %, {config['repetitions']} seeds)",
    )
    ax.title.set_fontsize(9)
    _chem_acc_line(ax)
    ax.legend(loc="lower left")
    save_figure(fig, figdir / config["name"])
    log(json.dumps(needed))
    return stats


# --------------------------------------------------------------------- E7 noise/mitigation


def _noise_job(job: dict) -> dict:
    from qcc_common.backends import noisy_simulator
    from vqe_lih.ansatze import AnsatzSpec, build_ansatz, compile_for_simulation
    from vqe_lih.hamiltonian import ActiveSpace, build_hamiltonian
    from vqe_lih.mitigation import noisy_energy
    from vqe_lih.reference import exact_energy
    from vqe_lih.solver import exact_expectations

    ham = build_hamiltonian(
        job["molecule"],
        job["bond_length"],
        job["basis"],
        ActiveSpace.from_dict(job["active_space"]),
    )
    spec = AnsatzSpec.from_dict(job["ansatz"])
    ansatz = compile_for_simulation(build_ansatz(spec, ham))
    params = np.asarray(job["parameters"])
    e_exact = exact_energy(ham).energy
    (ideal,) = exact_expectations(ansatz, [ham.qubit_op], params)
    backend = noisy_simulator(job["fake_backend"], scale=job["noise_scale"], seed=job["seed"])
    res = noisy_energy(
        ansatz.assign_parameters(params),
        ham.qubit_op,
        backend,
        job["shots"],
        job["seed"],
        job["scale_factors"],
    )
    shift = ham.energy_shift
    row = {
        "bond_length": job["bond_length"],
        "ansatz": spec.label,
        "e_exact": e_exact,
        "ideal": ideal + shift,
        "noisy_raw": res["raw"][1] + shift,
        "readout": res["readout"][1] + shift,
        "zne_linear": res["zne_linear"] + shift,
        "zne_richardson": res["zne_richardson"] + shift,
        "readout_zne_linear": res["readout_zne_linear"] + shift,
        "readout_zne_richardson": res["readout_zne_richardson"] + shift,
        "std_raw": res["std_raw"][1],
        "device_depth": res["device_depth"],
        "device_two_qubit_gates": res["device_two_qubit_gates"],
        "num_groups": res["num_groups"],
        "readout_error_mean": res["readout_error_mean"],
        "raw_by_scale": {str(k): v + shift for k, v in res["raw"].items()},
        "readout_by_scale": {str(k): v + shift for k, v in res["readout"].items()},
        "hardware": "not run (no IBM Quantum access in this environment)",
    }
    log(
        f"[e7] {spec.label} R={job['bond_length']} raw dE={row['noisy_raw'] - e_exact:.3e} "
        f"readout+ZNE dE={row['readout_zne_linear'] - e_exact:.3e}"
    )
    return row


METHODS = [
    "ideal",
    "noisy_raw",
    "readout",
    "zne_linear",
    "readout_zne_linear",
    "readout_zne_richardson",
]


def run_noise(config: dict) -> pd.DataFrame:
    """E7 (optional): energies at simulator-optimized parameters on a noisy fake-backend
    simulator — raw, readout-mitigated and ZNE. Real hardware is not used here."""
    from qcc_common.plotting import FIGSIZE_WIDE, SERIES_COLORS, save_figure

    space = config["active_space"]
    jobs = []
    for ansatz in config["ansatze"]:
        from vqe_lih.ansatze import AnsatzSpec

        label = AnsatzSpec.from_dict(ansatz).label
        source = {
            (row["active_space"], row["ansatz"], round(row["bond_length"], 6)): row["parameters"]
            for row in load_raw_rows(config, ansatz["source"])
        }
        for r in config["points"]:
            jobs.append(
                {
                    "molecule": config["molecule"],
                    "basis": config.get("basis", "sto3g"),
                    "active_space": space,
                    "ansatz": ansatz,
                    "bond_length": float(r),
                    "parameters": source[(space["label"], label, round(float(r), 6))],
                    "fake_backend": config.get("fake_backend", "FakeSherbrooke"),
                    "noise_scale": float(config.get("noise_scale", 1.0)),
                    "shots": int(config["shots"]),
                    "seed": int(config["seed"]),
                    "scale_factors": list(config.get("scale_factors", [1, 3, 5])),
                }
            )
    rows = _parallel_map(_noise_job, jobs, config.get("workers", 1))
    table = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.endswith("_by_scale")} for r in rows]
    )
    for m in METHODS:
        table[f"delta_e_{m}"] = table[m] - table.e_exact
    _save(config, rows, table)

    plt = _plt()
    _, _, figdir = _dirs(config)
    labels = list(dict.fromkeys(table.ansatz))
    fig, axes = plt.subplots(1, len(labels), figsize=FIGSIZE_WIDE, sharey=True)
    width = 0.8 / len(METHODS)
    for ax, label in zip(np.atleast_1d(axes), labels, strict=True):
        t = table[table.ansatz == label].sort_values("bond_length")
        x = np.arange(len(t))
        for k, m in enumerate(METHODS):
            ax.bar(
                x + (k - (len(METHODS) - 1) / 2) * width,
                _abs_floor(t[f"delta_e_{m}"]),
                width=width * 0.9,
                color=SERIES_COLORS[k],
                label=m.replace("_", " "),
            )
        ax.set_xticks(x, [f"{r} Å" for r in t.bond_length])
        ax.set_yscale("log")
        _chem_acc_line(ax, label=label == labels[-1])
        ax.set_title(f"{label} (device 2q gates: {int(t.device_two_qubit_gates.iloc[0])})")
    np.atleast_1d(axes)[0].set_ylabel("|E − E_exact| (Ha)")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(
        f"E7 — {config.get('fake_backend', 'FakeSherbrooke')} noisy simulator, "
        f"{config['shots']} shots/group",
        fontsize=10,
    )
    save_figure(fig, figdir / config["name"])
    return table


RUNNERS: dict[str, Callable[[dict], pd.DataFrame]] = {
    "e1": run_curve,
    "e2": run_curve,
    "e3": run_curve,
    "e4": run_tradeoff,
    "e5": run_optimizers,
    "e6": run_shots,
    "e7": run_noise,
    "beh2": run_curve,
}
