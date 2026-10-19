"""Experiment runners E1-E5 (SPEC §7), driven by TOML configs.

Each run writes ``results/raw/<name>.json`` (gitignored), ``results/summary/<name>*.csv|json``
and ``results/figures/<name>*.{png,pdf}``. Prices and errors are in price units.
"""

from __future__ import annotations

import json
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qcc_common.parallel import parallel_map
from qcc_common.results import ExperimentRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
METHOD_SLOTS = {"iqae": 0, "mlae": 1, "qae": 2, "qae_mle": 3, "mc": 4, "qmc": 5}
METHOD_LABELS = {
    "iqae": "IQAE",
    "mlae": "MLAE",
    "qae": "QAE (grid)",
    "qae_mle": "QAE (MLE)",
    "mc": "Monte Carlo",
    "qmc": "quasi-MC (Sobol)",
}
QUANTUM = ("qae", "iqae", "mlae")


# ---------------------------------------------------------------------------- config & I/O


def load_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        config = tomllib.load(f)
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


def _parts(config: dict) -> Path:
    return Path(config["results_dir"]) / "raw" / f"{config['name']}.parts"


def _save(config: dict, payload: Any, tables: dict[str, pd.DataFrame], extra: dict | None = None):
    raw, summ, _ = _dirs(config)
    ExperimentRecord(
        experiment=config["experiment"],
        config=config,
        seed=config.get("seed"),
        payload=payload,
        metrics=extra or {},
    ).save(raw / f"{config['name']}.json")
    for suffix, table in tables.items():
        table.to_csv(summ / f"{config['name']}{suffix}.csv", index=False, float_format="%.8g")


def _load_table(config: dict, name: str) -> pd.DataFrame:
    path = Path(config["results_dir"]) / "summary" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run the experiment that produces it first")
    return pd.read_csv(path)


# --------------------------------------------------------------------------------- models


def build_distribution(model: dict):
    from qae_options.distributions import BimodalLogNormal, LogNormal

    if model["distribution"] == "lognormal":
        return LogNormal(model["s0"], model["r"], model["sigma"], model["maturity"])
    if model["distribution"] == "bimodal":
        return BimodalLogNormal(
            model["w"],
            np.log(model["center1"]),
            model["sigma1"],
            np.log(model["center2"]),
            model["sigma2"],
        )
    raise ValueError(f"unknown distribution {model['distribution']!r}")


def build_problem(model: dict, payoff: dict, n: int):
    """(PricingProblem, continuous reference price, quadrature check price)."""
    from qae_options.distributions import discretize
    from qae_options.payoffs import digital_call, european_call
    from qae_options.reference import mixture_call, mixture_digital, quad_price

    dist = build_distribution(model)
    grid = discretize(dist, n)
    r, t, k = model["r"], model["maturity"], payoff["strike"]
    if payoff["type"] == "digital":
        q = payoff.get("payout", 1.0)
        problem = digital_call(grid, k, r, t, q)
        continuous = mixture_digital(dist, k, r, t, q)
        check = quad_price(dist, lambda s: q * float(s >= k), r, t)
    elif payoff["type"] == "call":
        problem = european_call(grid, k, r, t, payoff["rescaling_factor"])
        continuous = mixture_call(dist, k, r, t)
        check = quad_price(dist, lambda s: max(s - k, 0.0), r, t)
    else:
        raise ValueError(f"unknown payoff {payoff['type']!r}")
    return problem, continuous, check


def _quantum_reps(problem, method: str, knob: float, shots: int, seeds: list[int], backend=None):
    """Run one estimator setting for each seed, sharing circuit/distribution caches."""
    from qae_options.estimators import run_estimator
    from qae_options.sampling import ExactDistributionSampler

    cache: dict = {}
    rows = []
    for seed in seeds:
        sampler = ExactDistributionSampler(shots, seed=seed, backend=backend, cache=cache)
        res = run_estimator(problem, method, knob, sampler)
        base = {
            "knob": knob,
            "seed": seed,
            "queries": res.oracle_queries,
            "max_k": max(res.grover_powers),
        }
        rows.append(
            {
                **base,
                "method": method,
                "estimate": res.estimate,
                "error": abs(res.estimate - problem.exact_price),
            }
        )
        if method == "qae":
            mle = problem.to_price(res.extra["mle_amplitude"])
            rows.append(
                {
                    **base,
                    "method": "qae_mle",
                    "estimate": mle,
                    "error": abs(mle - problem.exact_price),
                }
            )
    return rows


# ------------------------------------------------------------------------------ E1 validate


def _e1_job(job: dict) -> list[dict]:
    problem, continuous, check = build_problem(job["model"], job["payoff"], job["n"])
    seeds = [job["seed"] + i for i in range(job["repetitions"])]
    rows = _quantum_reps(problem, job["method"], job["knob"], job["shots"], seeds)
    out = []
    for method in sorted({r["method"] for r in rows}):
        est = np.array([r["estimate"] for r in rows if r["method"] == method])
        out.append(
            {
                "payoff": job["payoff"]["type"],
                "rescaling_factor": job["payoff"].get("rescaling_factor"),
                "n": job["n"],
                "method": method,
                "knob": job["knob"],
                "repetitions": len(est),
                "estimate_median": float(np.median(est)),
                "estimate_p16": float(np.quantile(est, 0.16)),
                "estimate_p84": float(np.quantile(est, 0.84)),
                "exact_a_price": problem.exact_price,
                "discretized_exact": problem.discretized_price,
                "continuous": continuous,
                "quad_check": check,
                "estimation_error": float(np.median(np.abs(est - problem.exact_price))),
                "payoff_approximation_error": problem.exact_price - problem.discretized_price,
                "discretization_error": problem.discretized_price - continuous,
                "num_qubits": problem.a_operator.num_qubits,
            }
        )
    log(f"[e1] {job['payoff']['type']} n={job['n']} {job['method']}={job['knob']} done")
    return out


def _e1_barrier_job(job: dict) -> list[dict]:
    from qae_options.distributions import LogNormal, discretize_two_dates
    from qae_options.payoffs import barrier_digital
    from qae_options.reference import barrier_digital_price

    m, b = job["model"], job["barrier"]
    dist = LogNormal(m["s0"], m["r"], m["sigma"], m["maturity"])
    joint = discretize_two_dates(dist, b["n"], b["t1"])
    problem = barrier_digital(
        joint, b["barrier"], b["strike"], m["r"], m["maturity"], b.get("payout", 1.0)
    )
    continuous = barrier_digital_price(
        dist, b["barrier"], b["strike"], b["t1"], b.get("payout", 1.0)
    )
    seeds = [job["seed"] + i for i in range(job["repetitions"])]
    rows = _quantum_reps(problem, job["method"], job["knob"], job["shots"], seeds)
    est = np.array([r["estimate"] for r in rows])
    log(f"[e1] barrier {job['method']} done")
    return [
        {
            "payoff": "barrier_digital",
            "rescaling_factor": None,
            "n": b["n"],
            "method": job["method"],
            "knob": job["knob"],
            "repetitions": len(est),
            "estimate_median": float(np.median(est)),
            "estimate_p16": float(np.quantile(est, 0.16)),
            "estimate_p84": float(np.quantile(est, 0.84)),
            "exact_a_price": problem.exact_price,
            "discretized_exact": problem.discretized_price,
            "continuous": continuous,
            "quad_check": None,
            "estimation_error": float(np.median(np.abs(est - problem.exact_price))),
            "payoff_approximation_error": 0.0,
            "discretization_error": problem.discretized_price - continuous,
            "num_qubits": problem.a_operator.num_qubits,
        }
    ]


def run_validate(config: dict) -> pd.DataFrame:
    """E1: does the pipeline price correctly? Lognormal, call (per c) and digital, n = 3-5."""
    base = {
        "model": config["model"],
        "shots": config["shots"],
        "seed": config["seed"],
        "repetitions": config["repetitions"],
    }
    jobs = [
        {**base, "payoff": payoff, "n": n, "method": method, "knob": knob}
        for payoff in config["payoffs"]
        for n in config["n_values"]
        for method, knob in config["estimators"]
    ]
    rows = [
        r
        for res in parallel_map(_e1_job, jobs, config.get("workers", 1), _parts(config))
        for r in res
    ]
    if "barrier" in config:
        bjobs = [
            {**base, "barrier": config["barrier"], "method": method, "knob": knob}
            for method, knob in config["barrier"]["estimators"]
        ]
        rows += [
            r
            for res in parallel_map(
                _e1_barrier_job, bjobs, config.get("workers", 1), _parts(config)
            )
            for r in res
        ]
    table = pd.DataFrame(rows)
    _save(config, rows, {"": table})
    return table


# ---------------------------------------------------------------------------- E2 convergence


def _e2_job(job: dict) -> list[dict]:
    from qae_options.classical import monte_carlo, quasi_monte_carlo
    from qae_options.reference import mixture_digital

    problem, continuous, _ = build_problem(job["model"], job["payoff"], job["n"])
    seeds = [job["seed"] + i for i in range(job["rep_start"], job["rep_start"] + job["rep_count"])]
    method, knob = job["method"], job["knob"]
    if method in QUANTUM:
        backend = None
        if job.get("noise_scale") is not None:
            from qcc_common.backends import noisy_simulator

            backend = noisy_simulator(job["fake_backend"], scale=job["noise_scale"], seed=seeds[0])
        rows = _quantum_reps(problem, method, knob, job["shots"], seeds, backend)
    else:
        m, p = job["model"], job["payoff"]
        discount = np.exp(-m["r"] * m["maturity"]) * p.get("payout", 1.0)
        rows = []
        if method == "mc":
            grid = _grid(job)
            values = (grid.x >= p["strike"]).astype(float)
            for seed in seeds:
                est = monte_carlo(grid, values, int(knob), np.random.default_rng(seed), discount)
                rows.append(
                    {
                        "method": "mc",
                        "knob": knob,
                        "seed": seed,
                        "queries": int(knob),
                        "max_k": 0,
                        "estimate": est,
                        "error": abs(est - problem.exact_price),
                    }
                )
        elif method == "qmc":
            dist = build_distribution(m)
            target = mixture_digital(dist, p["strike"], m["r"], m["maturity"], p.get("payout", 1.0))
            for seed in seeds:
                est = quasi_monte_carlo(
                    dist, lambda s: (s >= p["strike"]).astype(float), int(knob), seed, discount
                )
                rows.append(
                    {
                        "method": "qmc",
                        "knob": knob,
                        "seed": seed,
                        "queries": int(knob),
                        "max_k": 0,
                        "estimate": est,
                        "error": abs(est - target),
                    }
                )
        else:
            raise ValueError(method)
    for r in rows:
        r["noise_scale"] = job.get("noise_scale")
    log(
        f"[{job['tag']}] {method}={knob} reps {job['rep_start']}+{job['rep_count']} "
        f"median err {np.median([r['error'] for r in rows]):.2e}"
    )
    return rows


def _grid(job):
    from qae_options.distributions import discretize

    return discretize(build_distribution(job["model"]), job["n"])


def _convergence_jobs(config: dict, tag: str, noise_scale=None) -> list[dict]:
    reps, chunk = config["repetitions"], config.get("chunk", config["repetitions"])
    base = {
        "tag": tag,
        "model": config["model"],
        "payoff": config["payoff"],
        "n": config["n"],
        "shots": config["shots"],
        "seed": config["seed"],
        "noise_scale": noise_scale,
        "fake_backend": config.get("fake_backend", "FakeSherbrooke"),
    }
    jobs = []
    for method, knobs in config["sweeps"].items():
        if noise_scale is not None and method not in QUANTUM:
            continue
        for knob in knobs:
            for start in range(0, reps, chunk):
                jobs.append(
                    {
                        **base,
                        "method": method,
                        "knob": knob,
                        "rep_start": start,
                        "rep_count": min(chunk, reps - start),
                    }
                )
    # Expensive settings first (MLAE's likelihood grid grows with 2^M).
    jobs.sort(key=lambda j: -(2 ** j["knob"] if j["method"] == "mlae" else 1))
    return jobs


def summarize(runs: pd.DataFrame, by=("method", "knob")) -> pd.DataFrame:
    return (
        runs.groupby(list(by), dropna=False)
        .agg(
            runs=("error", "size"),
            median_error=("error", "median"),
            p16_error=("error", lambda v: np.quantile(v, 0.16)),
            p84_error=("error", lambda v: np.quantile(v, 0.84)),
            median_queries=("queries", "median"),
            median_max_k=("max_k", "median"),
        )
        .reset_index()
    )


def fit_slope(stats: pd.DataFrame, min_queries: float = 0) -> dict:
    """Least-squares slope of log10(median error) vs log10(median queries), with std error."""
    s = stats[(stats.median_error > 0) & (stats.median_queries >= min_queries)]
    if len(s) < 3:
        return {"slope": None, "slope_std": None, "intercept": None, "points": len(s)}
    (slope, intercept), cov = np.polyfit(
        np.log10(s.median_queries), np.log10(s.median_error), 1, cov=True
    )
    return {
        "slope": float(slope),
        "slope_std": float(np.sqrt(cov[0, 0])),
        "intercept": float(intercept),
        "points": len(s),
    }


def queries_for(stats: pd.DataFrame, fit: dict, eps: float) -> dict:
    """Queries to reach median error ``eps``: log-log interpolation where the curve crosses
    eps, else extrapolation of the fit."""
    s = stats.sort_values("median_queries")
    q, e = s.median_queries.to_numpy(float), s.median_error.to_numpy(float)
    for i in range(1, len(s)):
        if e[i - 1] > eps >= e[i] > 0:
            t = (np.log10(eps) - np.log10(e[i - 1])) / (np.log10(e[i]) - np.log10(e[i - 1]))
            return {
                "queries": float(
                    10 ** (np.log10(q[i - 1]) + t * (np.log10(q[i]) - np.log10(q[i - 1])))
                ),
                "how": "interpolated",
            }
    if fit.get("slope"):
        return {
            "queries": float(10 ** ((np.log10(eps) - fit["intercept"]) / fit["slope"])),
            "how": "extrapolated from fit",
        }
    return {"queries": None, "how": "not reached"}


def run_convergence(config: dict) -> pd.DataFrame:
    """E2: estimation error vs oracle queries, R repetitions per setting."""
    jobs = _convergence_jobs(config, "e2")
    rows = [
        r
        for res in parallel_map(_e2_job, jobs, config.get("workers", 1), _parts(config))
        for r in res
    ]
    runs = pd.DataFrame(rows)
    stats = summarize(runs)
    eps = config.get("epsilon", 1e-3)
    fits, needed = {}, {}
    for method, s in stats.groupby("method"):
        fits[method] = fit_slope(s, config.get("fit_min_queries", {}).get(method, 0))
        needed[method] = queries_for(s, fits[method], eps)
    _save(
        config,
        rows,
        {"": runs, "_stats": stats},
        extra={"fits": fits, "queries_for_epsilon": needed},
    )
    _, summ, figdir = _dirs(config)
    (summ / f"{config['name']}_fits.json").write_text(
        json.dumps({"epsilon": eps, "fits": fits, "queries_for_epsilon": needed}, indent=1)
    )
    plot_convergence(
        stats,
        fits,
        eps,
        figdir / config["name"],
        f"E2 — bimodal digital, n = {config['n']}: error vs oracle queries "
        f"(median, 16–84 %, R = {config['repetitions']})",
    )
    log(json.dumps(needed))
    return stats


def plot_convergence(stats, fits, eps, stem, title, methods=None):
    import matplotlib.pyplot as plt

    from qcc_common.plotting import (
        FIGSIZE_SINGLE,
        MUTED,
        REFERENCE,
        apply_style,
        save_figure,
        series_style,
    )

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in methods or [m for m in METHOD_SLOTS if m in set(stats.method)]:
        s = stats[stats.method == method].sort_values("median_queries")
        if s.empty:
            continue
        style = series_style(METHOD_SLOTS[method])
        floor = 1e-7
        ax.fill_between(
            s.median_queries,
            np.maximum(s.p16_error, floor),
            np.maximum(s.p84_error, floor),
            color=style["color"],
            alpha=0.12,
            linewidth=0,
        )
        f = fits.get(method, {})
        slope = (
            f" (slope {f['slope']:.2f} ± {f['slope_std']:.2f})"
            if f.get("slope") is not None
            else ""
        )
        ax.plot(
            s.median_queries,
            np.maximum(s.median_error, floor),
            label=METHOD_LABELS[method] + slope,
            **style,
        )
    ax.axhline(eps, color=REFERENCE, linestyle="--", linewidth=1.0)
    ax.annotate(
        f"ε = {eps:g}",
        xy=(0.0, eps),
        xycoords=("axes fraction", "data"),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=7.5,
        color=MUTED,
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="oracle queries (A or A† applications; MC: samples)",
        ylabel="|estimate − exact| (price units)",
        title=title,
    )
    ax.title.set_fontsize(9.5)
    ax.legend(loc="lower left", fontsize=7.5)
    del FIGSIZE_SINGLE
    save_figure(fig, stem)


# ----------------------------------------------------------------------------- E3 resources


def run_resources(config: dict) -> pd.DataFrame:
    """E3: qubits, depth and 2q gates per variant at the ε operating point from E2."""
    from qae_options import resources

    stats = _load_table(config, f"{config['source']}_stats")
    fits = json.loads(
        (Path(config["results_dir"]) / "summary" / f"{config['source']}_fits.json").read_text()
    )
    eps = fits["epsilon"]
    problem, _, _ = build_problem(config["model"], config["payoff"], config["n"])

    def first_reaching(method):
        s = stats[(stats.method == method) & (stats.median_error <= eps)].sort_values(
            "median_queries"
        )
        return None if s.empty else s.iloc[0]

    points = {m: first_reaching(m) for m in ("qae_mle", "qae", "iqae", "mlae")}
    max_m = int(stats[stats.method == "qae"].knob.max())
    m_star = (
        int(points["qae_mle"].knob)
        if points["qae_mle"] is not None
        else int(stats[stats.method == "qae"].knob.max())
    )
    costs = resources.component_costs(problem, max_eval_qubits=max(m_star, 1))
    rows = []
    for label, method in (("QAE (MLE post-processing)", "qae_mle"), ("QAE (grid)", "qae")):
        pt = points[method]
        m = int(pt.knob) if pt is not None else None
        if m is not None:
            rows.append(
                {
                    "variant": label,
                    "operating_point": f"m = {m}",
                    "reaches_epsilon": True,
                    "median_queries": pt.median_queries,
                    **resources.qae_cost(costs, m),
                }
            )
        else:
            rows.append(
                {
                    "variant": label,
                    "operating_point": f"not reached up to m = {max_m}",
                    "reaches_epsilon": False,
                    "median_queries": None,
                }
            )
    for label, method, knob_name in (("IQAE", "iqae", "ε_target"), ("MLAE", "mlae", "M")):
        pt = points[method]
        k = int(pt.median_max_k)
        rows.append(
            {
                "variant": label,
                "operating_point": f"{knob_name} = {pt.knob:g}, largest k = {k}",
                "reaches_epsilon": True,
                "median_queries": pt.median_queries,
                **resources.grover_power_cost(costs, k),
            }
        )
    rows.append(
        {
            "variant": "A operator alone",
            "operating_point": "k = 0",
            "reaches_epsilon": None,
            "median_queries": None,
            **resources.grover_power_cost(costs, 0),
        }
    )
    if "barrier" in config:
        from qae_options.distributions import LogNormal, discretize_two_dates
        from qae_options.payoffs import barrier_digital

        b, mm = config["barrier"], config["barrier_model"]
        dist = LogNormal(mm["s0"], mm["r"], mm["sigma"], mm["maturity"])
        bp = barrier_digital(
            discretize_two_dates(dist, b["n"], b["t1"]),
            b["barrier"],
            b["strike"],
            mm["r"],
            mm["maturity"],
        )
        bc = resources.component_costs(bp)
        for k in (0, 1):
            rows.append(
                {
                    "variant": "barrier digital (stretch)",
                    "operating_point": f"k = {k}",
                    "reaches_epsilon": None,
                    "median_queries": None,
                    **resources.grover_power_cost(bc, k),
                }
            )
    table = pd.DataFrame(rows)
    iqae = table[table.variant == "IQAE"].iloc[0]
    qae = table[table.variant == "QAE (MLE post-processing)"].iloc[0]
    e2 = fits["queries_for_epsilon"]
    claim = {
        "epsilon": eps,
        "X_iqae_queries": e2["iqae"],
        "Y_mc_samples": e2["mc"],
        "Z_extra_qubits_qae": int(qae.num_qubits - iqae.num_qubits),
        "W_depth_ratio_qae_over_iqae": float(qae.depth / iqae.depth),
        "qae_operating_point": qae.operating_point,
        "iqae_operating_point": iqae.operating_point,
        "also": {
            "mlae_queries": e2["mlae"],
            "qmc_samples": e2.get("qmc"),
            "qae_mle_queries": e2.get("qae_mle"),
        },
    }
    check = pd.DataFrame(resources.composition_check(problem, costs))
    _save(
        config,
        {"rows": rows, "claim": claim, "costs": costs},
        {"": table, "_composition_check": check},
        extra={"claim": claim},
    )
    _, summ, _ = _dirs(config)
    (summ / f"{config['name']}_claim.json").write_text(json.dumps(claim, indent=1))
    log(json.dumps(claim, indent=1))
    return table


# -------------------------------------------------------------------------------- E4 noise


def _contrast_job(job: dict) -> list[dict]:
    """P(objective = 1) of Q^k A on the noisy backend vs the ideal sin²((2k+1)θ)."""
    from qiskit import ClassicalRegister, transpile

    from qcc_common.backends import noisy_simulator
    from qcc_common.metrics import two_qubit_gate_count

    problem, _, _ = build_problem(job["model"], job["payoff"], job["n"])
    backend = noisy_simulator(job["fake_backend"], scale=job["noise_scale"], seed=job["seed"])
    grover = problem.estimation_problem().grover_operator
    theta = np.arcsin(np.sqrt(problem.exact_amplitude))
    rows = []
    for k in job["powers"]:
        circuit = problem.a_operator.copy()
        circuit.compose(grover.power(k), inplace=True)
        creg = ClassicalRegister(1)
        circuit.add_register(creg)
        circuit.measure(problem.objective_qubit, creg[0])
        tqc = transpile(circuit, backend, optimization_level=1, seed_transpiler=1234)
        counts = (
            backend.run(tqc, shots=job["shots"], seed_simulator=job["seed"] + k)
            .result()
            .get_counts()
        )
        p_noisy = counts.get("1", 0) / job["shots"]
        p_ideal = float(np.sin((2 * k + 1) * theta) ** 2)
        rows.append(
            {
                "noise_scale": job["noise_scale"],
                "k": k,
                "p_noisy": p_noisy,
                "p_ideal": p_ideal,
                "deviation": abs(p_noisy - p_ideal),
                "device_depth": tqc.depth(),
                "device_two_qubit_gates": two_qubit_gate_count(tqc),
            }
        )
    log(f"[e4] contrast scale={job['noise_scale']} done")
    return rows


def run_noise(config: dict) -> pd.DataFrame:
    """E4: the E2 sweep for the quantum estimators on the noisy fake-backend simulator."""
    jobs = []
    for scale in config["noise_scales"]:
        jobs += _convergence_jobs(config, "e4", noise_scale=float(scale))
    quantum_only = {k: v for k, v in config["sweeps"].items() if k in QUANTUM}
    jobs += _convergence_jobs({**config, "sweeps": quantum_only}, "e4", None)  # noiseless baseline
    rows = [
        r
        for res in parallel_map(_e2_job, jobs, config.get("workers", 1), _parts(config))
        for r in res
    ]
    cjobs = [
        {
            "model": config["model"],
            "payoff": config["payoff"],
            "n": config["n"],
            "fake_backend": config.get("fake_backend", "FakeSherbrooke"),
            "noise_scale": float(s),
            "seed": config["seed"],
            "shots": config["contrast_shots"],
            "powers": config["contrast_powers"],
        }
        for s in config["noise_scales"]
    ]
    contrast = pd.DataFrame(
        [
            r
            for res in parallel_map(_contrast_job, cjobs, config.get("workers", 1), _parts(config))
            for r in res
        ]
    )
    runs = pd.DataFrame(rows)
    runs["noise_scale"] = runs.noise_scale.fillna(0.0)
    stats = summarize(runs, by=("noise_scale", "method", "knob"))
    frontier = {}
    for scale, s in stats[stats.method == "mlae"].groupby("noise_scale"):
        best = s.loc[s.median_error.idxmin()]
        k_star = int(best.median_max_k)
        c = contrast[(contrast.noise_scale == scale) & (contrast.k == k_star)]
        frontier[str(scale)] = {
            "best_M": int(best.knob),
            "k_star": k_star,
            "median_error_at_best": float(best.median_error),
            "device_depth_at_k_star": int(c.device_depth.iloc[0]) if len(c) else None,
            "device_two_qubit_gates_at_k_star": int(c.device_two_qubit_gates.iloc[0])
            if len(c)
            else None,
        }
    _save(
        config,
        rows,
        {"": runs, "_stats": stats, "_contrast": contrast},
        extra={"frontier": frontier},
    )
    _, summ, figdir = _dirs(config)
    (summ / f"{config['name']}_frontier.json").write_text(json.dumps(frontier, indent=1))
    plot_noise(config, stats, contrast, figdir)
    log(json.dumps(frontier, indent=1))
    return stats


def plot_noise(config, stats, contrast, figdir):
    import matplotlib.pyplot as plt

    from qcc_common.plotting import MUTED, REFERENCE, apply_style, save_figure, series_style

    apply_style()
    scales = sorted(stats.noise_scale.unique())
    methods = [m for m in ("mlae", "iqae") if m in set(stats.method)]
    fig, axes = plt.subplots(1, len(methods), figsize=(10, 4.2), sharey=True)
    for ax, method in zip(np.atleast_1d(axes), methods, strict=True):
        for i, scale in enumerate(scales):
            s = stats[(stats.method == method) & (stats.noise_scale == scale)].sort_values(
                "median_queries"
            )
            style = series_style(i)
            ax.fill_between(
                s.median_queries,
                np.maximum(s.p16_error, 1e-6),
                np.maximum(s.p84_error, 1e-6),
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )
            label = "noiseless" if scale == 0 else f"noise × {scale:g}"
            ax.plot(s.median_queries, np.maximum(s.median_error, 1e-6), label=label, **style)
        ax.axhline(config.get("epsilon", 1e-3), color=REFERENCE, linestyle="--", linewidth=1.0)
        ax.set(xscale="log", yscale="log", xlabel="oracle queries", title=METHOD_LABELS[method])
        ax.legend(loc="lower left", fontsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel("|estimate − exact| (price units)")
    fig.suptitle(
        f"E4 — {config.get('fake_backend', 'FakeSherbrooke')} noise model, n = {config['n']} "
        f"(median, 16–84 %, R = {config['repetitions']})",
        fontsize=10,
        color=MUTED,
    )
    save_figure(fig, figdir / config["name"])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, scale in enumerate(sorted(contrast.noise_scale.unique())):
        c = contrast[contrast.noise_scale == scale].sort_values("k")
        ax.plot(
            c.k + 1,
            np.maximum(c.deviation, 1e-4),
            label=f"noise × {scale:g}",
            **series_style(i + 1),
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Grover power k + 1",
        ylabel="|P_noisy(good) − P_ideal(good)|",
        title=f"E4 — signal error of Q^k A ({config['contrast_shots']} shots)",
    )
    ax.legend(loc="lower right")
    save_figure(fig, figdir / f"{config['name']}_contrast")


# ------------------------------------------------------------------------------ E5 loading


def run_loading(config: dict) -> pd.DataFrame:
    """E5: loader cost vs n, lognormal vs bimodal, and its share of the A operator."""
    from qae_options.distributions import discretize, loader
    from qae_options.payoffs import digital_call
    from qcc_common.metrics import circuit_metrics
    from qcc_common.plotting import apply_style, save_figure, series_style

    rows = []
    for label, model in config["models"].items():
        dist = build_distribution(model)
        for n in config["n_values"]:
            grid = discretize(dist, n)
            lm = circuit_metrics(loader(grid.probs))
            problem = digital_call(grid, config["strike"], model["r"], model["maturity"])
            am = circuit_metrics(problem.a_operator)
            rows.append(
                {
                    "distribution": label,
                    "n": n,
                    "loader_depth": lm["depth"],
                    "loader_two_qubit_gates": lm["two_qubit_gates"],
                    "a_depth": am["depth"],
                    "a_two_qubit_gates": am["two_qubit_gates"],
                    "loader_share_depth": lm["depth"] / am["depth"],
                    "loader_share_two_qubit_gates": lm["two_qubit_gates"] / am["two_qubit_gates"],
                }
            )
            log(f"[e5] {label} n={n}")
    table = pd.DataFrame(rows)
    _save(config, rows, {"": table})
    import matplotlib.pyplot as plt

    apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0))
    for i, (label, t) in enumerate(table.groupby("distribution", sort=False)):
        st = series_style(i)
        a1.plot(t.n, t.loader_two_qubit_gates, label=f"loader, {label}", **st)
        a1.plot(t.n, t.a_two_qubit_gates, label=f"A operator, {label}", **{**st, "linestyle": "--"})
        a2.plot(t.n, 100 * t.loader_share_two_qubit_gates, label=label, **st)
    a1.plot(
        table.n.unique(),
        [2**n for n in table.n.unique()],
        color="#8a8980",
        linestyle=":",
        label="2ⁿ",
    )
    a1.set(
        yscale="log", xlabel="qubits n", ylabel="two-qubit gates (transpiled)", title="Loading cost"
    )
    a2.set(
        xlabel="qubits n",
        ylabel="loader share of A's two-qubit gates (%)",
        ylim=(0, 100),
        title="Share of the A operator",
    )
    a1.legend(fontsize=7.5)
    a2.legend()
    _, _, figdir = _dirs(config)
    save_figure(fig, figdir / config["name"])
    return table


RUNNERS: dict[str, Callable[[dict], pd.DataFrame]] = {
    "e1": run_validate,
    "e2": run_convergence,
    "e3": run_resources,
    "e4": run_noise,
    "e5": run_loading,
}
