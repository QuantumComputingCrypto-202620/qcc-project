"""Live demonstration: `uv run qae-demo` (see demo/DEMO.md for the run sheet).

Seven steps, each paused until Enter is pressed. Everything noiseless is computed live with the
project's own library (the exact-distribution sampler makes it take seconds); the full study
results (E2-E4, ~1.5 h to regenerate) are read from the committed ``results/summary`` files.
Figures are written to ``demo/out/`` and opened in the system viewer.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = PROJECT_ROOT / "results" / "summary"
FIGURES = PROJECT_ROOT / "results" / "figures"

# The main study's model (configs/e2_convergence.toml): a binary event before expiry.
MODEL = {
    "distribution": "bimodal",
    "w": 0.4,
    "center1": 85.0,
    "sigma1": 0.10,
    "center2": 115.0,
    "sigma2": 0.10,
    "r": 0.03,
    "maturity": 0.5,
}
PAYOFF = {"type": "digital", "strike": 100.0, "payout": 1.0}
SHOTS = 100

console = Console(highlight=False)


@dataclass
class Demo:
    args: argparse.Namespace
    out: Path
    state: dict = field(default_factory=dict)

    @property
    def quick(self) -> bool:
        return self.args.quick

    def problem(self, n: int):
        """(PricingProblem, distribution, continuous price) for the main model on 2^n points."""
        key = ("problem", n)
        if key not in self.state:
            from qae_options.experiments import build_distribution, build_problem

            problem, continuous, _ = build_problem(MODEL, PAYOFF, n)
            self.state[key] = (problem, build_distribution(MODEL), continuous)
        return self.state[key]

    def show(self, path: Path) -> None:
        console.print(f"  [dim]figura:[/] {path.relative_to(PROJECT_ROOT)}")
        if self.args.no_open:
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif sys.platform == "win32":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)], stderr=subprocess.DEVNULL)
        except OSError:
            pass


def say(text: str) -> None:
    """Narration box; lines are kept under 78 characters so an 80-column terminal fits."""
    console.print(Panel(text, border_style="cyan", box=box.ROUNDED, padding=(0, 1), expand=False))


def schematic(circuit):
    """Copy of ``circuit`` with every top-level block as an opaque labelled box (no params)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate

    labels = {"state_preparation": "P(S)", "IntComp": "i ≥ K"}
    out = QuantumCircuit(circuit.num_qubits)
    for inst in circuit.data:
        name = labels.get(inst.operation.name, inst.operation.name)
        out.append(
            Gate(name, inst.operation.num_qubits, []),
            [circuit.find_bit(q).index for q in inst.qubits],
        )
    return out


def fmt(x: float, digits: int = 3) -> str:
    if x == 0:
        return "0"
    if 1e-2 <= abs(x) < 1e5:
        return f"{x:,.{max(digits - 1 - int(np.floor(np.log10(abs(x)))), 0)}f}"
    return f"{x:.{digits - 1}e}"


def progress() -> Progress:
    return Progress(
        TextColumn("  {task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def _figure():
    from qcc_common.plotting import apply_style

    apply_style()
    import matplotlib.pyplot as plt

    return plt


# ------------------------------------------------------------------------------------ steps


def step_problem(demo: Demo) -> None:
    """The pricing problem: a digital call on a bimodal price distribution."""
    n = demo.args.n
    problem, dist, continuous = demo.problem(n)
    k = PAYOFF["strike"]
    say(
        "Un evento binario antes del vencimiento (p. ej. una decisión regulatoria):\n"
        "  40 % «malas noticias»  -> precio cerca de  85\n"
        "  60 % «buenas noticias» -> precio cerca de 115\n"
        "Call digital: paga 1 si S_T >= K = 100 a T = 6 meses.\n"
        "Precio = e^(-rT) P(S_T >= K)"
    )
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Cantidad")
    table.add_column("Valor", justify="right")
    table.add_row("Precio continuo (forma cerrada)", f"{continuous:.6f}")
    table.add_row(
        f"Precio en la malla, n = {n} qubits ({2**n} puntos)", f"{problem.discretized_price:.6f}"
    )
    table.add_row("Error de discretización", f"{problem.discretized_price - continuous:+.2e}")
    table.add_row(
        "Amplitud a estimar  a = P(S_T >= K) en la malla", f"{problem.exact_amplitude:.6f}"
    )
    console.print(table)

    plt = _figure()
    from qae_options.distributions import discretize
    from qcc_common.plotting import REFERENCE, SERIES_COLORS, save_figure

    grid = discretize(dist, n)
    dx = grid.x[1] - grid.x[0]
    xs = np.linspace(grid.low, grid.high, 400)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = [SERIES_COLORS[1] if x >= k else SERIES_COLORS[0] for x in grid.x]
    ax.bar(
        grid.x,
        grid.probs / dx,
        width=dx * 0.9,
        color=colors,
        alpha=0.55,
        label="celdas de la malla",
    )
    ax.plot(xs, dist.pdf(xs), color="#1f1f1e", lw=1.6, label="densidad continua")
    ax.axvline(k, color=REFERENCE, ls="--", lw=1)
    ax.text(k + 1, ax.get_ylim()[1] * 0.92, "K = 100", color=REFERENCE)
    ax.set_xlabel("precio al vencimiento S_T")
    ax.set_ylabel("densidad")
    ax.set_title(f"Distribución bimodal en 2^{n} = {2**n} puntos de malla  (naranja: pago = 1)")
    ax.legend(loc="upper left")
    demo.show(save_figure(fig, demo.out / "01_distribution", formats=("png",))[0])


def step_circuit(demo: Demo) -> None:
    """Encoding it as a quantum circuit A: loader + comparator."""
    from qiskit.quantum_info import Statevector

    from qcc_common.metrics import circuit_metrics

    n = demo.args.n
    problem, _, _ = demo.problem(n)
    say(
        f"A = cargador + comparador en {n + 1} qubits.\n"
        f"  P(S):  prepara sum_i sqrt(p_i)|i>  en {n} qubits de precio\n"
        "  cmp:   invierte el qubit objetivo si y solo si i >= índice(K)\n"
        "=> P(objetivo = 1) = a, el precio que buscamos.\n"
        "Una aplicación de A (o de su inversa) = una consulta al oráculo."
    )
    console.print(schematic(problem.a_operator).draw("text"))
    p1 = float(Statevector(problem.a_operator).probabilities([problem.objective_qubit])[1])
    console.print(
        f"  vector de estado: P(objetivo = 1) = [bold]{p1:.10f}[/]\n"
        f"  suma de la cola de la malla:       {problem.exact_amplitude:.10f}"
    )
    with console.status("  transpilando A a {rz, sx, x, ecr} en una cadena lineal ..."):
        metrics = circuit_metrics(problem.a_operator)
    console.print(
        f"  costo en hardware de UNA consulta: profundidad [bold]{metrics['depth']:,}[/],\n"
        f"  compuertas de dos qubits [bold]{metrics['two_qubit_gates']:,}[/]"
    )


def step_grover(demo: Demo) -> None:
    """Grover amplification: the amplitude becomes an oscillation frequency."""
    from qiskit import QuantumCircuit

    from qae_options.estimators import grover_power
    from qae_options.sampling import exact_statevector

    n = demo.args.n
    problem, _, _ = demo.problem(n)
    ep = problem.estimation_problem()
    a = problem.exact_amplitude
    theta = float(np.arcsin(np.sqrt(a)))
    say(
        "Escribimos a = sin^2(theta).  El operador de Grover Q rota el estado\n"
        "2 theta, así que tras k aplicaciones:\n"
        "  P(objetivo = 1) = sin^2((2k + 1) theta).\n"
        "Leer una frecuencia con precisión eps toma ~1/eps consultas;\n"
        "promediar lanzamientos de moneda (Monte Carlo) toma ~1/eps^2."
    )
    rng = np.random.default_rng(demo.args.seed)
    cache: dict = {}
    ks = [0, 1, 2, 3, 4, 6, 8] if demo.quick else [0, 1, 2, 3, 4, 6, 8, 12, 16]
    table = Table(box=box.SIMPLE_HEAVY, title=f"theta = {theta:.5f} rad")
    for col in (
        "k",
        "consultas/shot",
        "P(1) circuito",
        "sin^2((2k+1)theta)",
        f"muestra {SHOTS} shots",
    ):
        table.add_column(col, justify="right")
    measured = []
    for k in ks:
        circuit = QuantumCircuit(problem.a_operator.num_qubits)
        circuit.compose(problem.a_operator, inplace=True)
        if k:
            circuit.append(grover_power(ep, k), range(circuit.num_qubits))
        p = float(exact_statevector(circuit, cache).probabilities([problem.objective_qubit])[1])
        sample = rng.binomial(SHOTS, p) / SHOTS
        measured.append((k, p, sample))
        table.add_row(
            str(k),
            str(2 * k + 1),
            f"{p:.6f}",
            f"{np.sin((2 * k + 1) * theta) ** 2:.6f}",
            f"{sample:.2f}",
        )
    console.print(table)

    plt = _figure()
    from qcc_common.plotting import SERIES_COLORS, save_figure

    kk = np.linspace(0, max(ks), 600)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(kk, np.sin((2 * kk + 1) * theta) ** 2, color=SERIES_COLORS[0], label="sin²((2k+1)θ)")
    ax.plot(
        [m[0] for m in measured],
        [m[1] for m in measured],
        "o",
        color=SERIES_COLORS[0],
        label="circuito (exacto)",
    )
    ax.plot(
        [m[0] for m in measured],
        [m[2] for m in measured],
        "s",
        color=SERIES_COLORS[1],
        mfc="none",
        label=f"{SHOTS} shots",
    )
    ax.set_xlabel("aplicaciones de Grover k")
    ax.set_ylabel("P(objetivo = 1)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Amplificación de amplitud: a fija la frecuencia de la oscilación")
    ax.legend(loc="lower right")
    demo.show(save_figure(fig, demo.out / "03_grover", formats=("png",))[0])


def step_race(demo: Demo) -> None:
    """Live race: IQAE vs Monte Carlo on the same query budget."""
    from qae_options.classical import monte_carlo
    from qae_options.distributions import discretize
    from qae_options.estimators import LazyIterativeAmplitudeEstimation, run_iqae
    from qae_options.sampling import ExactDistributionSampler

    n = demo.args.n
    problem, dist, _ = demo.problem(n)
    exact = problem.exact_price
    discount = problem.to_price(1.0)
    grid = discretize(dist, n)
    payoff = (np.arange(2**n) >= problem.metadata["strike_index"]).astype(float)

    # 1. One IQAE run, iteration by iteration.
    eps_one = 1e-2 if demo.quick else 1e-3
    say(
        f"QAE iterativa (sin estimación de fase, sin qubits extra), eps = {eps_one:g}.\n"
        "Cada ronda elige la mayor potencia de Grover k que mantiene el ángulo\n"
        f"sin ambigüedad, mide {SHOTS} shots y reduce el intervalo de confianza."
    )
    cache: dict = {}
    sampler = ExactDistributionSampler(SHOTS, seed=demo.args.seed, cache=cache)
    t0 = time.perf_counter()
    result = LazyIterativeAmplitudeEstimation(eps_one, 0.05, sampler=sampler).estimate(
        problem.estimation_problem()
    )
    elapsed = time.perf_counter() - t0
    table = Table(box=box.SIMPLE_HEAVY)
    for col in (
        "ronda",
        "k de Grover",
        "consultas",
        "intervalo de precio",
        "ancho",
    ):
        table.add_column(col, justify="right")
    queries = 0
    for i, (k, (lo, hi)) in enumerate(
        zip(result.powers[1:], result.estimate_intervals[1:], strict=False), start=1
    ):
        queries += SHOTS * (2 * int(k) + 1)
        plo, phi = problem.to_price(lo), problem.to_price(hi)
        table.add_row(
            str(i), str(int(k)), f"{queries:,}", f"[{plo:.5f}, {phi:.5f}]", f"{phi - plo:.1e}"
        )
    console.print(table)
    est = problem.to_price(float(result.estimation))
    console.print(
        f"  estimación [bold]{est:.6f}[/]  exacto {exact:.6f}  "
        f"error [bold]{abs(est - exact):.1e}[/]"
        f"\n  con [bold]{queries:,}[/] consultas al oráculo  ({elapsed:.2f} s en este portátil)"
    )

    # 2. The race: R seeds per target, Monte Carlo gets exactly IQAE's query budget.
    reps = demo.args.reps or (5 if demo.quick else 20)
    targets = [1e-1, 1e-2] if demo.quick else [1e-1, 1e-2, 1e-3, 3e-4]
    say(
        f"Ahora {reps} semillas independientes por objetivo.  Monte Carlo recibe el MISMO\n"
        "número de muestras que consultas usó IQAE.  Errores: medianas frente al\n"
        "precio exacto de la malla."
    )
    rows, points = [], []
    with progress() as bar:
        task = bar.add_task("IQAE + MC", total=reps * len(targets))
        for target in targets:
            q_list, e_iqae, e_mc = [], [], []
            for r in range(reps):
                seed = demo.args.seed + 1000 + r
                s = ExactDistributionSampler(SHOTS, seed=seed, cache=cache)
                res = run_iqae(problem, target, s)
                mc = monte_carlo(
                    grid, payoff, res.oracle_queries, np.random.default_rng(seed), discount
                )
                q_list.append(res.oracle_queries)
                e_iqae.append(abs(res.estimate - exact))
                e_mc.append(abs(mc - exact))
                points.append((res.oracle_queries, e_iqae[-1], e_mc[-1]))
                bar.advance(task)
            rows.append((target, np.median(q_list), np.median(e_iqae), np.median(e_mc)))
    sigma = discount * np.sqrt(problem.exact_amplitude * (1 - problem.exact_amplitude))
    table = Table(box=box.SIMPLE_HEAVY)
    for col in (
        "eps IQAE",
        "consultas",
        "error IQAE",
        "error MC",
        "MC necesarias",
        "MC / IQAE",
    ):
        table.add_column(col, justify="right")
    for target, q, ei, em in rows:
        n_mc = (0.6745 * sigma / ei) ** 2  # median |error| of a mean = 0.6745 sigma / sqrt(N)
        table.add_row(
            f"{target:g}", f"{q:,.0f}", fmt(ei), fmt(em), f"{n_mc:,.0f}", f"{n_mc / q:.1f}x"
        )
    console.print(table)
    console.print(
        "  [dim]consultas: mediana de IQAE.  error MC: con el mismo presupuesto.\n"
        "  MC necesarias: muestras de MC para igualar el error de IQAE,\n"
        "  = (0.6745 sigma / error)^2 con\n"
        f"  sigma = {sigma:.4f} = desviación estándar del pago descontado.[/]"
    )

    plt = _figure()
    from qcc_common.plotting import REFERENCE, save_figure, series_style

    q = np.array([p[0] for p in points], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.loglog(
        q,
        [p[1] for p in points],
        ls="none",
        alpha=0.6,
        label="IQAE",
        **{k: v for k, v in series_style(0).items() if k != "linestyle"},
    )
    ax.loglog(
        q,
        [p[2] for p in points],
        ls="none",
        alpha=0.6,
        label="Monte Carlo (mismo presupuesto)",
        **{k: v for k, v in series_style(1).items() if k != "linestyle"},
    )
    qq = np.geomspace(q.min(), q.max(), 50)
    med = rows[0]
    ax.loglog(qq, med[2] * med[1] / qq, color=REFERENCE, ls="--", lw=1, label="∝ 1/N")
    ax.loglog(qq, med[3] * np.sqrt(med[1] / qq), color=REFERENCE, ls=":", lw=1, label="∝ 1/√N")
    ax.set_xlabel("consultas al oráculo N (muestras para MC)")
    ax.set_ylabel("|error| (unidades de precio)")
    ax.set_title(f"Corrida en vivo: {reps} semillas por objetivo de IQAE, n = {n}")
    ax.legend(loc="lower left")
    demo.show(save_figure(fig, demo.out / "04_race", formats=("png",))[0])


def step_study(demo: Demo) -> None:
    """The full study (E2 + E3), from the committed results."""
    fits = json.loads((SUMMARY / "e2_convergence_fits.json").read_text())
    claim = json.loads((SUMMARY / "e3_resources_claim.json").read_text())
    say(
        "El estudio completo: 49 configuraciones x 50 semillas (E2) y recursos\n"
        "transpilados (E3).  Leído de results/summary/\n"
        "(run_all.sh lo regenera en ~1.5 h)."
    )
    labels = {
        "iqae": "IQAE",
        "mlae": "MLAE",
        "qae_mle": "QAE + MLE",
        "qae": "QAE (malla)",
        "mc": "Monte Carlo",
        "qmc": "quasi-MC (Sobol)",
    }
    table = Table(box=box.SIMPLE_HEAVY, title="error ∝ N^pendiente  (E2)")
    table.add_column("Método")
    table.add_column("pendiente", justify="right")
    table.add_column("consultas para eps = 1e-3", justify="right")
    for key, label in labels.items():
        f, q = fits["fits"][key], fits["queries_for_epsilon"][key]
        note = "" if q["how"] == "interpolated" else " (extrapolado)"
        style = "bold" if key in ("iqae", "mc", "qmc") else ""
        table.add_row(
            label,
            f"{f['slope']:+.2f} ± {f['slope_std']:.2f}",
            f"{q['queries']:,.0f}{note}",
            style=style,
        )
    console.print(table)
    x = claim["X_iqae_queries"]["queries"]
    y = claim["Y_mc_samples"]["queries"]
    qmc = claim["also"]["qmc_samples"]["queries"]
    console.print(
        Panel(
            f"Para alcanzar eps = 1e-3, IQAE usa X ≈ {x:,.0f} consultas y\n"
            f"Monte Carlo usa Y ≈ {y:,.0f} muestras ({y / x:.0f}x más).\n"
            f"La QAE canónica necesita Z = {claim['Z_extra_qubits_qae']} qubits extra y\n"
            f"W ≈ {claim['W_depth_ratio_qae_over_iqae']:.0f}x la profundidad de IQAE.\n"
            f"[bold]Pero[/] quasi-Monte Carlo necesita solo ≈ {qmc:,.0f} muestras:\n"
            "no hay ventaja frente a un buen método clásico.",
            title="Afirmación verificable",
            border_style="green",
            expand=False,
        )
    )
    demo.show(FIGURES / "e2_convergence.png")


def step_noise(demo: Demo) -> None:
    """Noise: the same circuits on a simulated IBM device."""
    from qae_options.experiments import _contrast_job

    n = 3
    say(
        "Los mismos circuitos en un simulador con ruido: modelo FakeSherbrooke de IBM\n"
        f"(errores de compuerta, lectura y T1/T2), n = {n} qubits de precio.\n"
        "«escala» multiplica cada tasa de error: 1 = el dispositivo actual."
    )
    scales = [1.0] if demo.quick else [1.0, 0.03]
    powers = [0, 1, 2] if demo.quick else [0, 1, 2, 4, 8]
    shots = 2000
    table = Table(box=box.SIMPLE_HEAVY)
    for col in ("escala de ruido", "k", "compuertas 2q", "P ideal", "P con ruido", "|desviación|"):
        table.add_column(col, justify="right")
    for scale in scales:
        with console.status(
            f"  simulando Q^k A en el dispositivo con ruido, escala = {scale:g} ..."
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                rows = _contrast_job(
                    {
                        "model": MODEL,
                        "payoff": PAYOFF,
                        "n": n,
                        "fake_backend": "FakeSherbrooke",
                        "noise_scale": scale,
                        "seed": demo.args.seed,
                        "powers": powers,
                        "shots": shots,
                    }
                )
        for row in rows:
            dev = row["deviation"]
            colour = "green" if dev < 0.02 else "yellow" if dev < 0.08 else "red"
            table.add_row(
                f"{scale:g}",
                str(row["k"]),
                f"{row['device_two_qubit_gates']:,}",
                f"{row['p_ideal']:.3f}",
                f"{row['p_noisy']:.3f}",
                f"[{colour}]{dev:.3f}[/]",
            )
        table.add_section()
    console.print(table)
    frontier = json.loads((SUMMARY / "e4_noise_frontier.json").read_text())
    console.print(
        "  E4: con ruido realista (escala 0.1-1) el mejor estimador usa k <= 1;\n"
        f"  su error mediano es {frontier['0.1']['median_error_at_best']:.3f}-"
        f"{frontier['1.0']['median_error_at_best']:.3f}, es decir 25-95x eps.\n"
        "  Las iteraciones de Grover solo ayudan por debajo del ~3 % del ruido actual."
    )
    demo.show(FIGURES / "e4_noise.png")


def step_takeaways(demo: Demo) -> None:
    """Takeaways."""
    console.print(
        Panel(
            "1. Cada estimador reproduce su precio exacto: pipeline validado (E1).\n"
            "2. Los estimadores cuánticos escalan como 1/N, Monte Carlo como 1/√N (E2):\n"
            "   IQAE necesita ~32x menos consultas que MC con eps = 1e-3.\n"
            "3. IQAE: sin qubits extra y ~38x menos profundidad que la QAE con QPE (E3).\n"
            "4. Quasi-Monte Carlo supera aquí a todo método cuántico (~170 muestras).\n"
            "5. Ruido realista: nada alcanza eps.  eps requiere ~55k compuertas de dos\n"
            "   qubits; el ruido permite unos cientos (E4).  Domina el comparador (E5).",
            title="Conclusiones",
            border_style="magenta",
            expand=False,
        )
    )


STEPS: Sequence[tuple[str, Callable[[Demo], None]]] = (
    ("El problema de valoración", step_problem),
    ("El operador A: un circuito cuántico para el precio", step_circuit),
    ("Amplificación de Grover", step_grover),
    ("Carrera en vivo: IQAE frente a Monte Carlo", step_race),
    ("El estudio completo (E2, E3)", step_study),
    ("Ruido (E4)", step_noise),
    ("Conclusiones", step_takeaways),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qae-demo", description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=int, default=1, help="start at this step (1-7)")
    parser.add_argument("--last", type=int, default=len(STEPS), help="stop after this step")
    parser.add_argument("--only", type=int, default=None, help="run just this step")
    parser.add_argument("--no-pause", action="store_true", help="do not wait for Enter")
    parser.add_argument("--no-open", action="store_true", help="do not open figures")
    parser.add_argument("--quick", action="store_true", help="smaller sweeps (~20 s total)")
    parser.add_argument("--n", type=int, default=5, help="price qubits for steps 1-4")
    parser.add_argument("--reps", type=int, default=None, help="seeds per IQAE target in step 4")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for i, (title, _fn) in enumerate(STEPS, start=1):
            console.print(f"{i}. {title}")
        return 0
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    demo = Demo(args=args, out=PROJECT_ROOT / "demo" / "out")
    demo.out.mkdir(parents=True, exist_ok=True)
    last = min(args.last, len(STEPS))
    selected = [args.only] if args.only else list(range(args.step, last + 1))
    console.print(
        Rule("[bold]Valoración de opciones con estimación de amplitud cuántica[/]  ·  demo en vivo")
    )
    for position, index in enumerate(selected):
        title, fn = STEPS[index - 1]
        console.print()
        console.print(Rule(f"[bold cyan]{index}/{len(STEPS)}  {title}", align="left"))
        try:
            fn(demo)
        except KeyboardInterrupt:
            console.print("  [yellow]paso interrumpido[/]")
        if not args.no_pause and position < len(selected) - 1:
            try:
                console.input("\n[dim]  pulsa Enter para el siguiente paso (Ctrl-C para salir) [/]")
            except (KeyboardInterrupt, EOFError):
                console.print()
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
