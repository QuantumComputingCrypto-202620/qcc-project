"""VQE runner: optimizer + initialization strategy -> :class:`VQERecord`. Energies in Ha."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA, Optimizer
from qiskit_algorithms.utils import algorithm_globals

from vqe_lih.hamiltonian import QubitHamiltonian

CHEMICAL_ACCURACY = 1.6e-3  # Ha (about 1 kcal/mol)
SECTOR_TOLERANCE = 1e-2
DEFAULT_MAXITER = {"cobyla": 1000, "l_bfgs_b": 1000, "slsqp": 1000, "spsa": 300, "spsa_small": 300}
SPSA_SMALL_STEP = 0.05  # rad: first-step and perturbation size for ``spsa_small``


class SmallStepSPSA(SPSA):
    """SPSA with the standard power-series calibration, but targeting a first step (and
    perturbation) of ``SPSA_SMALL_STEP`` rad instead of the default 2π/10 ≈ 0.63 rad, which is
    far larger than typical UCCSD amplitudes (~0.05 rad)."""

    def minimize(self, fun, x0, jac=None, bounds=None):
        self.learning_rate, self.perturbation = self.calibrate(
            fun, np.asarray(x0), c=SPSA_SMALL_STEP, target_magnitude=SPSA_SMALL_STEP
        )
        return super().minimize(fun, x0, jac, bounds)


RANDOM_INIT_SCALE = 0.1


def make_optimizer(name: str, maxiter: int | None = None, seed: int | None = None) -> Optimizer:
    name = name.lower()
    maxiter = maxiter or DEFAULT_MAXITER[name]
    if name in ("spsa", "spsa_small"):
        algorithm_globals.random_seed = seed  # SPSA draws its perturbations from here
        return SPSA(maxiter=maxiter) if name == "spsa" else SmallStepSPSA(maxiter=maxiter)
    optimizers = {"cobyla": COBYLA, "l_bfgs_b": L_BFGS_B, "slsqp": SLSQP}
    return optimizers[name](maxiter=maxiter)


def initial_point(
    strategy: str,
    num_parameters: int,
    seed: int | None = None,
    warm: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """``zeros``; ``random`` = N(0, 0.1²) with ``seed``; ``warm_start`` = ``warm`` parameters."""
    if strategy == "zeros":
        return np.zeros(num_parameters)
    if strategy == "random":
        return np.random.default_rng(seed).normal(0.0, RANDOM_INIT_SCALE, num_parameters)
    if strategy == "warm_start":
        if warm is None:
            raise ValueError("warm_start needs the neighbouring point's parameters")
        warm = np.asarray(warm, dtype=float)
        if warm.shape != (num_parameters,):
            raise ValueError(f"warm-start vector has shape {warm.shape}, need ({num_parameters},)")
        return warm.copy()
    raise ValueError(f"unknown initialization {strategy!r}")


@dataclass
class VQERecord:
    """One VQE run. ``energy`` is the noiseless energy at the optimal parameters;
    ``energy_estimate`` is what the optimizer's estimator reported (they differ with shots)."""

    energy: float
    energy_estimate: float
    delta_e: float | None
    chemical_accuracy: bool | None
    num_particles: float
    s_squared: float
    wrong_sector: bool
    nfev: int
    nit: int | None
    evals_to_chemical_accuracy: int | None
    wall_time: float
    optimizer: str
    init: str
    seed: int | None
    parameters: list[float]
    history: list[float] = field(repr=False)

    def summary(self) -> dict:
        """All fields except the parameter vector and energy history."""
        d = dict(self.__dict__)
        d.pop("parameters")
        d.pop("history")
        return d


def exact_expectations(
    circuit: QuantumCircuit, operators: list, parameters: np.ndarray
) -> np.ndarray:
    result = StatevectorEstimator().run([(circuit, operators, parameters)]).result()
    return np.real(result[0].data.evs)


def run_vqe(
    ham: QubitHamiltonian,
    ansatz: QuantumCircuit,
    optimizer: str,
    x0: np.ndarray,
    *,
    estimator: BaseEstimatorV2 | None = None,
    e_exact: float | None = None,
    init: str = "zeros",
    seed: int | None = None,
    maxiter: int | None = None,
) -> VQERecord:
    """Minimize ``<ansatz|H|ansatz>`` starting from ``x0``.

    ``ansatz`` should already be compiled (see ``ansatze.compile_for_simulation``). ⟨N⟩ and
    ⟨S²⟩ are evaluated exactly at the optimum, so the symmetry check is free of shot noise.
    """
    estimator = estimator or StatevectorEstimator()
    history: list[float] = []

    def callback(count, params, mean, metadata):
        history.append(float(mean) + ham.energy_shift)

    vqe = VQE(
        estimator,
        ansatz,
        make_optimizer(optimizer, maxiter, seed),
        initial_point=np.asarray(x0, dtype=float),
        callback=callback,
    )
    start = time.perf_counter()
    result = vqe.compute_minimum_eigenvalue(ham.qubit_op)
    wall_time = time.perf_counter() - start

    params = np.asarray(result.optimal_point, dtype=float)
    ops = [ham.qubit_op, ham.aux_ops["ParticleNumber"], ham.aux_ops["AngularMomentum"]]
    energy_raw, num_particles, s_squared = exact_expectations(ansatz, ops, params)
    energy = float(energy_raw) + ham.energy_shift

    delta_e = chem = evals_to_acc = None
    if e_exact is not None:
        delta_e = energy - e_exact
        chem = bool(delta_e < CHEMICAL_ACCURACY)
        below = np.flatnonzero(np.asarray(history) - e_exact < CHEMICAL_ACCURACY)
        evals_to_acc = int(below[0]) + 1 if below.size else None
    wrong_sector = bool(
        abs(num_particles - ham.total_particles) > SECTOR_TOLERANCE or s_squared > SECTOR_TOLERANCE
    )
    opt_result = result.optimizer_result
    return VQERecord(
        energy=energy,
        energy_estimate=float(np.real(result.eigenvalue)) + ham.energy_shift,
        delta_e=delta_e,
        chemical_accuracy=chem,
        num_particles=float(num_particles),
        s_squared=float(s_squared),
        wrong_sector=wrong_sector,
        nfev=int(result.cost_function_evals),
        nit=getattr(opt_result, "nit", None),
        evals_to_chemical_accuracy=evals_to_acc,
        wall_time=wall_time,
        optimizer=optimizer,
        init=init,
        seed=seed,
        parameters=params.tolist(),
        history=history,
    )
