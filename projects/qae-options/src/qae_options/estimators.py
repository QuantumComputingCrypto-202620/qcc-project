"""One interface over canonical QAE, IQAE and MLAE (qiskit-algorithms, SamplerV2).

Oracle queries (SPEC §5): one application of A or A† is one query, the Grover operator Q costs 2,
so a circuit with k Grover applications costs 2k + 1, times its shots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import AnnotatedOperation
from qiskit.circuit.annotated_operation import ControlModifier, PowerModifier
from qiskit_algorithms import (
    AmplitudeEstimation,
    IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
)

from qae_options.payoffs import PricingProblem


@dataclass
class EstimateResult:
    """``estimate`` and ``ci`` are in price units; ``amplitude`` is the raw estimate of a."""

    method: str
    amplitude: float
    estimate: float
    ci: tuple[float, float] | None
    oracle_queries: int
    grover_powers: list[int]
    shots_per_circuit: int
    extra: dict = field(default_factory=dict)


def qae_queries(num_eval_qubits: int, shots: int) -> int:
    return shots * (2 * (2**num_eval_qubits - 1) + 1)


def schedule_queries(powers: list[int], shots: int) -> int:
    return int(sum(shots * (2 * k + 1) for k in powers))


def grover_power(estimation_problem, power: int, num_controls: int = 0) -> AnnotatedOperation:
    """Q^power (optionally controlled) as one lazy annotated operation on a Q gate that is
    built once per problem. Same unitary as ``grover_operator.power(k)``; the simulator applies
    it as a matrix power and the transpiler expands it when gates are needed."""
    gate = estimation_problem.__dict__.get("_q_gate")
    if gate is None:
        gate = estimation_problem.grover_operator.to_gate(label="Q")
        estimation_problem.__dict__["_q_gate"] = gate
    modifiers = [PowerModifier(power)]
    if num_controls:
        modifiers.append(ControlModifier(num_controls))
    return AnnotatedOperation(gate, modifiers)


class LazyIterativeAmplitudeEstimation(IterativeAmplitudeEstimation):
    """IQAE whose Q^k A circuits use :func:`grover_power` (otherwise identical to
    ``IterativeAmplitudeEstimation.construct_circuit``)."""

    def construct_circuit(self, estimation_problem, k=0, measurement=False):
        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        circuit = QuantumCircuit(num_qubits, name="circuit")
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits))
            circuit.add_register(c)
        circuit.compose(estimation_problem.state_preparation, inplace=True)
        if k != 0:
            circuit.append(
                grover_power(estimation_problem, k),
                range(estimation_problem.grover_operator.num_qubits),
            )
        if measurement:
            circuit.barrier()
            circuit.measure(estimation_problem.objective_qubits, c[:])
        return circuit


class LazyMaximumLikelihoodAmplitudeEstimation(MaximumLikelihoodAmplitudeEstimation):
    """MLAE whose circuits use :func:`grover_power` (otherwise identical to
    ``MaximumLikelihoodAmplitudeEstimation.construct_circuits``)."""

    def construct_circuits(self, estimation_problem, measurement=False):
        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        q = QuantumRegister(num_qubits, "q")
        qc_0 = QuantumCircuit(q, name="qc_a")
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits))
            qc_0.add_register(c)
        qc_0.compose(estimation_problem.state_preparation, inplace=True)
        circuits = []
        for k in self._evaluation_schedule:
            qc_k = qc_0.copy(name=f"qc_a_q_{k}")
            if k != 0:
                qc_k.append(
                    grover_power(estimation_problem, k),
                    list(q[: estimation_problem.grover_operator.num_qubits]),
                )
            if measurement:
                qc_k.barrier()
                qc_k.measure(estimation_problem.objective_qubits, c[:])
            circuits.append(qc_k)
        return circuits


def qpe_circuit(estimation_problem, num_eval_qubits: int, measurement: bool) -> QuantumCircuit:
    """The canonical-QAE circuit of ``AmplitudeEstimation.construct_circuit`` (same registers,
    H layer, controlled Q^(2^j) on evaluation qubit j, ``QFT(m, inverse, no swaps)`` with
    reversed bits), but with *annotated* (lazy) controls: qiskit's default eagerly synthesizes
    a controlled copy of every gate in Q^(2^j), which takes minutes at m >= 6."""
    from qiskit.circuit.library import QFT

    a_op = estimation_problem.state_preparation
    grover = estimation_problem.grover_operator
    eval_reg = QuantumRegister(num_eval_qubits, "eval")
    state_reg = QuantumRegister(grover.num_qubits, "q")
    circuit = QuantumCircuit(eval_reg, state_reg)
    circuit.compose(a_op, list(state_reg), inplace=True)
    circuit.h(eval_reg)
    for j in range(num_eval_qubits):
        circuit.append(grover_power(estimation_problem, 2**j, 1), [eval_reg[j], *state_reg])
    iqft = QFT(num_eval_qubits, inverse=True, do_swaps=False).reverse_bits()
    circuit.compose(iqft, list(eval_reg), inplace=True)
    if measurement:
        creg = ClassicalRegister(num_eval_qubits)
        circuit.add_register(creg)
        circuit.measure(list(eval_reg), list(creg))
    return circuit


class CachedAmplitudeEstimation(AmplitudeEstimation):
    """Canonical QAE whose circuit (:func:`qpe_circuit`) is built once per (problem, m) and
    reused across seeded repetitions."""

    _circuits: dict = {}

    def construct_circuit(self, estimation_problem, measurement=False):
        key = (id(estimation_problem), self._m, measurement)
        hit = self._circuits.get(key)
        if hit is None or hit[0] is not estimation_problem:
            hit = (estimation_problem, qpe_circuit(estimation_problem, self._m, measurement))
            self._circuits[key] = hit
        return hit[1]


def run_qae(problem: PricingProblem, num_eval_qubits: int, sampler) -> EstimateResult:
    """Canonical QAE (QPE with ``num_eval_qubits`` evaluation qubits); grid estimate."""
    result = CachedAmplitudeEstimation(num_eval_qubits, sampler=sampler).estimate(
        problem.estimation_problem()
    )
    shots = sampler.default_shots
    return EstimateResult(
        method="qae",
        amplitude=float(result.estimation),
        estimate=problem.to_price(float(result.estimation)),
        ci=None,
        oracle_queries=qae_queries(num_eval_qubits, shots),
        grover_powers=[2**num_eval_qubits - 1],
        shots_per_circuit=shots,
        extra={"mle_amplitude": float(result.mle), "num_eval_qubits": num_eval_qubits},
    )


def run_iqae(
    problem: PricingProblem, epsilon_target: float, sampler, alpha: float = 0.05
) -> EstimateResult:
    """Iterative QAE; ``epsilon_target`` is the target half-width on the amplitude."""
    result = LazyIterativeAmplitudeEstimation(epsilon_target, alpha, sampler=sampler).estimate(
        problem.estimation_problem()
    )
    powers = [int(k) for k in result.powers[1:]]  # powers[0] = 0 is a placeholder, not run
    shots = sampler.default_shots
    lo, hi = result.confidence_interval
    return EstimateResult(
        method="iqae",
        amplitude=float(result.estimation),
        estimate=problem.to_price(float(result.estimation)),
        ci=(problem.to_price(lo), problem.to_price(hi)),
        oracle_queries=schedule_queries(powers, shots),
        grover_powers=powers,
        shots_per_circuit=shots,
        extra={"epsilon_target": epsilon_target, "alpha": alpha},
    )


def run_mlae(problem: PricingProblem, max_power_exponent: int, sampler) -> EstimateResult:
    """MLAE with schedule {0, 1, 2, 4, …, 2^M}, M = ``max_power_exponent``."""
    schedule = [0] + [2**j for j in range(max_power_exponent + 1)]
    result = LazyMaximumLikelihoodAmplitudeEstimation(schedule, sampler=sampler).estimate(
        problem.estimation_problem()
    )
    shots = sampler.default_shots
    return EstimateResult(
        method="mlae",
        amplitude=float(result.estimation),
        estimate=problem.to_price(float(result.estimation)),
        ci=None,
        oracle_queries=schedule_queries(schedule, shots),
        grover_powers=schedule,
        shots_per_circuit=shots,
        extra={"max_power_exponent": max_power_exponent},
    )


def run_estimator(problem: PricingProblem, method: str, knob: float, sampler) -> EstimateResult:
    if method == "qae":
        return run_qae(problem, int(knob), sampler)
    if method == "iqae":
        return run_iqae(problem, float(knob), sampler)
    if method == "mlae":
        return run_mlae(problem, int(knob), sampler)
    raise ValueError(f"unknown estimator {method!r}")


def estimate_amplitude_error(problem: PricingProblem, result: EstimateResult) -> float:
    """|â − a| in price units, relative to the exact amplitude of this A operator."""
    return float(abs(result.estimate - problem.exact_price))


__all__ = [
    "EstimateResult",
    "estimate_amplitude_error",
    "np",
    "qae_queries",
    "run_estimator",
    "run_iqae",
    "run_mlae",
    "run_qae",
    "schedule_queries",
]
