"""Payoff circuits: the A operator, its objective qubit and the amplitude → price map.

A prepares √(1−a)|ψ₀⟩|0⟩ + √a|ψ₁⟩|1⟩ on the objective qubit; prices are discounted, in price
units.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import IntegerComparatorGate, LinearAmplitudeFunctionGate
from qiskit_algorithms import EstimationProblem

from qae_options.distributions import Grid, JointGrid, loader


@dataclass
class PricingProblem:
    """``price = to_price(a)``; ``exact_amplitude`` is ``a`` on the grid (from the statevector
    of A for payoffs with approximation error, else from the probabilities)."""

    name: str
    a_operator: QuantumCircuit
    objective_qubit: int
    to_price: Callable[[float], float]
    exact_amplitude: float
    discretized_price: float
    num_state_qubits: int
    metadata: dict = field(default_factory=dict)

    @property
    def exact_price(self) -> float:
        """Price implied by the exact amplitude of this A operator."""
        return self.to_price(self.exact_amplitude)

    def estimation_problem(self) -> EstimationProblem:
        """The same ``EstimationProblem`` object on every call, so circuit caches keyed on it
        are reused across repetitions."""
        if "_estimation_problem" not in self.__dict__:
            self.__dict__["_estimation_problem"] = EstimationProblem(
                state_preparation=self.a_operator, objective_qubits=[self.objective_qubit]
            )
        return self.__dict__["_estimation_problem"]


def _comparator_value(index: int, n: int) -> int:
    return int(min(max(index, 0), 2**n))


def digital_call(grid: Grid, strike: float, r: float, maturity: float, payout: float = 1.0):
    """Cash-or-nothing call paying ``payout`` if S_T ≥ K. On the grid, a = P(S_T ≥ K) exactly."""
    n = grid.num_qubits
    k_index = grid.index_at_or_above(strike)
    a_op = QuantumCircuit(n + 1, name="A")
    a_op.compose(loader(grid.probs), range(n), inplace=True)
    if k_index < 2**n:
        a_op.append(IntegerComparatorGate(n, _comparator_value(k_index, n), geq=True), range(n + 1))
    exact_a = float(grid.probs[k_index:].sum())
    discount = np.exp(-r * maturity) * payout
    return PricingProblem(
        name="digital_call",
        a_operator=a_op,
        objective_qubit=n,
        to_price=lambda a, d=discount: d * a,
        exact_amplitude=exact_a,
        discretized_price=discount * exact_a,
        num_state_qubits=n,
        metadata={"strike": strike, "strike_index": k_index, "payout": payout},
    )


def european_call(grid: Grid, strike: float, r: float, maturity: float, rescaling_factor: float):
    """max(S_T − K, 0) through a piecewise-linear amplitude function with rescaling factor c.

    ``exact_amplitude`` comes from the statevector of A; the difference between
    ``exact_price`` and ``discretized_price`` is the payoff-approximation error (depends on c).
    """
    from qiskit.quantum_info import Statevector

    n = grid.num_qubits
    if not grid.low < strike < grid.high:
        raise ValueError("strike must lie inside the grid")
    function = LinearAmplitudeFunctionGate(
        n,
        slope=[0, 1],
        offset=[0, 0],
        domain=(grid.low, grid.high),
        image=(0, grid.high - strike),
        rescaling_factor=rescaling_factor,
        breakpoints=[grid.low, strike],
    )
    a_op = QuantumCircuit(function.num_qubits, name="A")
    a_op.compose(loader(grid.probs), range(n), inplace=True)
    a_op.append(function, range(function.num_qubits))
    probs = Statevector(a_op).probabilities([n])
    exact_a = float(probs[1])
    discount = np.exp(-r * maturity)
    return PricingProblem(
        name="european_call",
        a_operator=a_op,
        objective_qubit=n,
        to_price=lambda a, d=discount, f=function: d * float(f.post_processing(a)),
        exact_amplitude=exact_a,
        discretized_price=float(discount * grid.probs @ np.maximum(grid.x - strike, 0.0)),
        num_state_qubits=n,
        metadata={"strike": strike, "rescaling_factor": rescaling_factor},
    )


def barrier_digital(
    joint: JointGrid, barrier: float, strike: float, r: float, maturity: float, payout: float = 1.0
):
    """Up-and-out digital with two monitoring dates: pays if S_t1 < B and S_T ≥ K.

    Qubits: date-1 register (n), date-2 register (n), two comparator flags, objective.
    """
    n = joint.num_qubits_per_date
    b_index = int(np.searchsorted(joint.x, barrier, side="left"))  # x[i] < B  <=>  i < b_index
    k_index = int(np.searchsorted(joint.x, strike, side="left"))  # x[j] >= K <=>  j >= k_index
    flag1, flag2, objective = 2 * n, 2 * n + 1, 2 * n + 2
    a_op = QuantumCircuit(2 * n + 3, name="A")
    a_op.compose(loader(joint.probs), range(2 * n), inplace=True)
    a_op.append(
        IntegerComparatorGate(n, _comparator_value(b_index, n), geq=False),
        [*range(n), flag1],
    )
    a_op.append(
        IntegerComparatorGate(n, _comparator_value(k_index, n), geq=True),
        [*range(n, 2 * n), flag2],
    )
    a_op.ccx(flag1, flag2, objective)
    exact_a = float(joint.probs[:b_index, k_index:].sum())
    discount = np.exp(-r * maturity) * payout
    return PricingProblem(
        name="barrier_digital",
        a_operator=a_op,
        objective_qubit=objective,
        to_price=lambda a, d=discount: d * a,
        exact_amplitude=exact_a,
        discretized_price=discount * exact_a,
        num_state_qubits=2 * n,
        metadata={"barrier": barrier, "strike": strike, "payout": payout},
    )
