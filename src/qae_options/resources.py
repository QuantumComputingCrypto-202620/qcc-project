"""Circuit cost of the amplitude-estimation circuits (qcc_common.metrics settings: linear
coupling, basis {rz, sx, x, ecr}, optimization level 2, fixed seed).

Grover powers are too deep to transpile whole at large k, so costs are composed from transpiled
components: depth(Q^k A) ≈ depth(A) + k·depth(Q), and canonical QAE with m evaluation qubits ≈
1 + depth(A) + Σⱼ 2^j·depth(cQ) + depth(QFT⁻¹ₘ). Composition ignores parallelism across
blocks, so it is an upper bound on depth; ``composition_check`` compares it with direct
transpilation at small k.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

from qae_options.payoffs import PricingProblem
from qcc_common.metrics import circuit_metrics

KEYS = ("depth", "two_qubit_gates")


def _cost(circuit: QuantumCircuit) -> dict:
    m = circuit_metrics(circuit)
    return {
        "num_qubits": m["num_qubits"],
        "depth": m["depth"],
        "two_qubit_gates": m["two_qubit_gates"],
    }


def component_costs(problem: PricingProblem, max_eval_qubits: int = 0) -> dict:
    """Costs of A, Q, controlled-Q and QFT⁻¹ₘ for m = 1..max_eval_qubits."""
    grover = problem.estimation_problem().grover_operator
    costs = {"A": _cost(problem.a_operator), "Q": _cost(grover)}
    if max_eval_qubits:
        controlled = QuantumCircuit(grover.num_qubits + 1)
        controlled.append(grover.to_gate().control(1), range(grover.num_qubits + 1))
        costs["cQ"] = _cost(controlled)
        for m in range(1, max_eval_qubits + 1):
            costs[f"iqft{m}"] = _cost(QFT(m, inverse=True, do_swaps=False).reverse_bits())
    return costs


def grover_power_cost(costs: dict, k: int) -> dict:
    """Q^k A: the circuit run by IQAE / MLAE at power k."""
    out = {"num_qubits": costs["A"]["num_qubits"]}
    for key in KEYS:
        out[key] = costs["A"][key] + k * costs["Q"][key]
    return out


def qae_cost(costs: dict, m: int) -> dict:
    """Canonical QAE with m evaluation qubits (QPE over Q)."""
    out = {"num_qubits": costs["A"]["num_qubits"] + m}
    iqft = costs[f"iqft{m}"]
    for key in KEYS:
        powers = (2**m - 1) * costs["cQ"][key]
        out[key] = costs["A"][key] + powers + iqft[key] + (1 if key == "depth" else 0)
    return out


def composition_check(problem: PricingProblem, costs: dict, powers=(1, 2, 4)) -> list[dict]:
    """Composed vs directly transpiled cost of Q^k A for small k."""
    grover = problem.estimation_problem().grover_operator
    rows = []
    for k in powers:
        circuit = problem.a_operator.copy()
        circuit.compose(grover.power(k), inplace=True)
        direct = _cost(circuit)
        composed = grover_power_cost(costs, k)
        rows.append(
            {
                "k": k,
                **{f"direct_{key}": direct[key] for key in KEYS},
                **{f"composed_{key}": composed[key] for key in KEYS},
            }
        )
    return rows
