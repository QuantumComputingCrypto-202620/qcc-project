"""Circuit cost metrics computed with one fixed transpilation setup, so numbers are comparable
across projects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

DEFAULT_BASIS: tuple[str, ...] = ("rz", "sx", "x", "ecr")
DEFAULT_OPTIMIZATION_LEVEL = 2
DEFAULT_SEED_TRANSPILER = 1234


def two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    """Number of two-qubit operations, excluding directives such as barriers."""
    return sum(
        1
        for inst in circuit.data
        if inst.operation.num_qubits == 2 and not getattr(inst.operation, "_directive", False)
    )


def circuit_metrics(
    circuit: QuantumCircuit,
    basis: Sequence[str] = DEFAULT_BASIS,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
) -> dict[str, Any]:
    """Transpile ``circuit`` onto a linear coupling map and report its cost.

    Returns a dict with:

    - ``num_qubits``: logical qubits of the input circuit;
    - ``num_parameters``: unbound variational parameters;
    - ``logical_depth``: depth after unrolling to ``{u, cx}`` with no coupling map and no
      optimization, so library blocks are not counted as depth 1;
    - ``depth`` and ``two_qubit_gates``: of the transpiled circuit;
    - ``transpile``: the settings used, so they can be stored with every result.
    """
    coupling_map = CouplingMap.from_line(max(circuit.num_qubits, 1), bidirectional=True)
    transpiled = transpile(
        circuit,
        basis_gates=list(basis),
        coupling_map=coupling_map,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    return {
        "num_qubits": circuit.num_qubits,
        "num_parameters": circuit.num_parameters,
        "logical_depth": logical_depth(circuit),
        "depth": transpiled.depth(),
        "two_qubit_gates": two_qubit_gate_count(transpiled),
        "transpile": {
            "basis_gates": list(basis),
            "coupling_map": "linear",
            "optimization_level": optimization_level,
            "seed_transpiler": seed_transpiler,
        },
    }


def logical_depth(circuit: QuantumCircuit) -> int:
    """Depth of ``circuit`` unrolled to ``{u, cx}``, before mapping or optimization."""
    unrolled = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    return unrolled.depth()
