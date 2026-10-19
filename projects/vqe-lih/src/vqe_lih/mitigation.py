"""Noisy-simulator energy evaluation with readout mitigation and zero-noise extrapolation (E7).

Energies are evaluated at fixed (simulator-optimized) parameters on an Aer simulator that
mimics a fake IBM backend. Mitigation methods:

- readout: per-qubit confusion matrices from |0...0> / |1...1> calibration, inverted on the
  measured distribution (assumes independent readout errors);
- ZNE: local folding of every two-qubit gate (G -> G G† G ...) at odd scale factors, then
  linear and Richardson extrapolation to zero noise. Single-qubit gate noise is not amplified.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from qcc_common.metrics import two_qubit_gate_count
from vqe_lih.measurement import group_expectation, measurement_circuit, qwc_groups


def fold_two_qubit_gates(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """Replace each two-qubit gate G by G (G† G)^k with k = (scale_factor - 1) / 2."""
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError("scale factor must be an odd positive integer")
    folds = (scale_factor - 1) // 2
    folded = circuit.copy_empty_like()
    for inst in circuit.data:
        folded.append(inst)
        op = inst.operation
        if op.num_qubits == 2 and not getattr(op, "_directive", False) and op.name != "measure":
            for _ in range(folds):
                folded.append(op.inverse(), inst.qubits, inst.clbits)
                folded.append(op, inst.qubits, inst.clbits)
    return folded


def extrapolate_to_zero(
    scale_factors: Sequence[float], values: Sequence[float], degree: int
) -> float:
    """Polynomial fit of ``values`` vs ``scale_factors`` evaluated at 0.

    ``degree=1`` is linear extrapolation; ``degree=len(values)-1`` is Richardson extrapolation.
    """
    coeffs = np.polyfit(np.asarray(scale_factors, float), np.asarray(values, float), degree)
    return float(np.polyval(coeffs, 0.0))


def measured_physical_qubits(circuit: QuantumCircuit) -> dict[int, int]:
    """Map clbit index -> physical qubit index for every measurement in a transpiled circuit."""
    mapping = {}
    for inst in circuit.data:
        if inst.operation.name == "measure":
            mapping[circuit.find_bit(inst.clbits[0]).index] = circuit.find_bit(inst.qubits[0]).index
    return mapping


def calibrate_readout(
    backend: AerSimulator, physical_qubits: Sequence[int], shots: int, seed: int
) -> dict[int, np.ndarray]:
    """2x2 confusion matrix ``A[measured, prepared]`` per physical qubit."""
    n = len(physical_qubits)
    circuits = []
    for prepared in (0, 1):
        qc = QuantumCircuit(backend.num_qubits, n)
        if prepared:
            qc.x(list(physical_qubits))
        qc.measure(list(physical_qubits), range(n))
        circuits.append(qc)
    result = backend.run(circuits, shots=shots, seed_simulator=seed).result()
    confusion = {q: np.zeros((2, 2)) for q in physical_qubits}
    for prepared in (0, 1):
        for key, count in result.get_counts(prepared).items():
            bits = key.replace(" ", "")[::-1]  # bits[i] = clbit i
            for i, q in enumerate(physical_qubits):
                confusion[q][int(bits[i]), prepared] += count / shots
    return confusion


def noisy_energy(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend: AerSimulator,
    shots: int,
    seed: int,
    scale_factors: Sequence[int] = (1, 3, 5),
) -> dict:
    """Estimate ``<observable>`` for a bound ``circuit`` on a noisy ``backend``.

    Returns raw and readout-mitigated estimates at each scale factor, their ZNE
    extrapolations, and the transpiled-circuit cost on the device.
    """
    if circuit.num_parameters:
        raise ValueError("bind the circuit parameters first")
    base = transpile(circuit, backend, optimization_level=3, seed_transpiler=seed)
    layout = base.layout.initial_index_layout(filter_ancillas=True)
    groups = qwc_groups(observable)
    measured = [
        transpile(
            measurement_circuit(circuit, g),
            backend,
            initial_layout=layout,
            optimization_level=3,
            seed_transpiler=seed,
        )
        for g in groups
    ]
    clbit_maps = [measured_physical_qubits(c) for c in measured]
    physical = sorted({q for m in clbit_maps for q in m.values()})
    confusion = calibrate_readout(backend, physical, shots, seed)
    inverses = {q: np.linalg.inv(a) for q, a in confusion.items()}

    rng = np.random.default_rng(seed)
    raw, mitigated, std = {}, {}, {}
    for factor in scale_factors:
        folded = [fold_two_qubit_gates(c, factor) for c in measured]
        result = backend.run(folded, shots=shots, seed_simulator=int(rng.integers(2**31))).result()
        e_raw = e_mit = var = 0.0
        for i, (group, cmap) in enumerate(zip(groups, clbit_maps, strict=True)):
            counts = result.get_counts(i)
            m, v = group_expectation(counts, group)
            inv = [inverses[cmap[c]] for c in range(group.num_qubits)]
            mm, _ = group_expectation(counts, group, inv)
            e_raw, e_mit, var = e_raw + m, e_mit + mm, var + v
        raw[factor], mitigated[factor], std[factor] = e_raw, e_mit, float(np.sqrt(var))

    factors = list(scale_factors)
    deg_r = len(factors) - 1
    return {
        "raw": raw,
        "readout": mitigated,
        "std_raw": std,
        "zne_linear": extrapolate_to_zero(factors, [raw[f] for f in factors], 1),
        "zne_richardson": extrapolate_to_zero(factors, [raw[f] for f in factors], deg_r),
        "readout_zne_linear": extrapolate_to_zero(factors, [mitigated[f] for f in factors], 1),
        "readout_zne_richardson": extrapolate_to_zero(
            factors, [mitigated[f] for f in factors], deg_r
        ),
        "device_depth": max(c.depth() for c in measured),
        "device_two_qubit_gates": two_qubit_gate_count(base),
        "num_groups": len(groups),
        "readout_error_mean": float(np.mean([(a[1, 0] + a[0, 1]) / 2 for a in confusion.values()])),
    }
