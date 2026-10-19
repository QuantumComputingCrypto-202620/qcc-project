"""Shot-based energy estimation by measuring qubit-wise-commuting groups of Pauli terms.

Aer's ``EstimatorV2`` with ``precision > 0`` adds Gaussian noise to the exact value (and draws
the same noise on every call), so it does not model shot noise. :class:`ShotEstimator` samples
measurement outcomes instead: every group is measured with ``shots`` shots per evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, PrimitiveJob, PrimitiveResult
from qiskit.primitives.containers import DataBin, EstimatorPubLike, PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


def qwc_groups(op: SparsePauliOp) -> list[SparsePauliOp]:
    """Partition ``op`` into groups of qubit-wise commuting Pauli terms."""
    return op.group_commuting(qubit_wise=True)


def basis_rotation(group: SparsePauliOp) -> QuantumCircuit:
    """Rotate each qubit so every Pauli in ``group`` becomes a product of Z's."""
    n = group.num_qubits
    xs = np.any(group.paulis.x, axis=0)
    zs = np.any(group.paulis.z & group.paulis.x, axis=0)
    rotation = QuantumCircuit(n)
    for q in range(n):
        if zs[q]:  # Y basis
            rotation.sdg(q)
            rotation.h(q)
        elif xs[q]:
            rotation.h(q)
    return rotation


def measurement_circuit(circuit: QuantumCircuit, group: SparsePauliOp) -> QuantumCircuit:
    """``circuit`` + basis rotation + measurement of qubit q into clbit q."""
    measured = circuit.compose(basis_rotation(group))
    measured.measure_all()
    return measured


def pauli_masks(group: SparsePauliOp) -> np.ndarray:
    """Bit mask of the qubits each Pauli acts on (bit q = qubit q)."""
    support = group.paulis.x | group.paulis.z
    weights = 1 << np.arange(group.num_qubits, dtype=np.int64)
    return support.astype(np.int64) @ weights


def group_expectation(
    counts: dict[str, int],
    group: SparsePauliOp,
    inverse_confusion: Sequence[np.ndarray] | None = None,
) -> tuple[float, float]:
    """Estimate ``<group>`` from measurement ``counts``; returns (mean, variance of the mean).

    ``inverse_confusion[q]`` (2x2, columns = prepared state) undoes readout error on clbit q,
    assuming independent readout errors; the result is then a quasi-probability estimate.
    """
    shots = sum(counts.values())
    masks = pauli_masks(group)
    coeffs = np.real(group.coeffs)
    if inverse_confusion is None:
        outcomes = np.array([int(k.replace(" ", ""), 2) for k in counts], dtype=np.int64)
        weights = np.fromiter(counts.values(), dtype=float) / shots
    else:
        n = group.num_qubits
        probs = np.zeros(2**n)
        for key, c in counts.items():
            probs[int(key.replace(" ", ""), 2)] += c / shots
        tensor = probs.reshape((2,) * n)  # axis 0 = most significant bit = qubit n-1
        for q in range(n):
            axis = n - 1 - q
            tensor = np.moveaxis(
                np.tensordot(inverse_confusion[q], tensor, axes=([1], [axis])), 0, axis
            )
        weights = tensor.reshape(-1)
        outcomes = np.arange(2**n, dtype=np.int64)
    parities = (np.bitwise_count(outcomes[:, None] & masks[None, :]) & 1).astype(np.int64)
    values = (1 - 2 * parities) @ coeffs
    mean = float(weights @ values)
    variance = float(max(np.abs(weights) @ values**2 - mean**2, 0.0)) / shots
    return mean, variance


class ShotEstimator(BaseEstimatorV2):
    """Estimator V2 that samples ``shots`` shots per measurement group per evaluation.

    Each ``run`` draws fresh simulator seeds from a generator seeded with ``seed``, so runs are
    reproducible but evaluations have independent shot noise.
    """

    def __init__(self, shots: int, seed: int | None = None, backend: AerSimulator | None = None):
        if shots < 1:
            raise ValueError("shots must be positive")
        self.shots = int(shots)
        self._rng = np.random.default_rng(seed)
        self._backend = backend or AerSimulator(method="statevector")
        self._cache: dict[tuple[int, tuple], tuple[list[SparsePauliOp], list[QuantumCircuit]]] = {}
        self.total_shots = 0

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        coerced = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    def _groups(self, circuit: QuantumCircuit, obs: dict[str, float]):
        key = (id(circuit), tuple(sorted(obs.items())))
        if key not in self._cache:
            op = SparsePauliOp.from_list(list(obs.items()), num_qubits=circuit.num_qubits)
            groups = qwc_groups(op.simplify())
            self._cache[key] = (groups, [measurement_circuit(circuit, g) for g in groups])
        return self._cache[key]

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        circuit = pub.circuit
        param_shape = pub.parameter_values.shape
        values = pub.parameter_values.as_array(circuit.parameters).reshape(
            -1, circuit.num_parameters
        )
        param_index = np.arange(values.shape[0]).reshape(param_shape)
        bc_index, bc_obs = np.broadcast_arrays(param_index, pub.observables)
        evs = np.zeros(bc_index.shape)
        stds = np.zeros(bc_index.shape)
        for idx in np.ndindex(*bc_index.shape):
            groups, circuits = self._groups(circuit, bc_obs[idx])
            binds = values[bc_index[idx]]
            bound = (
                [c.assign_parameters(binds) for c in circuits]
                if circuit.num_parameters
                else circuits
            )
            seed = int(self._rng.integers(2**31 - 1))
            result = self._backend.run(bound, shots=self.shots, seed_simulator=seed).result()
            self.total_shots += self.shots * len(bound)
            mean, var = 0.0, 0.0
            for i, group in enumerate(groups):
                m, v = group_expectation(result.get_counts(i), group)
                mean += m
                var += v
            evs[idx], stds[idx] = mean, np.sqrt(var)
        data = DataBin(evs=evs, stds=stds, shape=evs.shape)
        return PubResult(data, metadata={"shots": self.shots, "circuit_metadata": circuit.metadata})
