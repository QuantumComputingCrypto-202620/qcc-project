"""A seeded shot-based ``SamplerV2`` for amplitude estimation that reuses work across repetitions.

Noiseless (``backend=None``): the exact output distribution of each distinct circuit is computed
once and cached; every run then draws ``shots`` outcomes with ``rng.multinomial``. This is
statistically identical to sampling the circuit, so the R seeded repetitions of an experiment
cost one simulation per circuit.

The noiseless simulation is exact but works on unitaries instead of gates: the matrix of every
distinct sub-instruction (loader, comparator, reflections, Q, controlled Q^(2^j), ...) is built
once from its definition and cached, so Q^k costs k small matrix products instead of k copies
of a ~3000-gate circuit. ``tests/test_sampling.py`` checks it against Aer gate-level simulation.

Noisy (``backend`` given): each distinct circuit is transpiled once for the backend (cached) and
every run samples it on the backend with a fresh seed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import AnnotatedOperation, ControlledGate, Instruction
from qiskit.circuit.annotated_operation import ControlModifier, InverseModifier, PowerModifier
from qiskit.primitives import BaseSamplerV2, PrimitiveJob, PrimitiveResult
from qiskit.primitives.containers import BitArray, DataBin, SamplerPubLike, SamplerPubResult
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator

SIMULATION_BASIS = ("u", "cx")


def circuit_key(circuit: QuantumCircuit) -> tuple:
    """Structural key of a circuit's top-level instructions (names, wires, parameters)."""
    items = []
    for inst in circuit.data:
        params = []
        if isinstance(inst.operation, AnnotatedOperation):
            # Annotated operations carry no params: the modifiers (power, control) are the key.
            params.append(repr(inst.operation.modifiers))
            params.append(inst.operation.base_op.name)
        for p in getattr(inst.operation, "params", []):
            if isinstance(p, np.ndarray):
                params.append(tuple(np.round(p.ravel(), 12).tolist()))
            elif isinstance(p, complex | float | int):
                params.append(round(complex(p).real, 12) + 1j * round(complex(p).imag, 12))
            else:
                params.append(str(p))
        items.append(
            (
                inst.operation.name,
                tuple(circuit.find_bit(q).index for q in inst.qubits),
                tuple(circuit.find_bit(c).index for c in inst.clbits),
                tuple(params),
            )
        )
    return (circuit.num_qubits, circuit.num_clbits, tuple(items))


def _measured(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    """(qubit, clbit) for every measurement, in order."""
    return [
        (circuit.find_bit(inst.qubits[0]).index, circuit.find_bit(inst.clbits[0]).index)
        for inst in circuit.data
        if inst.operation.name == "measure"
    ]


def _strip_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    stripped = circuit.copy_empty_like()
    for inst in circuit.data:
        if inst.operation.name not in ("measure", "barrier"):
            stripped.append(inst)
    return stripped


_SKIP = {"measure", "barrier", "delay"}


def _operation_key(op: Instruction, memo: dict | None = None) -> tuple:
    """Structural key of an operation. ``memo`` (keyed by object id) lives for one circuit
    evaluation only: the same Q gate object appears 2^j times inside a Grover power, but a
    long-lived id cache would keep every circuit's gate objects alive."""
    if memo is not None:
        hit = memo.get(id(op))
        if hit is not None and hit[0] is op:
            return hit[1]
    key = _compute_operation_key(op, memo)
    if memo is not None:
        memo[id(op)] = (op, key)
    return key


def _compute_operation_key(op: Instruction, cache: dict | None) -> tuple:
    if isinstance(op, AnnotatedOperation):
        return ("annotated", repr(op.modifiers), _operation_key(op.base_op, cache))
    params = []
    for p in getattr(op, "params", []):
        if isinstance(p, np.ndarray):
            params.append(tuple(np.round(p.ravel(), 12).tolist()))
        else:
            try:
                params.append(complex(p))
            except TypeError:
                params.append(str(p))
    definition = None
    if not hasattr(op, "_standard_gate") or op._standard_gate is None:
        if isinstance(op, ControlledGate):
            definition = (
                "ctrl",
                op.num_ctrl_qubits,
                op.ctrl_state,
                _operation_key(op.base_gate, cache),
            )
        elif op.definition is not None:
            definition = tuple(
                (
                    _operation_key(inst.operation, cache),
                    tuple(op.definition.find_bit(q).index for q in inst.qubits),
                )
                for inst in op.definition.data
            )
    return (op.name, op.num_qubits, tuple(params), definition)


def _controlled(base: np.ndarray, num_ctrl: int, ctrl_state: int) -> np.ndarray:
    """Matrix of a gate whose first ``num_ctrl`` qubits control ``base`` (little-endian)."""
    num_target = int(np.log2(base.shape[0]))
    dim = 2 ** (num_ctrl + num_target)
    matrix = np.eye(dim, dtype=complex)
    idx = np.array([ctrl_state + (t << num_ctrl) for t in range(2**num_target)])
    matrix[np.ix_(idx, idx)] = base
    return matrix


def operation_matrix(op: Instruction, cache: dict, memo: dict | None = None) -> np.ndarray:
    """Exact unitary of ``op``, built from its definition and memoized in ``cache`` (by
    structure, so repeated or re-created identical gates share one entry)."""
    memo = {} if memo is None else memo
    key = ("matrix", _operation_key(op, memo))
    if key in cache:
        return cache[key]
    if isinstance(op, AnnotatedOperation):
        matrix = operation_matrix(op.base_op, cache, memo)
        for modifier in op.modifiers:
            if isinstance(modifier, ControlModifier):
                matrix = _controlled(matrix, modifier.num_ctrl_qubits, modifier.ctrl_state)
            elif isinstance(modifier, InverseModifier):
                matrix = matrix.conj().T
            elif isinstance(modifier, PowerModifier):
                matrix = np.linalg.matrix_power(matrix, int(modifier.power))
            else:
                raise NotImplementedError(f"modifier {modifier!r}")
    elif getattr(op, "_standard_gate", None) is not None:
        matrix = op.to_matrix()
    elif isinstance(op, ControlledGate) and op.base_gate is not None:
        base = operation_matrix(op.base_gate, cache, memo)
        matrix = _controlled(base, op.num_ctrl_qubits, op.ctrl_state)
    elif op.definition is not None:
        matrix = circuit_matrix(op.definition, cache, memo)
    else:  # e.g. high-level objects synthesized only by the transpiler
        wrapper = QuantumCircuit(op.num_qubits)
        wrapper.append(op, range(op.num_qubits))
        matrix = Operator(transpile(wrapper, basis_gates=list(SIMULATION_BASIS))).data
    cache[key] = np.asarray(matrix, dtype=complex)
    return cache[key]


def circuit_matrix(circuit: QuantumCircuit, cache: dict, memo: dict | None = None) -> np.ndarray:
    unitary = Operator(np.eye(2**circuit.num_qubits, dtype=complex))
    for inst in circuit.data:
        if inst.operation.name in _SKIP:
            continue
        qargs = [circuit.find_bit(q).index for q in inst.qubits]
        matrix = operation_matrix(inst.operation, cache, memo)
        unitary = unitary.compose(Operator(matrix), qargs=qargs)
    if circuit.global_phase:
        unitary = unitary * np.exp(1j * float(circuit.global_phase))
    return unitary.data


def exact_statevector(circuit: QuantumCircuit, cache: dict) -> Statevector:
    """Final state of ``circuit`` (measurements ignored) from |0...0>."""
    state = Statevector.from_int(0, 2**circuit.num_qubits)
    memo: dict = {}
    for inst in circuit.data:
        if inst.operation.name in _SKIP:
            continue
        qargs = [circuit.find_bit(q).index for q in inst.qubits]
        matrix = operation_matrix(inst.operation, cache, memo)
        state = state.evolve(Operator(matrix), qargs=qargs)
    return state


class ExactDistributionSampler(BaseSamplerV2):
    """Seeded SamplerV2; see the module docstring. ``cache`` may be shared between instances
    (e.g. one instance per repetition seed) as long as they run the same circuits."""

    def __init__(
        self,
        default_shots: int = 100,
        seed: int | None = None,
        backend: AerSimulator | None = None,
        cache: dict | None = None,
        seed_transpiler: int = 1234,
    ):
        self._default_shots = int(default_shots)
        self._rng = np.random.default_rng(seed)
        self._backend = backend
        self._cache = {} if cache is None else cache
        self._seed_transpiler = seed_transpiler
        self.total_shots = 0

    @property
    def default_shots(self) -> int:
        return self._default_shots

    def run(self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None):
        coerced = [SamplerPub.coerce(pub, shots or self._default_shots) for pub in pubs]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs: list[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    # -- distributions ------------------------------------------------------------------------

    def _exact_probabilities(self, circuit: QuantumCircuit) -> np.ndarray:
        """P(clbit string) as a vector indexed by the integer value of all clbits."""
        key = ("exact", circuit_key(circuit))
        if key not in self._cache:
            measured = _measured(circuit)
            clbit_order = sorted(measured, key=lambda qc: qc[1])
            state = exact_statevector(circuit, self._cache)
            probs_measured = state.probabilities([q for q, _ in clbit_order])
            full = np.zeros(2**circuit.num_clbits)
            positions = [c for _, c in clbit_order]
            for idx, p in enumerate(probs_measured):
                full[sum(((idx >> b) & 1) << c for b, c in enumerate(positions))] += p
            self._cache[key] = full / full.sum()
        return self._cache[key]

    def _noisy_counts(self, circuit: QuantumCircuit, shots: int) -> dict[int, int]:
        key = ("noisy", id(self._backend), circuit_key(circuit))
        if key not in self._cache:
            self._cache[key] = transpile(
                circuit, self._backend, optimization_level=1, seed_transpiler=self._seed_transpiler
            )
        seed = int(self._rng.integers(2**31 - 1))
        counts = self._backend.run(self._cache[key], shots=shots, seed_simulator=seed).result()
        return {int(k.replace(" ", ""), 2): v for k, v in counts.get_counts().items()}

    def _run_pub(self, pub: SamplerPub) -> SamplerPubResult:
        circuit = pub.circuit
        if circuit.num_parameters:
            circuit = circuit.assign_parameters(
                pub.parameter_values.as_array(circuit.parameters).ravel()
            )
        shots = pub.shots
        if self._backend is None:
            probs = self._exact_probabilities(circuit)
            draws = self._rng.multinomial(shots, probs)
            outcomes = np.repeat(np.arange(len(probs)), draws)
        else:
            counts = self._noisy_counts(circuit, shots)
            outcomes = np.repeat(list(counts), list(counts.values()))
        self.total_shots += shots
        data = {}
        for creg in circuit.cregs:
            positions = [circuit.find_bit(c).index for c in creg]
            values = sum(((outcomes >> p) & 1) << b for b, p in enumerate(positions))
            data[creg.name] = BitArray.from_samples(
                [int(v) for v in np.atleast_1d(values)], creg.size
            )
        return SamplerPubResult(DataBin(**data), metadata={"shots": shots})
