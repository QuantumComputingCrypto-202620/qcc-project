from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2

from qcc_common.metrics import DEFAULT_BASIS, circuit_metrics, two_qubit_gate_count


def bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def test_bell_metrics():
    m = circuit_metrics(bell())
    assert m["num_qubits"] == 2
    assert m["num_parameters"] == 0
    assert m["two_qubit_gates"] == 1
    assert m["logical_depth"] == 2
    assert m["depth"] >= 2


def test_settings_are_recorded():
    m = circuit_metrics(bell(), optimization_level=1, seed_transpiler=5)
    assert m["transpile"] == {
        "basis_gates": list(DEFAULT_BASIS),
        "coupling_map": "linear",
        "optimization_level": 1,
        "seed_transpiler": 5,
    }


def test_metrics_are_deterministic():
    circuit = efficient_su2(4, reps=2, entanglement="linear")
    assert circuit_metrics(circuit) == circuit_metrics(circuit)


def test_parameterized_circuit():
    circuit = efficient_su2(4, reps=1, entanglement="linear")
    m = circuit_metrics(circuit)
    assert m["num_parameters"] == circuit.num_parameters == 16
    assert m["two_qubit_gates"] == 3


def test_routing_on_linear_coupling_adds_two_qubit_gates():
    # A triangle of CNOTs cannot be laid out on a line without a SWAP.
    qc = QuantumCircuit(3)
    for i, (a, b) in enumerate([(0, 1), (1, 2), (0, 2)]):
        qc.ry(Parameter(f"t{i}"), a)
        qc.cx(a, b)
    assert circuit_metrics(qc)["two_qubit_gates"] > 3


def test_barriers_are_not_two_qubit_gates():
    qc = bell()
    qc.barrier()
    assert two_qubit_gate_count(qc) == 1
