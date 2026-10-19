import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qcc_common.backends import noisy_simulator
from qcc_common.metrics import two_qubit_gate_count
from vqe_lih.ansatze import hartree_fock_circuit
from vqe_lih.mitigation import extrapolate_to_zero, fold_two_qubit_gates, noisy_energy
from vqe_lih.solver import exact_expectations


def test_folding_preserves_unitary_and_scales_two_qubit_gates():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.3, 1)
    qc.ecr(1, 0)
    folded = fold_two_qubit_gates(qc, 5)
    assert two_qubit_gate_count(folded) == 5 * two_qubit_gate_count(qc)
    assert Operator(folded).equiv(Operator(qc))
    with pytest.raises(ValueError):
        fold_two_qubit_gates(qc, 2)


def test_extrapolation():
    assert extrapolate_to_zero([1, 3, 5], [1.0, 2.0, 3.0], 1) == pytest.approx(0.5)
    assert extrapolate_to_zero([1, 3, 5], [1.0, 9.0, 25.0], 2) == pytest.approx(0.0, abs=1e-9)


def test_readout_mitigation_reduces_bias_on_hf_state(h2):
    hf = hartree_fock_circuit(h2).decompose()
    (ideal,) = exact_expectations(hf, [h2.qubit_op], np.zeros(0))
    res = noisy_energy(hf, h2.qubit_op, noisy_simulator(scale=1.0, seed=3), 20_000, 3, (1,))
    assert abs(res["readout"][1] - ideal) < abs(res["raw"][1] - ideal)
    assert res["readout_error_mean"] > 0
