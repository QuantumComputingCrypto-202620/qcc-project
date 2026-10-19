import numpy as np
import pytest

from qcc_common.metrics import circuit_metrics
from vqe_lih.ansatze import AnsatzSpec, build_ansatz, build_hea, build_uccsd, compile_for_simulation
from vqe_lih.solver import exact_expectations


def test_uccsd_zero_parameters_reproduce_hf(lih):
    ansatz = compile_for_simulation(build_uccsd(lih))
    (energy,) = exact_expectations(ansatz, [lih.qubit_op], np.zeros(ansatz.num_parameters))
    assert energy + lih.energy_shift == pytest.approx(lih.hf_energy, abs=1e-9)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (AnsatzSpec("uccsd"), 8),  # LiH (2e, 3o): 4 singles + 4 doubles
        (AnsatzSpec("hea", reps=1), 24),  # 2 rotations x 6 qubits x (reps + 1)
        (AnsatzSpec("hea", reps=3), 48),
    ],
)
def test_parameter_counts(lih, spec, expected):
    metrics = circuit_metrics(build_ansatz(spec, lih))
    assert metrics["num_parameters"] == expected
    assert metrics["num_qubits"] == 6


def test_hea_is_shallower_than_uccsd(lih):
    uccsd = circuit_metrics(build_uccsd(lih))
    hea = circuit_metrics(build_hea(lih, reps=1))
    assert hea["two_qubit_gates"] < uccsd["two_qubit_gates"]


def test_hea_zero_angles_are_not_hf(lih):
    ansatz = build_hea(lih, reps=1)
    (energy,) = exact_expectations(ansatz, [lih.qubit_op], np.zeros(ansatz.num_parameters))
    assert abs(energy + lih.energy_shift - lih.hf_energy) > 1e-3


def test_labels():
    assert AnsatzSpec("uccsd").label == "uccsd"
    assert AnsatzSpec("hea", reps=2).label == "hea_r2"
    assert AnsatzSpec("hea", reps=2, initial_state="zero").label == "hea_r2_zero"
