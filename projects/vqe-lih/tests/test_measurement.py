import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from vqe_lih.ansatze import build_uccsd, compile_for_simulation
from vqe_lih.measurement import ShotEstimator, group_expectation, qwc_groups
from vqe_lih.solver import exact_expectations


def test_shot_estimator_is_unbiased_and_seeded(h2):
    ansatz = compile_for_simulation(build_uccsd(h2))
    params = np.array([0.0, 0.0, -0.11])
    (exact,) = exact_expectations(ansatz, [h2.qubit_op], params)
    estimator = ShotEstimator(shots=20_000, seed=5)
    result = estimator.run([(ansatz, h2.qubit_op, params)]).result()[0]
    assert result.data.stds > 0
    assert abs(result.data.evs - exact) < 5 * result.data.stds
    again = ShotEstimator(shots=20_000, seed=5).run([(ansatz, h2.qubit_op, params)]).result()[0]
    assert again.data.evs == result.data.evs
    groups = len(qwc_groups(h2.qubit_op))
    assert estimator.total_shots == 20_000 * groups


def test_shot_estimator_batches_parameters(h2):
    ansatz = compile_for_simulation(build_uccsd(h2))
    params = np.zeros((2, ansatz.num_parameters))
    result = ShotEstimator(shots=100, seed=1).run([(ansatz, h2.qubit_op, params)]).result()[0]
    assert result.data.evs.shape == (2,)
    assert result.data.evs[0] != result.data.evs[1]  # independent shot noise


def test_readout_inversion_recovers_ideal_value():
    group = SparsePauliOp(["Z"])
    counts = {"0": 900, "1": 100}  # |0> read as 1 with probability 0.1
    confusion = np.array([[0.9, 0.2], [0.1, 0.8]])
    raw, _ = group_expectation(counts, group)
    mitigated, _ = group_expectation(counts, group, [np.linalg.inv(confusion)])
    assert raw == pytest.approx(0.8)
    assert mitigated == pytest.approx(1.0)
