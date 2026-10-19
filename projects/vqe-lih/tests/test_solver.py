import numpy as np
import pytest

from vqe_lih.ansatze import build_uccsd, compile_for_simulation
from vqe_lih.reference import exact_energy
from vqe_lih.solver import CHEMICAL_ACCURACY, initial_point, run_vqe


def test_uccsd_vqe_on_h2_reaches_chemical_accuracy(h2):
    ansatz = compile_for_simulation(build_uccsd(h2))
    e_exact = exact_energy(h2).energy
    record = run_vqe(h2, ansatz, "cobyla", np.zeros(ansatz.num_parameters), e_exact=e_exact)
    assert 0 <= record.delta_e < CHEMICAL_ACCURACY
    assert record.chemical_accuracy
    assert not record.wrong_sector
    assert record.num_particles == pytest.approx(2.0, abs=1e-6)
    assert record.evals_to_chemical_accuracy is not None
    assert len(record.history) == record.nfev


def test_initial_points():
    assert np.all(initial_point("zeros", 3) == 0)
    a, b = initial_point("random", 5, seed=1), initial_point("random", 5, seed=1)
    assert np.array_equal(a, b) and np.std(a) < 0.3
    assert np.array_equal(initial_point("warm_start", 2, warm=[1.0, 2.0]), [1.0, 2.0])
    with pytest.raises(ValueError):
        initial_point("warm_start", 2)
    with pytest.raises(ValueError):
        initial_point("warm_start", 3, warm=[1.0, 2.0])
