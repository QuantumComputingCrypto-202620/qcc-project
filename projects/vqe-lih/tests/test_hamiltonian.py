import numpy as np
import pytest

from vqe_lih.hamiltonian import (
    LIH_2E3O,
    LIH_4E4O,
    build_hamiltonian,
    select_sigma_active_orbitals,
)
from vqe_lih.molecules import geometry, warm_start_order
from vqe_lih.reference import exact_energy, fci_energy


def test_h2_exact_matches_fci(h2):
    assert h2.num_qubits == 4
    assert abs(exact_energy(h2).energy - fci_energy("H2", 0.735)) < 1e-6


def test_lih_default_active_space(lih):
    assert lih.num_qubits == 6
    assert lih.num_particles == (1, 1)
    assert lih.active_orbitals == [0, 1, 4]  # 2σ, 3σ, 4σ after freezing Li 1s; π dropped
    exact = exact_energy(lih)
    assert exact.num_particles == pytest.approx(2.0)
    assert exact.s_squared == pytest.approx(0.0, abs=1e-8)
    # Active-space energy lies between HF and full FCI.
    assert fci_energy("LiH", 1.6) < exact.energy < lih.hf_energy


def test_lih_four_electron_sigma_space():
    ham = build_hamiltonian("LiH", 1.6, active_space=LIH_4E4O)
    assert ham.num_qubits == 8
    assert ham.num_particles == (2, 2)
    assert ham.active_orbitals == [0, 1, 2, 5]


def test_parity_mapper_two_qubit_reduction(lih):
    parity = build_hamiltonian("LiH", 1.6, active_space=LIH_2E3O, mapper="parity")
    assert parity.num_qubits == 4
    assert exact_energy(parity).energy == pytest.approx(exact_energy(lih).energy, abs=1e-9)


def test_sigma_selection_requires_enough_orbitals():
    with pytest.raises(ValueError):
        select_sigma_active_orbitals([0, 1], num_occupied=1, num_electrons=2, num_orbitals=3)


def test_geometry_and_warm_start_order():
    assert geometry("LiH", 1.5) == "Li 0 0 0; H 0 0 1.5"
    with pytest.raises(ValueError):
        geometry("LiH", -1.0)
    order = warm_start_order([0.8, 1.0, 1.6, 2.0, 3.0], start=1.55)
    assert order == [(1.6, None), (2.0, 1.6), (3.0, 2.0), (1.0, 1.6), (0.8, 1.0)]
    assert sorted(r for r, _ in order) == [0.8, 1.0, 1.6, 2.0, 3.0]
    assert np.isclose(order[0][0], 1.6)
