import pytest

from vqe_lih.hamiltonian import LIH_2E3O, build_hamiltonian


@pytest.fixture(scope="session")
def h2():
    return build_hamiltonian("H2", 0.735)


@pytest.fixture(scope="session")
def lih():
    return build_hamiltonian("LiH", 1.6, active_space=LIH_2E3O)
