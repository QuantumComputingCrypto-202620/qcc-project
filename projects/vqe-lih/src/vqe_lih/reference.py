"""Reference energies (Hartree): exact diagonalization of the qubit Hamiltonian, full-space
FCI and Hartree-Fock."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pyscf import fci, gto, scf
from qiskit_algorithms import NumPyMinimumEigensolver

from vqe_lih.hamiltonian import QubitHamiltonian
from vqe_lih.molecules import geometry


@dataclass(frozen=True)
class ExactResult:
    energy: float
    num_particles: float
    s_squared: float


def exact_energy(ham: QubitHamiltonian) -> ExactResult:
    """Lowest eigenvalue of the qubit Hamiltonian in the correct particle-number sector with
    ⟨S²⟩ = 0, plus ``energy_shift`` (total energy in Ha)."""
    solver = NumPyMinimumEigensolver(filter_criterion=ham.problem.get_default_filter_criterion())
    result = solver.compute_minimum_eigenvalue(ham.qubit_op, ham.aux_ops)
    aux = result.aux_operators_evaluated or {}
    return ExactResult(
        energy=float(result.eigenvalue.real) + ham.energy_shift,
        num_particles=float(aux["ParticleNumber"][0].real),
        s_squared=float(aux["AngularMomentum"][0].real),
    )


@lru_cache(maxsize=256)
def fci_energy(molecule: str, bond_length: float, basis: str = "sto3g") -> float:
    """Full-space FCI energy (no frozen core, no active space), in Ha."""
    mol = gto.M(atom=geometry(molecule, bond_length), basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    return float(fci.FCI(mf).kernel()[0])


def hf_energy(ham: QubitHamiltonian) -> float:
    """Restricted Hartree-Fock energy, in Ha."""
    return ham.hf_energy
