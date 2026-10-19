"""Hamiltonian pipeline: PySCF driver -> (freeze core) -> active space -> qubit mapping.

All energies are in Hartree and bond lengths in Å. ``energy_shift`` collects the nuclear
repulsion and the frozen-core / inactive-orbital energy, so ``<H_qubit> + energy_shift`` is the
total electronic energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pyscf import gto, scf
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer

from vqe_lih.molecules import geometry

PI_AO_SUFFIXES = ("px", "py")


@dataclass(frozen=True)
class ActiveSpace:
    """Reduction of the orbital space.

    ``orbitals``: ``None`` lets ``ActiveSpaceTransformer`` pick orbitals around the Fermi
    level; ``"sigma"`` keeps only σ orbitals (no p_x/p_y character, molecule along z), taking
    the highest occupied and lowest virtual ones; a list gives explicit indices (counted after
    the core is frozen).
    """

    label: str = "full"
    freeze_core: bool = False
    num_electrons: int | None = None
    num_spatial_orbitals: int | None = None
    orbitals: str | tuple[int, ...] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActiveSpace:
        orbitals = d.get("orbitals")
        if isinstance(orbitals, list):
            orbitals = tuple(int(i) for i in orbitals)
        return cls(
            label=d.get("label", "full"),
            freeze_core=bool(d.get("freeze_core", False)),
            num_electrons=d.get("num_electrons"),
            num_spatial_orbitals=d.get("num_spatial_orbitals"),
            orbitals=orbitals,
        )


@dataclass
class QubitHamiltonian:
    molecule: str
    bond_length: float
    basis: str
    active_space: ActiveSpace
    mapper_name: str
    qubit_op: SparsePauliOp
    energy_shift: float
    hf_energy: float
    num_spatial_orbitals: int
    num_particles: tuple[int, int]
    mapper: QubitMapper
    problem: ElectronicStructureProblem
    aux_ops: dict[str, SparsePauliOp]
    active_orbitals: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        return self.qubit_op.num_qubits

    @property
    def total_particles(self) -> int:
        return sum(self.num_particles)


def run_rhf(molecule: str, bond_length: float, basis: str = "sto3g") -> scf.hf.RHF:
    mol = gto.M(atom=geometry(molecule, bond_length), basis=basis, verbose=0)
    return scf.RHF(mol).run()


def sigma_orbital_indices(mf: scf.hf.RHF, tol: float = 1e-6) -> list[int]:
    """Indices of molecular orbitals with no p_x/p_y weight (σ orbitals of a z-axis molecule)."""
    labels = mf.mol.ao_labels()
    pi_aos = [i for i, label in enumerate(labels) if label.strip().endswith(PI_AO_SUFFIXES)]
    weight = (np.asarray(mf.mo_coeff)[pi_aos] ** 2).sum(axis=0)
    return [int(i) for i in np.flatnonzero(weight < tol)]


def select_sigma_active_orbitals(
    sigma: list[int], num_occupied: int, num_electrons: int, num_orbitals: int
) -> list[int]:
    """Pick the ``num_electrons // 2`` highest occupied and the lowest virtual σ orbitals."""
    n_occ = num_electrons // 2
    occupied = [i for i in sigma if i < num_occupied]
    virtual = [i for i in sigma if i >= num_occupied]
    n_virt = num_orbitals - n_occ
    if len(occupied) < n_occ or len(virtual) < n_virt:
        raise ValueError(
            f"not enough σ orbitals for ({num_electrons}e, {num_orbitals}o): "
            f"{len(occupied)} occupied, {len(virtual)} virtual"
        )
    return sorted(occupied[len(occupied) - n_occ :] + virtual[:n_virt])


def make_mapper(name: str, num_particles: tuple[int, int]) -> QubitMapper:
    if name == "jordan_wigner":
        return JordanWignerMapper()
    if name == "parity":
        return ParityMapper(num_particles=num_particles)  # two-qubit reduction
    raise ValueError(f"unknown mapper {name!r}")


def build_hamiltonian(
    molecule: str,
    bond_length: float,
    basis: str = "sto3g",
    active_space: ActiveSpace | None = None,
    mapper: str = "jordan_wigner",
) -> QubitHamiltonian:
    """Build the qubit Hamiltonian of ``molecule`` at ``bond_length`` (Å)."""
    active_space = active_space or ActiveSpace()
    driver = PySCFDriver(atom=geometry(molecule, bond_length), basis=basis)
    problem = driver.run()
    full_orbitals = problem.num_spatial_orbitals

    if active_space.freeze_core:
        problem = FreezeCoreTransformer().transform(problem)
    num_frozen = full_orbitals - problem.num_spatial_orbitals

    active_orbitals = None
    if active_space.num_electrons is not None:
        orbitals = active_space.orbitals
        if orbitals == "sigma":
            mf = run_rhf(molecule, bond_length, basis)
            if not np.isclose(mf.e_tot, problem.reference_energy, atol=1e-6):
                raise RuntimeError("PySCF RHF and the driver's RHF disagree; orbitals may differ")
            sigma = [i - num_frozen for i in sigma_orbital_indices(mf) if i >= num_frozen]
            orbitals = select_sigma_active_orbitals(
                sigma,
                num_occupied=problem.num_alpha,
                num_electrons=active_space.num_electrons,
                num_orbitals=active_space.num_spatial_orbitals,
            )
        active_orbitals = list(orbitals) if orbitals is not None else None
        problem = ActiveSpaceTransformer(
            active_space.num_electrons,
            active_space.num_spatial_orbitals,
            active_orbitals=active_orbitals,
        ).transform(problem)

    qubit_mapper = make_mapper(mapper, problem.num_particles)
    qubit_op = qubit_mapper.map(problem.hamiltonian.second_q_op())
    aux_ops = {}
    for prop in (problem.properties.particle_number, problem.properties.angular_momentum):
        if prop is not None:
            aux_ops.update({k: qubit_mapper.map(v) for k, v in prop.second_q_ops().items()})

    return QubitHamiltonian(
        molecule=molecule,
        bond_length=float(bond_length),
        basis=basis,
        active_space=active_space,
        mapper_name=mapper,
        qubit_op=qubit_op,
        energy_shift=float(sum(problem.hamiltonian.constants.values())),
        hf_energy=float(problem.reference_energy),
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=tuple(int(n) for n in problem.num_particles),
        mapper=qubit_mapper,
        problem=problem,
        aux_ops=aux_ops,
        active_orbitals=active_orbitals,
        metadata={
            "num_frozen_orbitals": num_frozen,
            "num_pauli_terms": len(qubit_op),
            "constants": {k: float(v) for k, v in problem.hamiltonian.constants.items()},
        },
    )


# Default LiH active space (SPEC §3): freeze Li 1s, keep the three σ orbitals -> 6 qubits (JW).
LIH_2E3O = ActiveSpace(
    label="2e3o", freeze_core=True, num_electrons=2, num_spatial_orbitals=3, orbitals="sigma"
)
# Four-electron σ space (week-4 decision): Li 1s stays active, π dropped -> 8 qubits (JW).
LIH_4E4O = ActiveSpace(
    label="4e4o", freeze_core=False, num_electrons=4, num_spatial_orbitals=4, orbitals="sigma"
)
