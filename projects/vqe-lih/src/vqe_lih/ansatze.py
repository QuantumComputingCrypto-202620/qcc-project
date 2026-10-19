"""Ansatz circuits: UCCSD and a hardware-efficient ansatz (HEA), both starting from the
Hartree-Fock bitstring by default."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import efficient_su2
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

from vqe_lih.hamiltonian import QubitHamiltonian

SIMULATION_BASIS = ("rz", "sx", "x", "cx")


@dataclass(frozen=True)
class AnsatzSpec:
    """``type`` is ``"uccsd"`` or ``"hea"``; ``initial_state`` is ``"hf"`` or ``"zero"``."""

    type: str
    reps: int = 1
    initial_state: str = "hf"
    rotations: tuple[str, ...] = ("ry", "rz")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnsatzSpec:
        return cls(
            type=d["type"],
            reps=int(d.get("reps", 1)),
            initial_state=d.get("initial_state", "hf"),
            rotations=tuple(d.get("rotations", ("ry", "rz"))),
        )

    @property
    def label(self) -> str:
        if self.type == "uccsd":
            return "uccsd" if self.reps == 1 else f"uccsd_r{self.reps}"
        suffix = "" if self.initial_state == "hf" else f"_{self.initial_state}"
        return f"hea_r{self.reps}{suffix}"


def hartree_fock_circuit(ham: QubitHamiltonian) -> QuantumCircuit:
    return HartreeFock(ham.num_spatial_orbitals, ham.num_particles, ham.mapper)


def build_uccsd(ham: QubitHamiltonian, reps: int = 1) -> QuantumCircuit:
    """UCCSD with ``reps`` Trotter steps on top of the HF state."""
    return UCCSD(
        ham.num_spatial_orbitals,
        ham.num_particles,
        ham.mapper,
        reps=reps,
        initial_state=hartree_fock_circuit(ham),
    )


def build_hea(
    ham: QubitHamiltonian,
    reps: int,
    initial_state: str = "hf",
    rotations: tuple[str, ...] = ("ry", "rz"),
) -> QuantumCircuit:
    """``efficient_su2`` with linear CX entanglement, prefixed by the HF bitstring.

    Note: with all angles zero this is *not* the HF state, because the CX layers permute the
    HF bitstring into another determinant.
    """
    body = efficient_su2(
        ham.num_qubits, su2_gates=list(rotations), entanglement="linear", reps=reps
    )
    if initial_state == "zero":
        return body
    if initial_state != "hf":
        raise ValueError(f"unknown initial state {initial_state!r}")
    return hartree_fock_circuit(ham).compose(body)


def build_ansatz(spec: AnsatzSpec, ham: QubitHamiltonian) -> QuantumCircuit:
    if spec.type == "uccsd":
        return build_uccsd(ham, spec.reps)
    if spec.type == "hea":
        return build_hea(ham, spec.reps, spec.initial_state, spec.rotations)
    raise ValueError(f"unknown ansatz type {spec.type!r}")


def compile_for_simulation(circuit: QuantumCircuit) -> QuantumCircuit:
    """Unroll library blocks once, so each energy evaluation does not re-synthesize them.

    Parameters keep their names, so parameter vectors line up with the original circuit.
    """
    compiled = transpile(
        circuit, basis_gates=list(SIMULATION_BASIS), optimization_level=1, seed_transpiler=0
    )
    if compiled.num_parameters != circuit.num_parameters:
        raise RuntimeError("transpilation removed ansatz parameters")
    return compiled
