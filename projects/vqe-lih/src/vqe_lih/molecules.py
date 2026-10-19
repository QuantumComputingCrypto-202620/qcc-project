"""Molecular geometries and bond-length sweeps. All lengths are in Å."""

from __future__ import annotations

from collections.abc import Sequence

# Approximate STO-3G equilibrium bond lengths (Å); used to start warm-start sweeps.
EQUILIBRIUM: dict[str, float] = {"H2": 0.735, "LiH": 1.6, "BeH2": 1.33}


def geometry(molecule: str, bond_length: float) -> str:
    """PySCF atom string with the molecule along z.

    ``LiH``: Li at the origin, H at (0, 0, R). ``BeH2``: linear H-Be-H, both Be-H bonds = R.
    """
    r = float(bond_length)
    if r <= 0:
        raise ValueError(f"bond length must be positive, got {bond_length}")
    if molecule == "H2":
        return f"H 0 0 0; H 0 0 {r}"
    if molecule == "LiH":
        return f"Li 0 0 0; H 0 0 {r}"
    if molecule == "BeH2":
        return f"Be 0 0 0; H 0 0 {-r}; H 0 0 {r}"
    raise ValueError(f"unknown molecule {molecule!r}")


def warm_start_order(
    bond_lengths: Sequence[float], start: float
) -> list[tuple[float, float | None]]:
    """Order in which to visit ``bond_lengths`` for a warm-start sweep.

    Starts at the grid point closest to ``start`` (the equilibrium), sweeps outwards to larger
    R, then from the start point inwards to smaller R. Each entry is ``(R, R_previous)``, where
    ``R_previous`` is the neighbour whose optimal parameters seed this point (``None`` for the
    first point).
    """
    grid = sorted(set(float(r) for r in bond_lengths))
    if not grid:
        return []
    i0 = min(range(len(grid)), key=lambda i: abs(grid[i] - start))
    order: list[tuple[float, float | None]] = [(grid[i0], None)]
    order += [(grid[i], grid[i - 1]) for i in range(i0 + 1, len(grid))]
    order += [(grid[i], grid[i + 1]) for i in range(i0 - 1, -1, -1)]
    return order
