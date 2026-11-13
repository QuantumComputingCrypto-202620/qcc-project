"""Price distributions, their discretization on 2^n grid points, and the loader circuit.

Prices are in price units (S₀ = 100 in the default configs). Every distribution is a mixture of
lognormals given by its log-price components (weight, μ, s): log S ~ N(μ, s²).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from scipy import stats

DEFAULT_QUANTILES = (0.001, 0.999)


class LogNormalMixture:
    """Mixture of lognormals; subclasses define ``components``."""

    components: tuple[tuple[float, float, float], ...]

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        logx = np.log(np.maximum(x, 1e-300))
        out = sum(w * stats.norm.cdf((logx - mu) / s) for w, mu, s in self.components)
        return np.where(x > 0, out, 0.0)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        safe = np.maximum(x, 1e-300)
        out = sum(
            w * stats.norm.pdf((np.log(safe) - mu) / s) / (s * safe) for w, mu, s in self.components
        )
        return np.where(x > 0, out, 0.0)

    def ppf(self, q, iterations: int = 80):
        """Inverse CDF by vectorized bisection on log-price (exact enough for float64)."""
        q = np.asarray(q, dtype=float)
        lo = np.full(q.shape, min(mu - 12 * s for _, mu, s in self.components))
        hi = np.full(q.shape, max(mu + 12 * s for _, mu, s in self.components))
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            below = self.cdf(np.exp(mid)) < q
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return np.exp(0.5 * (lo + hi))

    def bounds(self, quantiles: tuple[float, float] = DEFAULT_QUANTILES) -> tuple[float, float]:
        low, high = self.ppf(np.array(quantiles))
        return float(low), float(high)


@dataclass(frozen=True)
class LogNormal(LogNormalMixture):
    """Terminal price of geometric Brownian motion: S_T = S₀ exp((r − σ²/2)T + σ√T Z)."""

    s0: float
    r: float
    sigma: float
    maturity: float

    @property
    def components(self):
        mu = np.log(self.s0) + (self.r - 0.5 * self.sigma**2) * self.maturity
        return ((1.0, float(mu), float(self.sigma * np.sqrt(self.maturity))),)


@dataclass(frozen=True)
class BimodalLogNormal(LogNormalMixture):
    """Two-component lognormal mixture: weight ``w`` on (μ₁, σ₁), 1 − w on (μ₂, σ₂), log-price."""

    w: float
    mu1: float
    sigma1: float
    mu2: float
    sigma2: float

    def __post_init__(self):
        if not 0 < self.w < 1:
            raise ValueError("mixture weight must be in (0, 1)")

    @property
    def components(self):
        return ((self.w, self.mu1, self.sigma1), (1 - self.w, self.mu2, self.sigma2))


@dataclass(frozen=True)
class Grid:
    """``x[i] = low + i (high − low) / (2^n − 1)``; ``probs[i]`` = mass of the cell around x[i]."""

    x: np.ndarray
    probs: np.ndarray
    low: float
    high: float

    @property
    def num_qubits(self) -> int:
        return int(np.log2(len(self.x)))

    def index_at_or_above(self, value: float) -> int:
        """Smallest grid index i with x[i] >= value (2^n if none)."""
        return int(np.searchsorted(self.x, value - 1e-12 * max(1.0, abs(value)), side="left"))


def cell_masses(dist: LogNormalMixture, x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Probability of the cells [mid_{i-1}, mid_i] around each grid point, renormalized to the
    truncated interval [low, high]."""
    edges = np.concatenate([[low], 0.5 * (x[1:] + x[:-1]), [high]])
    masses = np.diff(dist.cdf(edges))
    return masses / masses.sum()


def discretize(dist: LogNormalMixture, n: int, bounds: tuple[float, float] | None = None) -> Grid:
    """Discretize ``dist`` on 2^n equally spaced points over ``bounds`` (default: the 0.1 % and
    99.9 % quantiles)."""
    if n < 1:
        raise ValueError("need at least one qubit")
    low, high = bounds or dist.bounds()
    x = np.linspace(low, high, 2**n)
    return Grid(x=x, probs=cell_masses(dist, x, low, high), low=low, high=high)


@dataclass(frozen=True)
class JointGrid:
    """Two monitoring dates on a common price grid: ``probs[i, j]`` = P(S_t1 ≈ x[i], S_T ≈ x[j])."""

    x: np.ndarray
    probs: np.ndarray
    low: float
    high: float

    @property
    def num_qubits_per_date(self) -> int:
        return int(np.log2(len(self.x)))


def discretize_two_dates(
    dist: LogNormal, n: int, t1: float, bounds: tuple[float, float] | None = None
) -> JointGrid:
    """Discretize (S_t1, S_T) of the GBM ``dist`` on a common 2^n grid (Markov transition)."""
    low, high = bounds or dist.bounds()
    x = np.linspace(low, high, 2**n)
    first = cell_masses(LogNormal(dist.s0, dist.r, dist.sigma, t1), x, low, high)
    probs = np.empty((len(x), len(x)))
    for i, xi in enumerate(x):
        probs[i] = first[i] * cell_masses(
            LogNormal(xi, dist.r, dist.sigma, dist.maturity - t1), x, low, high
        )
    return JointGrid(x=x, probs=probs / probs.sum(), low=low, high=high)


def loader(probs: np.ndarray) -> QuantumCircuit:
    """Prepare Σ √p_i |i⟩ (qubit 0 = least significant bit) with a generic ``StatePreparation``.

    For a 2-D array ``probs[i, j]`` the index is ``i + 2^n j`` (first date in the low qubits).
    """
    flat = np.asarray(probs, dtype=float).reshape(-1, order="F")
    num_qubits = int(np.log2(len(flat)))
    if 2**num_qubits != len(flat):
        raise ValueError("number of probabilities must be a power of two")
    circuit = QuantumCircuit(num_qubits, name="P(S)")
    circuit.append(StatePreparation(np.sqrt(flat / flat.sum())), range(num_qubits))
    return circuit
