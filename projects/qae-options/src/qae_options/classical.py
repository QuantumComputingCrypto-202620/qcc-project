"""Classical estimators: Monte Carlo on the grid and quasi-Monte Carlo (Sobol) on the continuous
distribution. One sample = one oracle query. Prices are discounted, in price units."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import qmc

from qae_options.distributions import Grid, LogNormalMixture


def monte_carlo(
    grid: Grid,
    payoff_values: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
    discount: float,
) -> float:
    """Average of f over ``num_samples`` grid indices drawn from {pᵢ} (via one multinomial)."""
    counts = rng.multinomial(int(num_samples), grid.probs)
    return float(discount * counts @ payoff_values / num_samples)


def quasi_monte_carlo(
    dist: LogNormalMixture,
    payoff: Callable[[np.ndarray], np.ndarray],
    num_samples: int,
    seed: int,
    discount: float,
) -> float:
    """Scrambled Sobol points mapped through the inverse CDF of the continuous distribution.

    ``num_samples`` must be a power of two. The estimate targets the *continuous* price.
    """
    m = int(np.log2(num_samples))
    if 2**m != num_samples:
        raise ValueError("Sobol sample sizes must be powers of two")
    u = qmc.Sobol(d=1, scramble=True, seed=seed).random_base2(m).ravel()
    return float(discount * np.mean(payoff(dist.ppf(u))))
