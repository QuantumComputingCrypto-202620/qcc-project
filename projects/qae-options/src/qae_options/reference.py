"""Classical reference prices (price units): closed forms for lognormal mixtures, numerical
integration, and the discretized-exact price on a grid."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import integrate, stats

from qae_options.distributions import Grid, LogNormal, LogNormalMixture


def black_scholes_call(s0: float, strike: float, r: float, sigma: float, maturity: float) -> float:
    d1 = (np.log(s0 / strike) + (r + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return float(s0 * stats.norm.cdf(d1) - strike * np.exp(-r * maturity) * stats.norm.cdf(d2))


def mixture_call(dist: LogNormalMixture, strike: float, r: float, maturity: float) -> float:
    """e^{−rT} E[max(S − K, 0)]: the weight-averaged Black-Scholes formula per component."""
    total = 0.0
    for w, mu, s in dist.components:
        d2 = (mu - np.log(strike)) / s
        total += w * (
            np.exp(mu + 0.5 * s**2) * stats.norm.cdf(d2 + s) - strike * stats.norm.cdf(d2)
        )
    return float(np.exp(-r * maturity) * total)


def mixture_digital(
    dist: LogNormalMixture, strike: float, r: float, maturity: float, payout: float = 1.0
) -> float:
    """e^{−rT} Q P(S ≥ K) = Σⱼ wⱼ e^{−rT} Q Φ(d₂ⱼ)."""
    p = sum(w * stats.norm.cdf((mu - np.log(strike)) / s) for w, mu, s in dist.components)
    return float(np.exp(-r * maturity) * payout * p)


def quad_price(
    dist: LogNormalMixture, payoff: Callable[[float], float], r: float, maturity: float
) -> float:
    """e^{−rT} ∫ f(s) p(s) ds by adaptive quadrature over the bulk of the distribution."""
    low, high = dist.bounds((1e-12, 1 - 1e-12))
    kinks = [mu for _, mu, _ in dist.components]
    value, _ = integrate.quad(
        lambda s: payoff(s) * float(dist.pdf(s)), low, high, points=np.exp(kinks), limit=400
    )
    return float(np.exp(-r * maturity) * value)


def discretized_exact(
    grid: Grid, payoff: Callable[[np.ndarray], np.ndarray], r: float, maturity: float
) -> float:
    """e^{−rT} Σᵢ pᵢ f(xᵢ): the error baseline for the estimators."""
    return float(np.exp(-r * maturity) * grid.probs @ payoff(grid.x))


def barrier_digital_price(
    dist: LogNormal, barrier: float, strike: float, t1: float, payout: float = 1.0
) -> float:
    """Continuous price of the 2-date up-and-out digital: e^{−rT} Q P(S_t1 < B, S_T ≥ K)."""
    s1 = dist.sigma * np.sqrt(t1)
    s2 = dist.sigma * np.sqrt(dist.maturity - t1)
    m1 = np.log(dist.s0) + (dist.r - 0.5 * dist.sigma**2) * t1
    drift2 = (dist.r - 0.5 * dist.sigma**2) * (dist.maturity - t1)
    zb = (np.log(barrier) - m1) / s1

    def integrand(z):
        log_s1 = m1 + s1 * z
        return stats.norm.pdf(z) * stats.norm.cdf((log_s1 + drift2 - np.log(strike)) / s2)

    value, _ = integrate.quad(integrand, -12, zb, limit=200)
    return float(np.exp(-dist.r * dist.maturity) * payout * value)
