import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qae_options.distributions import (
    BimodalLogNormal,
    LogNormal,
    discretize,
    discretize_two_dates,
    loader,
)
from qae_options.payoffs import barrier_digital, digital_call, european_call
from qae_options.reference import (
    barrier_digital_price,
    black_scholes_call,
    discretized_exact,
    mixture_call,
    mixture_digital,
    quad_price,
)

GBM = LogNormal(100.0, 0.04, 0.2, 1.0)
BIMODAL = BimodalLogNormal(0.4, np.log(85), 0.10, np.log(115), 0.10)


@pytest.mark.parametrize("dist", [GBM, BIMODAL])
def test_probabilities_and_loader(dist):
    grid = discretize(dist, 4)
    assert grid.probs.sum() == pytest.approx(1.0)
    assert np.all(grid.probs > 0)
    amplitudes = Statevector(loader(grid.probs)).data
    assert np.allclose(np.abs(amplitudes), np.sqrt(grid.probs), atol=1e-10)


def test_bounds_are_quantiles():
    low, high = BIMODAL.bounds()
    assert BIMODAL.cdf(low) == pytest.approx(0.001, abs=1e-9)
    assert BIMODAL.cdf(high) == pytest.approx(0.999, abs=1e-9)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_digital_amplitude_is_tail_probability(n):
    grid = discretize(BIMODAL, n)
    problem = digital_call(grid, 100.0, 0.03, 0.5)
    p_good = Statevector(problem.a_operator).probabilities([problem.objective_qubit])[1]
    expected = grid.probs[grid.x >= 100.0].sum()
    assert p_good == pytest.approx(expected, abs=1e-10)
    assert problem.exact_amplitude == pytest.approx(expected, abs=1e-12)
    assert problem.discretized_price == pytest.approx(np.exp(-0.015) * expected)


def test_discretized_call_converges_to_black_scholes():
    """The grid price converges to the price of the distribution truncated to [q_0.1%, q_99.9%];
    the remaining gap to Black-Scholes is the truncated upper tail (part of the discretization
    error that E1 reports)."""
    from scipy import integrate

    def call(x):
        return np.maximum(x - 100, 0)

    bs = black_scholes_call(100.0, 100.0, 0.04, 0.2, 1.0)
    low, high = GBM.bounds()
    mass = GBM.cdf(high) - GBM.cdf(low)
    integral, _ = integrate.quad(lambda s: call(s) * GBM.pdf(s), low, high, limit=200)
    truncated = np.exp(-0.04) * integral / mass
    errors = [
        abs(discretized_exact(discretize(GBM, n), call, 0.04, 1.0) - truncated)
        for n in (3, 5, 7, 9)
    ]
    assert all(b < a for a, b in zip(errors, errors[1:], strict=False))
    assert errors[-1] < 1e-3
    assert abs(truncated - bs) / bs < 0.01


def test_closed_forms_match_quadrature():
    for dist, r, t in ((GBM, 0.04, 1.0), (BIMODAL, 0.03, 0.5)):
        digital = mixture_digital(dist, 100.0, r, t)
        assert digital == pytest.approx(quad_price(dist, lambda s: float(s >= 100), r, t), rel=1e-6)
        call = mixture_call(dist, 100.0, r, t)
        assert call == pytest.approx(quad_price(dist, lambda s: max(s - 100, 0), r, t), rel=1e-6)
    assert mixture_call(GBM, 100.0, 0.04, 1.0) == pytest.approx(
        black_scholes_call(100.0, 100.0, 0.04, 0.2, 1.0), rel=1e-12
    )


def test_call_payoff_approximation_shrinks_with_c():
    grid = discretize(GBM, 4)
    gaps = [
        abs(
            european_call(grid, 100.0, 0.04, 1.0, c).exact_price
            - european_call(grid, 100.0, 0.04, 1.0, c).discretized_price
        )
        for c in (0.25, 0.05)
    ]
    assert gaps[1] < gaps[0]


def test_barrier_amplitude_and_reference():
    joint = discretize_two_dates(GBM, 3, 0.5)
    assert joint.probs.sum() == pytest.approx(1.0)
    problem = barrier_digital(joint, 120.0, 100.0, 0.04, 1.0)
    p_good = Statevector(problem.a_operator).probabilities([problem.objective_qubit])[1]
    assert p_good == pytest.approx(problem.exact_amplitude, abs=1e-10)
    assert (
        0 < barrier_digital_price(GBM, 120.0, 100.0, 0.5) < mixture_digital(GBM, 100.0, 0.04, 1.0)
    )
