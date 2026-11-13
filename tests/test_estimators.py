import numpy as np
import pytest
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms import AmplitudeEstimation, IterativeAmplitudeEstimation

from qae_options.classical import monte_carlo, quasi_monte_carlo
from qae_options.distributions import BimodalLogNormal, discretize
from qae_options.estimators import (
    qae_queries,
    qpe_circuit,
    run_iqae,
    run_mlae,
    run_qae,
    schedule_queries,
)
from qae_options.payoffs import digital_call
from qae_options.reference import mixture_digital
from qae_options.sampling import ExactDistributionSampler, exact_statevector

BIMODAL = BimodalLogNormal(0.4, np.log(85), 0.10, np.log(115), 0.10)


@pytest.fixture(scope="module")
def problem():
    return digital_call(discretize(BIMODAL, 3), 100.0, 0.03, 0.5)


def test_query_counter_closed_forms():
    assert qae_queries(1, 10) == 10 * 3  # one Grover application: A, Q (= 2) -> 3
    assert qae_queries(3, 1) == 2 * 7 + 1
    assert schedule_queries([0, 1, 2], 100) == 100 * (1 + 3 + 5)


def test_estimator_query_counts(problem):
    s = ExactDistributionSampler(50, seed=1)
    assert run_qae(problem, 3, s).oracle_queries == 50 * 15
    mlae = run_mlae(problem, 2, s)
    assert mlae.grover_powers == [0, 1, 2, 4]
    assert mlae.oracle_queries == 50 * (1 + 3 + 5 + 9)
    iqae = run_iqae(problem, 1e-2, s)
    assert iqae.oracle_queries == sum(50 * (2 * k + 1) for k in iqae.grover_powers)


def test_exact_sampler_matches_aer(problem):
    """Matrix-level simulation equals gate-level Aer simulation of the same circuit."""
    ep = problem.estimation_problem()
    for k in (0, 1, 3):
        c = IterativeAmplitudeEstimation(0.01, 0.05).construct_circuit(ep, k, measurement=False)
        mine = exact_statevector(c, {}).probabilities([problem.objective_qubit])
        tc = transpile(c, basis_gates=["u", "cx"])
        tc.save_probabilities([problem.objective_qubit])
        aer = AerSimulator().run(tc).result().data(0)["probabilities"]
        assert np.allclose(mine, aer, atol=1e-10)
        theta = np.arcsin(np.sqrt(problem.exact_amplitude))
        assert mine[1] == pytest.approx(np.sin((2 * k + 1) * theta) ** 2, abs=1e-10)


def test_annotated_qpe_matches_qiskit(problem):
    ep = problem.estimation_problem()
    ref = exact_statevector(AmplitudeEstimation(3).construct_circuit(ep), {}).probabilities(
        range(3)
    )
    mine = exact_statevector(qpe_circuit(ep, 3, False), {}).probabilities(range(3))
    assert np.allclose(ref, mine, atol=1e-12)


def test_sampler_is_seeded_and_unbiased(problem):
    r1 = run_iqae(problem, 1e-3, ExactDistributionSampler(100, seed=7))
    r2 = run_iqae(problem, 1e-3, ExactDistributionSampler(100, seed=7))
    assert r1.estimate == r2.estimate
    assert abs(r1.estimate - problem.exact_price) < 5e-3
    lo, hi = r1.ci
    assert lo <= r1.estimate <= hi


def test_classical_estimators(problem):
    grid = discretize(BIMODAL, 3)
    values = (grid.x >= 100).astype(float)
    discount = np.exp(-0.015)
    est = [monte_carlo(grid, values, 10_000, np.random.default_rng(s), discount) for s in range(20)]
    assert np.mean(est) == pytest.approx(problem.exact_price, abs=3e-3)
    target = mixture_digital(BIMODAL, 100.0, 0.03, 0.5)
    qmc = quasi_monte_carlo(BIMODAL, lambda s: (s >= 100).astype(float), 4096, 1, discount)
    assert qmc == pytest.approx(target, abs=2e-3)
    with pytest.raises(ValueError):
        quasi_monte_carlo(BIMODAL, lambda s: s, 1000, 1, 1.0)


def test_lazy_circuits_match_library_circuits(problem):
    """The annotated-power circuits used by run_iqae / run_mlae give the same distributions as
    the circuits qiskit-algorithms builds."""
    from qiskit_algorithms import MaximumLikelihoodAmplitudeEstimation

    from qae_options.estimators import (
        LazyIterativeAmplitudeEstimation,
        LazyMaximumLikelihoodAmplitudeEstimation,
    )

    ep = problem.estimation_problem()
    obj = [problem.objective_qubit]
    for k in (1, 5):
        ref = IterativeAmplitudeEstimation(0.01, 0.05).construct_circuit(ep, k)
        lazy = LazyIterativeAmplitudeEstimation(0.01, 0.05).construct_circuit(ep, k)
        assert np.allclose(
            exact_statevector(ref, {}).probabilities(obj),
            exact_statevector(lazy, {}).probabilities(obj),
            atol=1e-12,
        )
    refs = MaximumLikelihoodAmplitudeEstimation([0, 1, 4]).construct_circuits(ep)
    lazies = LazyMaximumLikelihoodAmplitudeEstimation([0, 1, 4]).construct_circuits(ep)
    for ref, lazy in zip(refs, lazies, strict=True):
        assert np.allclose(
            exact_statevector(ref, {}).probabilities(obj),
            exact_statevector(lazy, {}).probabilities(obj),
            atol=1e-12,
        )


def test_cache_key_distinguishes_grover_powers(problem):
    from qae_options.estimators import LazyIterativeAmplitudeEstimation
    from qae_options.sampling import circuit_key

    ep = problem.estimation_problem()
    build = LazyIterativeAmplitudeEstimation(0.01, 0.05).construct_circuit
    assert circuit_key(build(ep, 1, True)) != circuit_key(build(ep, 2, True))
    assert circuit_key(build(ep, 2, True)) == circuit_key(build(ep, 2, True))
