# qae-options: Project Specification

> **Option pricing with quantum amplitude estimation: QAE, IQAE and MLAE compared on a
> non-standard payoff and distribution.**
> Original proposal: *Valoración de opciones con estimación de amplitud cuántica* (course line
> 2, option a). Individual project, 8 weeks.

Repo-wide rules (layout, uv usage, the shared `qcc-common` package, conventions, CI) are in the
[root SPEC](../../SPEC.md). This document covers only this project.

---

## 1. Goal and verifiable claim

Classical Monte Carlo (MC) estimates an option's expected payoff with error that falls as
O(1/√N) in the number of samples. Quantum amplitude estimation (QAE) promises error that falls
as O(1/N) in the number of oracle queries. This project **measures** how that promise holds up
in practice. It does not assume the quantum method wins.

**Verifiable claim** (X, Y, Z and W are filled in from measured data):

> "To reach a price error ε = 10⁻³, IQAE uses **X** oracle queries and Monte Carlo uses **Y**
> samples; the QPE-based version uses **Z** more qubits and a transpiled depth **W** times
> larger."

**Result** (see [REPORT.md](REPORT.md) §1): X ≈ 3,700 IQAE queries vs Y ≈ 117,000 MC samples;
Z = 6 more qubits and W ≈ 38× larger depth for QPE-based QAE (m = 6, MLE post-processing).
Scrambled-Sobol QMC needs only ≈ 170 samples, and on the FakeSherbrooke noise model at ≥ 10 % of
its strength the frontier is below k = 1, so no estimator reaches ε.

Non-goals:
- No claim of quantum advantage.
- No results on real hardware beyond one optional short run.
- No learned (qGAN) distribution loaders.

## 2. New component (beyond the tutorials)

Tutorials price a European call on a single lognormal. This project adds:

1. **A non-standard distribution**: a mixture of two lognormals, which is bimodal. A
   motivating scenario is a binary event such as a regulatory decision or earnings
   announcement before expiry.
2. **A non-standard payoff**: a digital (cash-or-nothing) call as the primary payoff, and a
   discretely monitored barrier option as a stretch goal.
3. **Three estimator variants compared on equal terms**:
   - canonical QAE with QPE;
   - iterative QAE (IQAE);
   - maximum-likelihood QAE (MLAE), which does not use QPE.

## 3. Model

### 3.1 Distributions

Each distribution is discretized on a grid of 2ⁿ points over a truncated interval
[S_low, S_high]: the 0.1 % and 99.9 % quantiles, configurable. The register size is
n = 3 to 7 qubits.

| Name | Parameters | Use |
|---|---|---|
| `lognormal` | S₀, r, σ, T (GBM terminal distribution) | Validation against Black-Scholes |
| `bimodal` | w; (μ₁, σ₁); (μ₂, σ₂) for the log-price | New component |

Loading: the probabilities pᵢ are encoded as amplitudes √pᵢ with a generic state-preparation
circuit (`qiskit.circuit.library.StatePreparation`). Both distributions use the **same** loader,
so loading cost is compared like for like.

### 3.2 Payoffs

The A operator prepares √(1−a)|ψ₀⟩|0⟩ + √a|ψ₁⟩|1⟩. The price is a function of a.

| Payoff | Definition | Circuit | Approximation error |
|---|---|---|---|
| European call (validation) | max(S_T − K, 0) | Piecewise-linear amplitude function (`LinearAmplitudeFunction`) with rescaling factor c | Yes: depends on c, so it is reported separately |
| **Digital call (primary)** | Q · 1{S_T ≥ K} | `IntegerComparator` flips the objective qubit | **None**: a = P(S_T ≥ K) exactly on the grid |
| Barrier (stretch) | Up-and-out digital: Q · 1{S_t₁ < B ∧ S_t₂ ≥ K}, 2 monitoring dates | Joint loader over 2 registers (2n qubits) + 2 comparators + AND | None on the grid; qubit count roughly doubles |

The price is e^{−rT} · scale · a, where `scale` is Q for the digital payoff and the rescaling
map for the call.

The digital payoff makes the main study clean: on the grid it has no payoff-approximation
error, so every measured error is **estimation** error.

### 3.3 Classical references

| Reference | How |
|---|---|
| **Discretized exact** (error baseline) | `Σᵢ pᵢ f(xᵢ)` on the same grid |
| Continuous analytic | Black-Scholes for the lognormal call. Mixtures: the call is the weight-averaged Black-Scholes price; the digital is `Σⱼ wⱼ e^{−rT} Q Φ(d₂ⱼ)` |
| Numerical check | `scipy.integrate.quad` on the continuous density |

## 4. Estimators

All three quantum estimators come from `qiskit-algorithms` (0.4: V2 primitives) and use a
seeded shot-based SamplerV2.

**Implementation notes (recorded during the project).**
- Noiseless runs use `sampling.ExactDistributionSampler`. It computes each circuit's exact
  output distribution once, from cached unitaries of its sub-instructions, and draws the shots
  with a seeded multinomial. This has the same statistics as running the circuit; it is tested
  against Aer gate-level simulation. Gate-level simulation of Q^k A is infeasible at the k IQAE
  needs (~10⁷ gates at k ≈ 1,500). Noisy runs (E4) use Aer gate-level simulation.
- Grover powers (IQAE, MLAE) and controlled powers (canonical QAE) are annotated operations on
  one Q gate. The circuits are otherwise identical to the library's, with identical output
  distributions (tested). Qiskit's eager controlled synthesis took over 30 min at m = 8.
- Canonical QAE is reported with both the library's grid estimate and its MLE
  post-processing.
- The circuit cost of deep Grover powers is composed from transpiled components (A, Q,
  controlled Q, inverse QFT) and checked against direct transpilation at small k.

| Estimator | Class | Accuracy knob | Uses QPE |
|---|---|---|---|
| Canonical QAE | `AmplitudeEstimation(num_eval_qubits=m)` | m = 1 to 8 | Yes (m extra qubits + inverse QFT) |
| IQAE | `IterativeAmplitudeEstimation(epsilon_target, alpha)` | ε_target from 10⁻¹ to 10⁻⁴, α = 0.05 | No |
| MLAE | `MaximumLikelihoodAmplitudeEstimation(evaluation_schedule)` | Schedule {0, 1, 2, 4, …, 2^M}, M = 0 to 7 | No |
| Classical MC | Our own code: sample N indices from {pᵢ}, average f | N = 10² to 10⁷ | n/a |
| Quasi-MC (honesty check, optional) | Sobol points via `scipy.stats.qmc` on the continuous distribution | N | n/a |

QMC is included because a "good classical method" can beat 1/√N on smooth problems. Leaving it
out would bias the comparison.

## 5. Definitions

These definitions are fixed before any data is collected.

- **Estimation error:** |â − a_exact| converted to price units. Here a_exact is computed from
  the discretized exact reference. This is the error behind ε = 10⁻³ in the claim.
- **Discretization error:** the gap between the discretized exact price and the continuous
  analytic price. It is reported separately for each n and never mixed with estimation error.
- **Payoff-approximation error:** the call only. It is reported for each value of c.
- **Oracle queries.** One query is one application of A or of A†. The Grover operator
  Q = A S₀ A† S_χ costs 2 queries, so a circuit with k Grover applications costs 2k + 1.
  - Canonical QAE: one circuit has k = 2^m − 1 controlled Grover applications. The total is
    shots · (2(2^m − 1) + 1).
  - IQAE and MLAE: N_oracle = Σ over circuits c of shots_c · (2k_c + 1).
  - Classical MC: one sample counts as one query.
- **Circuit cost:** `qcc_common.metrics.circuit_metrics` reports qubits, transpiled depth and
  2-qubit gate count. Settings: basis {rz, sx, x, ecr} (or cx), `optimization_level=2`, a fixed
  seed and a linear coupling map. All of these are recorded in each result.
- **Error statistics:** each point is run R = 50 times with seeds `seed + i`. The report gives
  the median absolute error with the 16th and 84th percentile band. Slopes are least-squares
  fits on log-log axes.

## 6. Code layout, dependencies, commands

```text
projects/qae-options/
├── pyproject.toml
├── src/qae_options/
│   ├── distributions.py   # LogNormal, BimodalLogNormal; discretize(n, bounds) -> grid, probs; loader circuit
│   ├── payoffs.py         # digital_call(), european_call(c), barrier_digital(); returns A, objective qubit, post-processing
│   ├── estimators.py      # run_qae / run_iqae / run_mlae -> EstimateResult(estimate, ci, oracle_queries, circuits)
│   ├── sampling.py        # ExactDistributionSampler (seeded SamplerV2; exact noiseless, Aer noisy)
│   ├── resources.py       # circuit cost of A, Q^k A and canonical QAE from transpiled components
│   ├── classical.py       # monte_carlo(), quasi_monte_carlo()
│   ├── reference.py       # black_scholes_call(), mixture_digital(), discretized_exact()
│   ├── experiments.py     # E1-E6 runners driven by config
│   └── cli.py             # `qae-options <experiment> --config ...`
├── configs/               # e1_validate.toml, e2_convergence.toml, e3_resources.toml, e4_noise.toml, e5_loading.toml
├── run_all.sh             # regenerates every result in order (resumable)
├── REPORT.md              # final report
├── notebooks/             # exploration and figure assembly only
├── tests/
└── results/{raw,summary,figures}/
```

Dependencies:
- Core: `qcc-common` (workspace), `qiskit`, `qiskit-aer`, `qiskit-algorithms`, `numpy`,
  `scipy`, `pandas`, `matplotlib`.
- Hardware access (E6) uses `qiskit-ibm-runtime`, which comes in through `qcc-common`
  (root SPEC §3.4).
- `qiskit-finance` is **not** a dependency: the circuits are built here (see §10).

```bash
uv sync --package qae-options
uv run --package qae-options qae-options e2 --config projects/qae-options/configs/e2_convergence.toml
uv run --package qae-options pytest projects/qae-options -m "not slow"
```

Required fast tests:
- Discretized probabilities sum to 1, and the loader statevector reproduces √pᵢ.
- On the grid, the digital-payoff amplitude equals P(S_T ≥ K) (exact statevector check).
- `discretized_exact` agrees with Black-Scholes as n grows, for the lognormal call.
- The mixture-digital closed form matches `quad`.
- The oracle-query counter gives the expected totals for small m and k.

## 7. Experiments

| ID | Question | Setup | Output |
|---|---|---|---|
| **E1** | Does the pipeline give correct prices? | Lognormal, European call, n = 3 to 5, all three estimators at high precision | Table comparing each estimate with the discretized-exact and Black-Scholes prices; discretization and c-approximation errors |
| **E2** | How does error scale with queries? | Bimodal + digital, n fixed (default 5); sweep the knob of each estimator plus MC (and QMC); R = 50 | Log-log plot of error vs oracle queries with error bands and fitted slopes (expect ≈ −1 for the QAE variants, ≈ −½ for MC); fills in X and Y |
| **E3** | What does each variant cost in hardware resources? | Same A operator; QAE at the m needed for ε = 10⁻³, IQAE and MLAE at their largest k | Table of qubits, depth and 2q gates per variant; fills in Z and W |
| **E4** | At what depth does noise remove the convergence advantage? | E2 repeated on `noisy_simulator` (fake IBM backend) with noise scale ∈ {0.1, 0.3, 1.0}. **As run:** n = 3 (gate-level noisy simulation), MLAE and IQAE, R = 20, with scales 0.01 and 0.03 added to locate the frontier | Error vs queries under noise; the largest useful Grover power k* and its depth, i.e. the "frontier" |
| **E5** | How much does loading the distribution cost? | Loader depth and 2q gates vs n = 3 to 7, lognormal vs bimodal; share of the total A-operator depth | Plot and discussion of loading as a possible bottleneck that can cancel the quadratic speedup |
| E6 (optional) | Does it run on real hardware? | IBM backend, digital payoff, n = 3, IQAE with small k, a few shots. **Not run:** no IBM Quantum access | Estimate compared with the simulator and the noisy simulator |
| Stretch | Barrier payoff | Barrier digital at n = 3 per date (6 + ancilla qubits) | E1 and E3 rows for the barrier |

## 8. Milestones

| Week | Work | Done when |
|---|---|---|
| 1 | Review Grover, QPE and amplitude amplification. `distributions.py` (lognormal) and the loader. `reference.py` with Black-Scholes | Loader tests pass; the discretized-exact price converges to Black-Scholes |
| 2 | European call A operator and a basic canonical QAE run (statevector) | E1 for canonical QAE within the expected error |
| 3 | `estimators.py`: QAE, IQAE and MLAE with one interface; the oracle-query counter | All three estimators reproduce E1; query-counter tests pass |
| 4 | Validation against Black-Scholes; `classical.py` (MC, QMC) | E1 table complete |
| 5 | Bimodal distribution, digital payoff, mixture closed form | Digital tests pass; reference prices fixed in the E2 config |
| 6 | E2 sweeps with R repetitions; E3 resource table | E2 figure and fitted slopes; claim fields X, Y, Z, W filled in |
| 7 | E4 noise sweeps; E5 loading analysis; optional E6 on hardware | E4 frontier identified; E5 figure |
| 8 | Final figures, report, conclusions | All figures in `results/figures/`; claim stated with numbers; README updated |

Checklist:
- [x] E1 validation
- [x] E2 convergence
- [x] E3 resources
- [x] E4 noise frontier
- [x] E5 loading cost
- [ ] E6 hardware (optional): not run, no IBM Quantum access
- [x] Barrier (stretch): E1 and E3 rows, n = 3 per date
- [x] Report

## 9. Acceptance criteria

- The verifiable claim (§1) is stated with measured values of X, Y, Z and W, and each value
  points to its config and result file.
- E2 fitted slopes are reported with uncertainty. A slope that differs from −1 or −½ is
  explained, not hidden.
- Discretization, payoff-approximation and estimation errors are reported separately.
- E4 names a concrete depth or Grover power where the advantage disappears under the chosen
  noise model, or states that it was not reached within the range tested.
- The report has an honest discussion of the loading cost (E5).
- Every figure can be regenerated from the CLI with a committed config.

## 10. Risks and fallback

| Risk | Mitigation |
|---|---|
| Canonical QAE circuits too deep to simulate or transpile at large m or n | Cut n to 3–5 qubits and cap m; focus the study on comparing QPE with the iterative variant (as in the proposal) |
| The barrier payoff needs too many qubits | Keep it as a stretch goal; the digital payoff alone satisfies the "new payoff" requirement |
| qiskit-algorithms API or primitive version mismatch with the pinned Qiskit | Check in week 1; isolate behind `estimators.py` so the rest of the code doesn't change |
| QMC beats quantum in query count | That is a valid result; report it |
| `qiskit-finance` unmaintained or incompatible | Not used. The loaders and payoff circuits are built here from `qiskit.circuit.library`; only the uncertainty-model *pattern* is borrowed |
| Noise destroys convergence even at k = 1 | Lower the noise scale to find where convergence starts; "the frontier is below k = 1 at realistic noise" is itself a measurable result |
