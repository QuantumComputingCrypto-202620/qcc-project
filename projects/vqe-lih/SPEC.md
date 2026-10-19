# vqe-lih: Project Specification

> **LiH dissociation curve with VQE: comparing ansätze and the distance to chemical accuracy.**
> Original proposal: *Curva de disociación de LiH con VQE* (course line 2, option c).
> Individual project, 8 weeks.

Repo-wide rules (layout, uv usage, the shared `qcc-common` package, conventions, CI) are in the
[root SPEC](../../SPEC.md). This document covers only this project.

---

## 1. Goal and verifiable claim

The project uses VQE to estimate the ground-state energy of LiH at many interatomic distances,
which traces the **full dissociation curve**. It compares two ansätze against exact
diagonalization. The target is **chemical accuracy**: an error below 1.6 mHa, which is about
1 kcal/mol.

**Verifiable claim** (the X, Y and Z values are filled in from measured data):

> "The UCCSD ansatz reaches chemical accuracy (error < 1.6 mHa) at **X** of **Y** points on the
> curve with depth **Z**; the hardware-efficient ansatz reaches it at **X′** points with depth
> **Z′** (lower), but fails near dissociation where the state becomes more strongly correlated."

The second half of the claim is a **hypothesis**. If the data contradicts it, the claim is
restated to match the data.

**Result** (filled in from data; see [REPORT.md](REPORT.md) §1): in the default (2e, 3o) space,
X = 20 of Y = 20 at depth Z = 953; X′ = 16 (HEA reps 3) at depth Z′ = 45. In (4e, 4o):
20/20 at depth 5177 vs 16 at depth 61. X and X′ count points that are chemically accurate
**and** in the correct symmetry sector (|⟨N⟩ − N| and ⟨S²⟩ below 0.01). The hypothesis is
confirmed: the best HEA is accurate at every point up to 2.0 Å and at 20 % of points from 2.5 Å.

Non-goals:
- No new ansatz design.
- No optimization on real hardware. Hardware is used only to evaluate energies at points
  already optimized on the simulator.

## 2. New component (beyond the tutorials)

Tutorials almost always use H₂ at a single geometry. This project goes further:

- It uses **LiH** along the **whole curve**, not only near equilibrium.
- It compares **two ansätze on equal terms**:
  - **UCCSD**: motivated by chemistry; deeper circuits.
  - **Hardware-efficient (HEA)**: shallower circuits with no explicit chemical structure.
- H₂ is used **only** to validate the pipeline.
- **BeH₂** is a stretch goal.

## 3. Model: Hamiltonian pipeline

Everything is built with qiskit-nature, using PySCF as the electronic-structure driver.

| Step | Choice (default) | Notes |
|---|---|---|
| Geometry | Li at the origin, H at (0, 0, R) | R in Å |
| Basis | STO-3G | 6 spatial orbitals, 4 electrons |
| Driver | `PySCFDriver` (RHF) | Also records E_HF |
| Reduction | `FreezeCoreTransformer` (freezes Li 1s), then `ActiveSpaceTransformer` | Configurable. Default active space: **(2e, 3o)**, which drops the π orbitals (p_x, p_y) because they do not couple to σ by symmetry |
| Mapping | **Jordan-Wigner** (as in the proposal) | (2e, 3o) gives 6 qubits. Optional comparison: `ParityMapper` with 2-qubit reduction gives 4 qubits |
| Larger setting (stretch) | Frozen core only: (2e, 5o), 10 qubits; or no frozen core: (4e, 6o), 12 qubits | Gives UCCSD a harder test (see note) |

**Design note.** With only 2 active electrons, CCSD is exact and Trotterized UCCSD is expected
to be very close to exact. In that setting the UCCSD side of the claim is almost trivially
satisfied, and the comparison is driven by the HEA. Week 4 decides whether the main study also
includes a 4-electron active space (for example (4e, 6o) without freezing the core) so that
UCCSD faces a non-trivial test. The decision is recorded in this spec.

**Week-4 decision (recorded).** The main study uses two active spaces: the default (2e, 3o) and
a 4-electron **(4e, 4o)** space with no frozen core and only the four σ orbitals (8 qubits, 26
UCCSD parameters). (4e, 6o) was measured at about 0.5 s per energy evaluation with 92 UCCSD
parameters, roughly 7 hours for the curve, and was dropped. σ orbitals are selected by symmetry:
orbitals with no p_x/p_y weight, taking the highest occupied and lowest virtual ones.

**Bond lengths.** About 20 points from 0.8 Å to 4.0 Å, sampled more densely around the
equilibrium bond length (about 1.6 Å). The exact list goes in `configs/curve.toml`.

## 4. Ansätze

| Ansatz | Construction | Initial state | Knobs |
|---|---|---|---|
| UCCSD | `qiskit_nature...circuit.library.UCCSD(num_spatial_orbitals, num_particles, mapper)` | `HartreeFock` | One Trotter step (`reps=1`); optional `reps=2` |
| HEA | `efficient_su2` / `n_local` with RY (and RZ) rotations and CX entanglers, linear entanglement | HF bitstring prepended (default) or \|0…0⟩ | `reps` ∈ {1, 2, 3, 4} |

For each ansatz and active space, the following are recorded:
- number of variational parameters;
- logical depth;
- transpiled depth and 2-qubit gate count, from `qcc_common.metrics.circuit_metrics` with the
  same settings as the rest of the repo.

The HEA does **not** conserve particle number or spin. For every VQE result the run also
evaluates ⟨N⟩ and ⟨S²⟩, so a result in the wrong symmetry sector can be detected and reported.
No penalty term is used by default; a penalty term is an optional ablation.

## 5. References and error definitions

| Quantity | Definition |
|---|---|
| **E_exact** (error baseline) | Exact diagonalization of the **same qubit Hamiltonian** (`NumPyMinimumEigensolver`), restricted to the correct particle-number sector. This isolates the error from the ansatz and optimizer. |
| E_FCI (full) | PySCF FCI in the full STO-3G space. Context only: E_exact − E_FCI is the **active-space error**, reported separately. |
| E_HF | Hartree-Fock energy. It shows how much correlation energy is recovered. |
| **VQE error** | ΔE = E_VQE − E_exact, in Hartree. A point "reaches chemical accuracy" when ΔE < 1.6 × 10⁻³ Ha. |
| Correlation energy recovered | (E_HF − E_VQE) / (E_HF − E_exact) |

All energies are total electronic energies that include nuclear repulsion and the frozen-core
energy shift, in Hartree.

## 6. VQE protocol

- Implementation: `qiskit_algorithms.VQE(estimator, ansatz, optimizer, initial_point, callback)`.
- Estimator:
  - Exact statevector Estimator for E2–E5.
  - Shot-based Aer Estimator for E6.
- Optimizers:
  - `COBYLA`, `L_BFGS_B` (or `SLSQP`) and `SPSA` from `qiskit_algorithms.optimizers`.
  - Default `maxiter` is 1000 for gradient-free optimizers and 300 for SPSA.
  - E2/E3 use L-BFGS-B. E5/E6 add `spsa_small`: SPSA with the standard calibration but a
    0.05 rad target step instead of 2π/10, because the default steps are an order of magnitude
    larger than UCCSD amplitudes (see REPORT §5).
- HEA initial point: with all angles zero the HEA is *not* the HF state, since the CX layers
  permute the HF bitstring. E3 therefore starts the equilibrium point from the best of 5
  random N(0, 0.1²) points, then warm-starts.
- Shot-based estimation (E6) uses `measurement.ShotEstimator`, which samples every
  qubit-wise-commuting Pauli group with `shots` shots per evaluation. Aer's `EstimatorV2` with
  `precision > 0` only adds the same Gaussian draw to the exact value on every call, so it does
  not model shot noise.
- Initialization strategies, compared in E5:
  1. **zeros**: equals HF for UCCSD.
  2. **random**: N(0, 0.1²), with K = 10 seeds.
  3. **warm start**: begin at equilibrium and sweep outwards in both directions, using the
     optimal parameters from the neighbouring point.
- Recorded for every run:
  - final energy and ΔE;
  - number of function evaluations and iterations to convergence;
  - energy history from the callback;
  - ⟨N⟩ and ⟨S²⟩;
  - wall-clock time;
  - seed.
- Convergence definition: the optimizer's own termination is reported as-is. "Iterations to
  1.6 mHa" is the first iteration at which ΔE drops below the threshold.

## 7. Code layout, dependencies, commands

```text
projects/vqe-lih/
├── pyproject.toml
├── src/vqe_lih/
│   ├── molecules.py      # geometries for H2, LiH, BeH2; bond-length grids
│   ├── hamiltonian.py    # driver -> transformers -> mapper -> (qubit_op, problem, metadata)
│   ├── ansatze.py        # build_uccsd(), build_hea(reps), with HF initial state
│   ├── solver.py         # run_vqe(): optimizer + init strategy -> VQERecord
│   ├── reference.py      # exact_energy(), fci_energy(), hf_energy()
│   ├── measurement.py    # ShotEstimator: shot-based EstimatorV2 over QWC Pauli groups
│   ├── mitigation.py     # readout-confusion inversion and two-qubit-gate-folding ZNE (E7)
│   ├── experiments.py    # E1-E7 runners driven by config
│   └── cli.py            # `vqe-lih <experiment> --config ...`
├── configs/              # e1_h2.toml, curve.toml, e2_uccsd.toml, e3_hea.toml, e4_tradeoff.toml, e5_optimizers.toml, e6_shots.toml, e7_noise.toml, beh2.toml
├── run_all.sh            # regenerates every result in order (resumable)
├── REPORT.md             # final report
├── notebooks/
├── tests/
└── results/{raw,summary,figures}/
```

Dependencies:
- Core: `qcc-common` (workspace), `qiskit`, `qiskit-aer`, `qiskit-algorithms`,
  `qiskit-nature`, `pyscf`, `numpy`, `scipy`, `pandas`, `matplotlib`.
- `qiskit-ibm-runtime` comes in through `qcc-common` (root SPEC §3.4). It provides EstimatorV2
  with built-in resilience options: TREX readout mitigation and ZNE.

```bash
uv sync --package vqe-lih
uv run --package vqe-lih vqe-lih e2 --config projects/vqe-lih/configs/e2_uccsd.toml
uv run --package vqe-lih pytest projects/vqe-lih -m "not slow"
```

Required fast tests:
- At equilibrium, the H₂ (STO-3G, JW, 4 qubits) qubit Hamiltonian has an exact energy within
  10⁻⁶ Ha of PySCF FCI.
- VQE with UCCSD on H₂ reaches ΔE < 1.6 mHa.
- LiH (2e, 3o) gives the expected qubit count (6 with JW) and particle number.
- UCCSD with zero parameters reproduces E_HF.
- `circuit_metrics` returns the expected parameter counts for both ansätze.

## 8. Experiments

| ID | Question | Setup | Output |
|---|---|---|---|
| **E1** | Is the pipeline correct? | H₂, STO-3G, JW; curve from 0.3 to 2.5 Å; UCCSD + COBYLA | H₂ curve: VQE vs exact vs FCI, all within chemical accuracy |
| **E2** | How accurate is UCCSD along the LiH curve? | LiH, default active space, all bond lengths, best optimizer, warm start | Dissociation curve; ΔE vs R; fills in X/Y and Z |
| **E3** | How accurate is the HEA along the LiH curve? | Same as E2 with HEA at reps 1 to 4 | Curves and ΔE vs R per value of reps; fills in X′ and Z′ |
| **E4** | What is the trade-off between depth and accuracy? | E2 + E3 results combined | Error vs depth and error vs distance plots with a 1.6 mHa line; parameter count table |
| **E5** | How do optimizers and initialization behave? | COBYLA vs L-BFGS-B vs SPSA × {zeros, random ×10, warm start}, at 3 representative points (equilibrium, ~2.5 Å, ~3.5 Å) | Iterations to convergence; spread of final ΔE across seeds; any local minima or plateaus documented |
| **E6** | How does shot noise affect the result? | Shot-based Estimator with shots ∈ {10², 10³, 10⁴, 10⁵}; SPSA vs COBYLA | ΔE vs shots; shots needed to reach chemical accuracy |
| E7 (optional) | How far is real hardware from the simulator, and how much does mitigation help? | 2–3 points; energies evaluated at simulator-optimized parameters on IBM hardware; raw vs readout mitigation vs ZNE. **Run as a substitute on the FakeSherbrooke noisy simulator** (no hardware access); hardware columns not filled | Table: simulator, noisy simulator, hardware raw, hardware mitigated |
| Stretch | Is BeH₂ feasible? | BeH₂ with a reduced active space; UCCSD vs HEA at a few points | Same metrics as E2–E4 where affordable |

## 9. Milestones

| Week | Work | Done when |
|---|---|---|
| 1 | `hamiltonian.py` (driver, transformers, JW mapper); `reference.py` | H₂ exact energy matches FCI; tests pass |
| 2 | `ansatze.py` and `solver.py`; E1 on H₂ | H₂ curve within chemical accuracy |
| 3 | LiH active space and qubit counts; UCCSD at equilibrium | LiH equilibrium ΔE < 1.6 mHa |
| 4 | E2: the full LiH curve with UCCSD; decide whether to add a 4-electron active space (§3) | E2 figure; decision recorded in this spec |
| 5 | HEA; E3 sweep; E4 comparison | E3 and E4 figures; claim fields filled in |
| 6 | E5 optimizers and initialization; E6 shot noise | E5 and E6 figures and tables |
| 7 | Optional E7 on IBM hardware with and without mitigation; or BeH₂ stretch | E7 table, or BeH₂ results / documented infeasibility |
| 8 | Final figures (curve, error vs depth, error vs distance), report, conclusions | All figures in `results/figures/`; claim stated with numbers; README updated |

Checklist:
- [x] E1 H₂ validation
- [x] E2 UCCSD curve
- [x] E3 HEA curve
- [x] E4 depth vs accuracy
- [x] E5 optimizers and initialization
- [x] E6 shot noise
- [x] E7 hardware (optional): noisy-simulator substitute only, no real hardware
- [x] BeH₂ (stretch): (4e, 4o) σ space, 5 geometries
- [x] Report

## 10. Acceptance criteria

- The verifiable claim (§1) is stated with measured X, Y, Z, X′ and Z′, and each value points
  to its config and result file. The hypothesis "HEA fails near dissociation" is confirmed or
  refuted explicitly.
- The ansatz/optimizer error (vs E_exact) and the active-space error (vs full FCI) are reported
  separately.
- Every HEA result carries its ⟨N⟩ and ⟨S²⟩, and results in the wrong sector are flagged.
- If the optimizer fails to converge, or local minima or barren plateaus appear, they are
  documented with evidence (energy histories, spread across seeds). They are not dropped.
- Every figure can be regenerated from the CLI with a committed config.
- The report includes the section "Why VQE rather than QPE" (§11).

## 11. Why VQE rather than QPE

This answers the proposal's background question, and the final report expands on it.

- **QPE** gives the ground-state energy to precision ε using coherent evolution of depth
  O(1/ε), with controlled-U^{2^k} applications. It is the long-term solution: provably
  accurate, but it needs deep circuits and, in practice, fault-tolerant error correction.
- **VQE** keeps circuits shallow: each energy estimate is one ansatz circuit plus measurements.
  It moves the hard work to a classical optimizer and pays with many more circuit executions
  (O(1/ε²) shots per estimate). This trade fits today's noisy hardware.
- What the project measures adds evidence:
  - E4 gives depth vs accuracy;
  - E6 gives the shot cost;
  - E7 shows the effect of noise and mitigation.
  Together they show where VQE's limits are: expressivity, optimization, measurement cost.

## 12. Risks and fallback

| Risk | Mitigation |
|---|---|
| BeH₂ too costly in qubits or time | It stays a stretch goal; LiH with two ansätze across the full curve is already enough for the project |
| UCCSD trivially exact in (2e, 3o) | Add a 4-electron active space in week 4 (§3), or state the triviality clearly and let the HEA comparison carry the claim |
| Optimizer fails at some points (local minima, barren plateaus) | Use multiple seeds and warm starts; document failures as results (E5), as stated in the proposal |
| HEA converges to the wrong particle-number sector | Detect it with ⟨N⟩ and ⟨S²⟩; optional penalty-term ablation |
| qiskit-nature is behind Qiskit core | Version pin set in week 0 (root SPEC §3.5); keep qiskit-nature use inside `hamiltonian.py` and `ansatze.py` |
| PySCF not available on Windows | Use WSL2 / Linux / macOS (root SPEC §8) |
| No IBM hardware access or long queues | E7 is optional; use the noisy simulator from `qcc_common.backends` as a substitute |
