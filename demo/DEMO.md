# Live demo: run sheet

`uv run qae-demo` walks through the project in 7 steps. It pauses after each one until you
press Enter. All noiseless numbers are computed live with this repository's code, which takes
about 10 s of compute in total on a laptop. The full-study numbers (E2–E4 take ~1.5 h to
regenerate) are read from the committed `results/summary/` files. Figures go to `demo/out/`
and open in the system image viewer.

## Before the talk (the day before, then again 10 minutes before)

```bash
cd qcc-project
git pull && uv sync                       # needs the network only the first time
uv run qae-demo --no-pause --no-open      # dry run: should finish in ~10-15 s with no errors
uv run pytest -m "not slow" -q            # optional: 40 passed
```

- Terminal: font size 18–20 pt, window at least **100 columns** wide (the tables fit 80).
  Use a light theme if the projector is washed out.
- Close the image viewer windows from the dry run, so each figure opens fresh during the talk.
- Turn off notifications. Keep this file open on a second screen or printed.
- **Offline fallback:** after one `uv sync`, the demo needs no network. If the laptop cannot
  run it at all, the committed figures in `results/figures/` and the slides cover every step.

## Where the demo sits in the talk

The slide deck has the demo in two parts:

| Slide | Command | Steps |
|---|---|---|
| 9 · “Price it live” | `uv run qae-demo --last 4` | 1–4: problem, A operator, Grover, IQAE vs Monte Carlo (~6 min) |
| 14 · E4 noise (optional) | `uv run qae-demo --only 6` | 6: the same circuits under IBM noise (~1 min) |

Steps 5 (full study) and 7 (takeaways) repeat what the slides show. They are there so that
`uv run qae-demo` also works as a standalone walkthrough, e.g. if the projector fails.

## Script, step by step (~8 minutes for all 7)

| # | Say | Output to point at | Time |
|---|---|---|---|
| — | `uv run qae-demo` | Title rule | — |
| 1 | "A binary event before expiry: 40 % bad news near 85, 60 % good news near 115. A digital call pays 1 above 100." | Continuous price **0.5636**; grid price 0.5572 at n = 5; the figure shows the two modes and the payoff region in orange. | 1 min |
| 2 | "The price becomes the probability of measuring 1 on one qubit. A = loader + comparator; one use of A is one *oracle query*." | The circuit (`P(S)` then `i ≥ K`); the statevector probability equals the grid tail to 10 digits; **2,114 two-qubit gates per query** on hardware. | 1 min |
| 3 | "Grover's operator turns the amplitude into a rotation angle. After k steps the probability is sin²((2k+1)θ). Reading a frequency needs ~1/ε queries, averaging coin flips needs ~1/ε²." | Circuit column = formula column exactly; 100-shot samples scatter around it. The figure shows the oscillation. | 1.5 min |
| 4 | "One IQAE run, round by round: it picks bigger Grover powers as the interval shrinks." Then: "Now a race: 20 seeds, and Monte Carlo gets exactly the same budget." | Rounds k = 0, 1, 1, 11, 51, 262; final error ~10⁻⁵. In the race table, the **MC / IQAE** column grows as the target error shrinks (≈ 2×, 18×, 250×, 1,100×). That growth is the quadratic speedup. Figure: blue slope −1, orange slope −½. | 2 min |
| 5 | "That was a live sample. The full study is 49 settings × 50 seeds." | Slopes −1.01 (IQAE) vs −0.52 (MC); **3,659 vs 117,102** queries at ε = 10⁻³; then the caveat: **quasi-MC needs 172**. | 1 min |
| 6 | "Now the same circuits with IBM's noise model." | At scale 1 the deviation jumps from 0.04 (k = 0) to 0.3–0.5 for k ≥ 1: P(1) collapses to ~0.5. At 3 % of today's noise it survives to k ≈ 4. | 1 min |
| 7 | "Five takeaways." | Read the panel; switch back to the slides for conclusions and Q&A. | 0.5 min |

The live numbers in step 4 and the noisy probabilities in step 6 depend on the seed. They are
reproducible with the default `--seed 2026`, so the numbers above are what you will see.

## If something goes wrong

| Problem | Fix |
|---|---|
| A step errors or hangs | Ctrl-C skips the step; or restart at the next one with `uv run qae-demo --step N` |
| Need to repeat one step (a question from the audience) | `uv run qae-demo --only 3` |
| Running late | `uv run qae-demo --step 4 --quick` (race + study + noise + takeaways, ~5 s) |
| Figures don't open | They are in `demo/out/` and `results/figures/`; open them by hand, or rerun with `--no-open` and use the slides |
| `uv` not found on the presentation machine | Install it (README Quickstart), then `uv sync`. It needs the network, so do this beforehand |

## Likely questions

- **"So is there a quantum advantage?"** Only in oracle-query count against plain Monte
  Carlo, and only noiselessly. QMC wins on this 1-D problem, and noise removes it entirely.
- **"Why a digital payoff?"** It has no payoff-approximation error, so every measured error is
  estimation error. The European call's rescaling factor c trades approximation error for
  amplified estimation error (E1).
- **"Why is the simulation so fast?"** The sampler computes each circuit's exact output
  distribution once, from cached unitaries (Q^k is a matrix power), and then draws seeded
  multinomial shots. Tests check it against Aer gate-level simulation to 10⁻¹⁰.
- **"What would it take on real hardware?"** IQAE at ε = 10⁻³ needs ~55,000 two-qubit gates
  on a line. At realistic noise only ~344 (k = 1) are usable (E3, E4).
- **"Is the loader the bottleneck?"** Not here: the comparator is 98 % of A's two-qubit
  gates at n = 5. The loader still grows as 2ⁿ (E5).
