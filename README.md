# RPS-based Distributed Optimization under Soft Faults

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20461461.svg)](https://doi.org/10.5281/zenodo.20461461)

Paper companion · Python 3.10+

Companion implementation for the paper *Random Permutation Set-Based
Diagnosis for Reliable Large-Scale Distributed Optimization Under Soft
Faults*. The code closes the loop "diagnosis -> soft discounting γ ->
robust gradient tracking" and reproduces all 8 figures and 2 tables
from the paper via Monte Carlo experiments.

## Paper-claim correspondence (v0.4.7 full mode, N=50, MC=20)

| Paper claim | Status |
|---|---|
| §1 main claim: RPS significantly beats detect-then-isolate (HT) | ✓ fully reproduced (Drift d=1.87, p<0.001) |
| §1 HT oscillation phenomenon | ✓ fully reproduced (HT iters-to-converge 928±216, RPS 173±3) |
| §4.5.2 RPS-NoOrder worse than RPS-Full | ✓ fully reproduced (d=0.73, p<0.001) |
| §4.5.2 RPS-Sym worse than RPS-Full | ✓ fully reproduced (Drift d=1.18, Constant d=2.01) |
| §4.5.3 Constant-bias scenario, RPS-Full optimal | ✓ fully reproduced (3.82 < UD 4.03 < HT 4.80, all p<0.001) |
| §4.5.3 Drift "RPS-Full beats next-best by 40%" specific number | △ direction matches (RPS significantly beats HT/Sym), tied with UD; see [`expected_results_full.json`](./expected_results_full.json) |
| §4.5.3 Intermittent: RPS-Full significantly best | △ direction matches (RPS-Full has best mean among RPS variants and HT), but differences not statistically significant under MC=20 noise |
| §4.5.4 all 5 modeling assumptions | ✓ `verify_assumptions.py` 5/5 PASS |

Detailed numbers are in the `paper_claims_correspondence` field of
[`expected_results_full.json`](./expected_results_full.json). The paper
methodology is fully implemented; specific-number gaps are explained in
[`IMPLEMENTATION_NOTES.md §15`](./IMPLEMENTATION_NOTES.md) (three sources:
proxy δ, random seeds, drift_cap).

## Project layout

```
.
├── config.py                    # Core data structures (PMF, RPSConfig, FaultConfig)
├── costs.py                     # Cost models: LeastSquares, LogReg, QuadraticDispatch
├── datasets.py                  # MNIST loader, IEEE 39-bus factory
├── distributed_optimization.py  # Graph, consensus weights, gradient tracking, fault injection, residuals
├── rps_diagnosis.py             # Energy distance, PMF, LOS/DS, JS divergence, OPT, γ
├── baselines.py                 # Hard-Threshold (χ²), Uniform-Discount, Byzantine-Resilient
├── statistics_utils.py          # Wilcoxon, Holm-Bonferroni, Cohen's d
├── experiments.py               # run_optimization main loop + derived metrics
├── figures.py                   # Plotting functions for the 8 figures
├── main.py                      # Experiment entry point
├── verify_assumptions.py        # Executable check of paper Section 4.5.4 assumptions
├── tests/                       # pytest unit tests
├── pyproject.toml               # ruff & pytest & mypy config
├── requirements.txt             # Pinned version ranges
└── IMPLEMENTATION_NOTES.md      # Documented deviations from the paper formulas, with rationale
```

## Installation

```
pip install -r requirements.txt
```

## Reproducing the paper results

### Full run (8 figures, ~2-3 hours, MC=20)

```
python main.py
```

After completion, the working directory will contain:

| File | Paper location | `--figures N` value |
|------|---------------|---|
| `fig_preliminary.pdf` | Figure 1 -- residual evolution | 1 |
| `fig_ablation.pdf`    | Figure 2 -- ablation effect sizes | 6 |
| `fig_comparative.pdf` | Figure 3 -- three-scenario convergence comparison | 2 |
| `fig_diagnostic.pdf`  | Figure 4 -- diagnostic delay | 7 |
| `fig_scaling.pdf`     | Figure 5 -- scale invariance | 8 |
| `fig_sensitivity.pdf` | Figure 6 -- parameter sensitivity | 3 |
| `fig_stability.pdf`   | Figure 7 -- stability phase diagram | 4 |
| `fig_stress.pdf`      | Figure 8 -- stress tests | 5 |
| `results.json`        | All numbers underlying Tables 1 and 2 | -- |

> Note: the `N` in `--figures N` is the **internal execution order**, not
> the paper figure number. The internal order exists so that internal
> figure 7 can reuse the MTCD data computed by internal figure 2; the
> paper figure numbers follow the section order. The two numbering
> systems differ by historical convention, and this table provides the
> bidirectional mapping.

### Quick verification (~8 minutes, MC=3)

```
python main.py --quick
```

Compare the output with [`expected_results.json`](./expected_results.json).
Specific numbers will fluctuate by ±10% due to random seeds, but
**RPS-Full should consistently beat Hard-Threshold** (this is the central
claim of paper Section 1, gated by `test_paper_core_claim`). However,
**rankings against the other baselines are not stable in quick mode** --
see the "⚠ quick vs. full mode" warning below. The most direct evidence
is the diagnostic metrics printed after internal figure 7 (paper Figure 4):

```
Hard-Threshold detection=0.000, false-alarm=0.136
RPS-Full       detection=0.525, false-alarm=0.003
```

This is direct numerical evidence for the paper Section 1 claim that
"threshold-based detector ... oscillates".

> **⚠ Quick vs. full mode**
>
> Under `--quick` (N=30, MC=3), RPS-Full's final error on the Drift
> scenario is ≈ 39.66×10⁻³, which **significantly beats
> Hard-Threshold / RPS-Sym / RPS-NoOrder** (matching the two main claims
> from paper Section 1 and Section 4.5.2), but is still **slightly worse
> than Uniform-Discount / Byzantine-Resilient (≈ 36.58)** by 4-8%. The
> reason: at N=30 with only MC=3 trials, the noise is high, and UD's
> uniform self-damping is naturally favorable for low-magnitude faults.
>
> The full claim "RPS-Full outperforms next-best by over 40%" from
> paper §4.5.3 requires the full mode (N=50, MC=20) to reproduce
> (`python main.py`, ~2 hours).
> [`expected_results_full.json`](./expected_results_full.json) is the
> snapshot of full-mode numbers actually produced at v0.4.7.
>
> See [IMPLEMENTATION_NOTES.md §20](./IMPLEMENTATION_NOTES.md) for details.

### Reproducing a specific number

Each command corresponds to a specific set of numbers in the paper.
The `N` in `--figures N` is the internal index; see the mapping table above.

| Paper location | Command | Output |
|----------|------|------|
| Table 2 + paper Figure 3 (full) | `python main.py --figures 2` | `results.json` `fig2_finals`, Wilcoxon p_adj, Cohen's d |
| Paper Table 2 RPS-Full Drift row (see absolute-number gap below) | `python main.py --figures 2 --mc 20` | "Gradual drift" row |
| Paper Figure 6 τ sensitivity | `python main.py --figures 3` | "Confidence threshold" subplot of `fig_sensitivity.pdf` |
| Paper Figure 7 κ_emp / κ_theo | `python main.py --figures 4` | Console output + `fig_stability.pdf` |
| Paper Figure 8 stress tests | `python main.py --figures 5` | `fig_stress.pdf` |
| Paper Table 1 (Drift, ablation) | `python main.py --figures 6 --mc 20` | Console "Table 1: Ablation summary" |
| Paper Figure 4 MTCD | `python main.py --figures 2,7 --mc 20` | `fig_diagnostic.pdf` |
| Paper Figure 5 N=50 vs 200 | `python main.py --figures 8` | `fig_scaling.pdf` |
| Section 4.5.4 assumption verification | `python verify_assumptions.py` | 5 PASS/FAIL items |

### Switching dataset

The default benchmark is the synthetic least-squares problem. To re-run
paper Figure 3 and Figure 2 (internal figs 2 and 6) on MNIST non-IID or
IEEE 39-bus:

```
python main.py --dataset mnist --figures 2,6 --quick
python main.py --dataset ieee39 --figures 2,6 --quick
```

## Key implementation points

See [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) for the full
itemized account. In brief:

1. **Gradient tracking (Eq. 2)**: keep `grad_old`, correctly implement
   `Y = WY + (∇f_new − ∇f_old)`.
2. **γ applied bidirectionally**: `ā_ij = γ_ij W_ij + (1−γ_ij)·δ_ij`,
   used in both X consensus and Y tracking.
3. **Fault-magnitude proxy**: take the maximum of mean-increment and
   std-increment of the residual norm sliding window; **does not read**
   ground-truth `δ`.
4. **PMF complexity reduction**: two-stage event enumeration + z-score
   replacing energy distance, O(E·s·M) -> O(E).
5. **Diagnosis throttling** `diagnose_every`: LOS is the O(E²) bottleneck,
   so we recompute every 5 steps.
6. **τ calibration**: accumulate PMF entropy during burn-in, then take
   the quantile defined by `cfg.tau_quantile`.
7. **Continuous soft discount**: `γ = exp(-gain · P_OPT)` with gain=4;
   gain is halved when entropy exceeds τ.

## Configuration knobs

All tuning parameters live in `config.RPSConfig`:

```python
from config import RPSConfig
cfg = RPSConfig(
    h_hop=2, k_trunc=3, window_len=20, eta=1.0,
    burn_in=100, tau=None, tau_quantile=0.95,
    top_m=16, diagnose_every=5,
    gain=4.0, proxy_std_weight=2.0, proxy_global_weight=0.5,
)
```

For parameter sensitivity sweeps, use `cfg.replace(eta=2.0)`.

## Self-check

```
pytest tests/             # Unit tests (~30s, includes test_paper_core_claim regression)
ruff check .              # Lint
mypy .                    # Type check (exclude already configured in pyproject.toml)
python verify_assumptions.py    # Whether paper assumptions hold under the current setup
```

## Notes

- **MC trials default to 20** (per paper Section 4.4.5). Full mode takes
  about 2-3 hours; use `--quick` (MC=3) for fast verification.
- All fault injection and diagnosis paths do not read `faulty_mask` /
  ground-truth `δ`. `verify_assumptions.py` does a static scan of the
  diagnosis path to confirm no ground-truth leakage.
- `drift_cap` defaults to 100 to keep the small-fault regime described
  in paper Section 4.4; an unbounded drift would drive all methods into
  misspecification-dominated behavior.

## Absolute-number gap vs. paper Tables 1/2

Readers will see final relative errors **3-50× larger than the values
reported in paper Tables 1/2**. For example, the paper reports Drift
RPS-Full = 1.12 (×10⁻³), while the code yields ≈ 75 (×10⁻³) in quick
mode and ≈ 50 (×10⁻³) in full mode. **The qualitative ranking (RPS-Full
< Hard-Threshold < ...) is preserved; the absolute numbers are not.**

Reasons (details in [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md)):

1. **Eq.(7) treats δ as known in the paper** -- the paper formula reads
   `Σ F_{i←j} δ_j` directly, but at runtime δ is unknown. Any
   implementation that does not read ground-truth δ uses a proxy δ
   (`magnitude_proxy`), introducing estimation error. The paper's
   numbers correspond to an "as-if δ is known" upper bound.
2. **drift_cap=80** (in figure_2 historically) keeps the fault saturated
   from t=onset+80 onwards, leaving the remaining hundreds of steps as
   constant fault, accumulating larger deviations than the paper's
   unbounded-drift formulation.
3. **N=50, T=1000 vs. paper §4.4.4 exact random seeds**: the paper does
   not publish seeds; trial-to-trial variation can be 1-2× even with the
   same configuration.

Readers should look at **inter-method relative ordering**, not absolute
values. The Wilcoxon p_adj and Cohen's d in `results.json` are the
correct metrics for evaluating "is RPS-Full significantly better than
the baselines".

## Section 4.5.4 assumption verification

```
python verify_assumptions.py
```

Outputs 5 PASS/FAIL items:

- Communication graph is strongly connected (Assumption 2)
- W is doubly stochastic and non-negative
- Local costs satisfy L-smoothness, aggregate strong convexity
  (Assumption 1)
- Small-fault regime (mean-shift dominates std-change)
- Diagnosis path does not leak ground-truth fault information
