"""Regression test for the paper's core claim.

Sections 1 and 4.5.3 of the paper make the central claim:

    Under Gradual drift, RPS-Full's final relative error is meaningfully
    lower than Hard-Threshold's.

If a future refactor breaks this property (e.g. a regression in gamma
computation, tau calibration, or PMF fusion), this test fails immediately.

The test deliberately uses small sizes (N=20, T=400, MC=3) so the total
pytest runtime stays around ~30s, while the fault magnitude is large
enough for the RPS advantage to show stably across multiple seeds without
being drowned in noise.

.. note::
   **This test only pins down "RPS-Full vs Hard-Threshold"**, not
   "RPS-Full vs Uniform-Discount / Byzantine-Resilient". The reason is
   in ``IMPLEMENTATION_NOTES.md`` Sec. 20: at N=30, MC=3 (quick mode),
   Uniform-Discount benefits from indiscriminate self-damping, and
   RPS-Full's ranking can drop below UD by ~4%. The paper's
   "40% over next-best" claim only reproduces at N=50, MC=20 (full
   mode). We deliberately lock down only "RPS vs HT" to avoid hiding
   "the paper's claim does not hold under quick mode" inside a false
   negative.
"""

import numpy as np
import pytest

from config import RPSConfig
from costs import LeastSquaresCost, generate_least_squares_data
from distributed_optimization import build_graph
from experiments import run_optimization

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def drift_setup():
    # N=30 is the smallest size at which RPS-Full's advantage shows up
    # stably (below N=20, RPS actually fails because once the faulty
    # agent is masked out, misspecification error from the remaining
    # N-1 agents dominates). RPS-Full runs three trials in ~17s, which
    # fits within the fast-regression budget for pytest.
    N, d, p, T = 30, 5, 3, 500
    W, adj, _ = build_graph(N, seed=0)
    A_list, b_list = generate_least_squares_data(N, d, p, seed=0)
    cost = LeastSquaresCost(A_list, b_list)
    fault_cfg = {'onset': T // 3, 'agents': [3], 'type': 'drift',
                 'delta': 0.005 * np.ones(d), 'drift_cap': 40}
    cfg = RPSConfig(burn_in=max(50, fault_cfg['onset'] - 50),
                    window_len=20, top_m=16, diagnose_every=5)
    return dict(N=N, d=d, T=T, W=W, adj=adj, cost=cost,
                fault_cfg=fault_cfg, cfg=cfg)


def _final_mean_over_seeds(method, setup, n_trials=3):
    """Run ``n_trials`` Monte Carlo trials and return the mean final
    error."""
    finals = []
    for trial in range(n_trials):
        err, _, _ = run_optimization(
            N=setup['N'], d=setup['d'], T=setup['T'], alpha=0.05,
            fault_config=setup['fault_cfg'], method=method,
            W=setup['W'], adj=setup['adj'], cost=setup['cost'],
            cfg=setup['cfg'], seed=trial * 7,
        )
        finals.append(err[-1])
    return float(np.mean(finals))


# ---------------------------------------------------------------------------
# Core claim: RPS-Full beats Hard-Threshold on Drift
# ---------------------------------------------------------------------------

def test_rps_full_beats_hard_threshold_on_drift(drift_setup):
    """Regression test for the central claim of Section 1.

    If this test fails, a recent commit broke RPS's core advantage on
    drift; investigate before merging.
    """
    rps_full = _final_mean_over_seeds("RPS-Full", drift_setup)
    hard_t = _final_mean_over_seeds("Hard-Threshold", drift_setup)

    # RPS-Full should beat Hard-Threshold by at least 5% (a conservative
    # margin at the quick test scale).
    margin = 0.95
    assert rps_full < hard_t * margin, (
        f"PAPER CORE CLAIM FAILED: RPS-Full ({rps_full:.4e}) is not "
        f"meaningfully better than Hard-Threshold ({hard_t:.4e}) on Gradual "
        f"drift. Required: rps_full < {margin} * hard_t = {hard_t * margin:.4e}. "
        f"This is the central claim of Section 1; investigate recent commits."
    )


def test_rps_full_beats_no_diagnosis_baseline(drift_setup):
    """RPS-Full should beat the "Ideal-with-fault" (do-nothing) baseline.

    The ``Ideal`` method does no discounting in the fault phase; its
    final error must be much higher than RPS-Full's.
    """
    # Same fault config but using the ``Ideal`` method (no diagnosis,
    # no discount).
    setup = drift_setup
    finals_ideal = []
    for trial in range(3):
        err, _, _ = run_optimization(
            N=setup['N'], d=setup['d'], T=setup['T'], alpha=0.05,
            fault_config=setup['fault_cfg'], method="Ideal",
            W=setup['W'], adj=setup['adj'], cost=setup['cost'],
            cfg=setup['cfg'], seed=trial * 7,
        )
        finals_ideal.append(err[-1])
    no_diag_err = float(np.mean(finals_ideal))
    rps_full = _final_mean_over_seeds("RPS-Full", setup)

    assert rps_full < no_diag_err * 0.95, (
        f"RPS-Full ({rps_full:.4e}) does not improve over no-diagnosis "
        f"baseline ({no_diag_err:.4e}). The diagnostic discount is not "
        f"providing meaningful benefit."
    )


def test_rps_full_diagnostic_log_records_top1(drift_setup):
    """RPS-Full must record the true-fault top-1 probability in
    ``diag_log`` during the fault period.

    Without this field, the MTCD metric in Section 4.4.3 of the paper
    has no basis.
    """
    err, _, log = run_optimization(
        N=drift_setup['N'], d=drift_setup['d'], T=drift_setup['T'],
        alpha=0.05, fault_config=drift_setup['fault_cfg'], method="RPS-Full",
        W=drift_setup['W'], adj=drift_setup['adj'], cost=drift_setup['cost'],
        cfg=drift_setup['cfg'], seed=0,
    )
    assert "true_fault_top1_prob" in log
    assert len(log["true_fault_top1_prob"]) > 0, (
        "RPS-Full diag_log['true_fault_top1_prob'] is empty; "
        "MTCD metric (paper Section 4.4.3) cannot be computed."
    )


def test_gamma_history_recorded_during_fault(drift_setup):
    """The gamma matrix history must be recorded into ``diag_log``
    during the fault period.

    Without ``gamma_history`` the detection / false-alarm rate metrics
    (paper Section 4.4.3) cannot be computed.
    """
    _, _, log = run_optimization(
        N=drift_setup['N'], d=drift_setup['d'], T=drift_setup['T'],
        alpha=0.05, fault_config=drift_setup['fault_cfg'], method="RPS-Full",
        W=drift_setup['W'], adj=drift_setup['adj'], cost=drift_setup['cost'],
        cfg=drift_setup['cfg'], seed=0,
    )
    assert "gamma_history" in log
    assert len(log["gamma_history"]) > 0, (
        "RPS-Full diag_log['gamma_history'] is empty; "
        "detection / false-alarm rate metrics (paper Section 4.4.3) "
        "cannot be computed."
    )
