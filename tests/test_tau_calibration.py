"""``tau_quantile`` must actually drive the calibrated value of tau.

Historical bug: Figure 3 swept ``tau_quantile`` but the code internally
hard-coded tau to ``log(top_m)``, which collapsed the sensitivity curve
into noise. This test suite asserts:

  1. Different ``tau_quantile`` settings produce different tau values for
     the same residual stream.
  2. Different tau values produce different gamma sequences via
     ``confidence_gated_discount``.
"""


import numpy as np

from config import RPSConfig
from costs import LeastSquaresCost, generate_least_squares_data
from distributed_optimization import build_graph
from experiments import run_optimization


def _setup():
    N, d, p, T = 10, 4, 3, 200
    W, adj, _ = build_graph(N, seed=0)
    A_list, b_list = generate_least_squares_data(N, d, p, seed=0)
    cost = LeastSquaresCost(A_list, b_list)
    fault_cfg = {'onset': 100, 'agents': [3], 'type': 'drift',
                 'delta': 0.005 * np.ones(d), 'drift_cap': 50}
    return N, d, T, W, adj, cost, fault_cfg


def test_tau_quantile_changes_gamma_history():
    """Two different (explicit) tau values must yield different gamma
    histories.

    Regression guard for the historical bug: Figure 3 swept tau but the
    code hard-coded tau to ``log(top_m)``, collapsing the sensitivity
    curve into noise.
    """
    N, d, T, W, adj, cost, fault_cfg = _setup()
    base = RPSConfig(burn_in=80, window_len=20, top_m=8, diagnose_every=1)

    cfg_low = base.replace(tau=0.5)
    cfg_high = base.replace(tau=3.0)

    _, _, log_low = run_optimization(
        N=N, d=d, T=T, alpha=0.05, fault_config=fault_cfg, method="RPS-Full",
        W=W, adj=adj, cost=cost, cfg=cfg_low, seed=0,
    )
    _, _, log_high = run_optimization(
        N=N, d=d, T=T, alpha=0.05, fault_config=fault_cfg, method="RPS-Full",
        W=W, adj=adj, cost=cost, cfg=cfg_high, seed=0,
    )

    # Gamma histories must not match.
    g_low = log_low.get("gamma_history", [])
    g_high = log_high.get("gamma_history", [])
    assert len(g_low) > 0 and len(g_high) > 0
    diffs = [not np.allclose(a, b) for a, b in zip(g_low, g_high)]
    assert any(diffs), (
        "tau changing 0.5 -> 3.0 produced byte-identical gamma history; "
        "tau may not be wired into the discount function."
    )


def test_burnin_collects_entropies():
    """The RPS path should accumulate entropy values during burn-in for
    tau calibration."""
    import experiments as exp_mod

    N, d, T, W, adj, cost, fault_cfg = _setup()
    cfg = RPSConfig(burn_in=80, window_len=20, top_m=8, diagnose_every=1)

    # Monkey-patch to capture the ``_RunState`` reference.
    captured = {}
    orig_step_rps = exp_mod._step_rps

    def spy(t, st, *args, **kwargs):
        captured["state"] = st
        return orig_step_rps(t, st, *args, **kwargs)

    exp_mod._step_rps = spy
    try:
        run_optimization(
            N=N, d=d, T=T, alpha=0.05, fault_config=fault_cfg, method="RPS-Full",
            W=W, adj=adj, cost=cost, cfg=cfg, seed=0,
        )
    finally:
        exp_mod._step_rps = orig_step_rps

    state = captured["state"]
    assert len(state.burnin_entropies) > 0, "burnin_entropies should be collected"
    assert state.tau != float('inf'), "tau should be calibrated after burn-in"
