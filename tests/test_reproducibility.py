"""Reproducibility tests: identical seeds must produce byte-identical
output."""

import numpy as np

from config import RPSConfig
from costs import LeastSquaresCost, generate_least_squares_data
from distributed_optimization import build_graph
from experiments import run_optimization


def _setup():
    N, d, p, T = 8, 4, 3, 100
    W, adj, _ = build_graph(N, seed=0)
    A_list, b_list = generate_least_squares_data(N, d, p, seed=0)
    cost = LeastSquaresCost(A_list, b_list)
    cfg = RPSConfig(burn_in=40, window_len=10, top_m=8, diagnose_every=5)
    fault_cfg = {'onset': 50, 'agents': [2], 'type': 'drift',
                 'delta': 0.005 * np.ones(d), 'drift_cap': 30}
    return N, d, T, W, adj, cost, cfg, fault_cfg


def test_same_seed_same_output_rps_full():
    N, d, T, W, adj, cost, cfg, fault_cfg = _setup()
    err1, _, _ = run_optimization(
        N=N, d=d, T=T, alpha=0.05, fault_config=fault_cfg, method="RPS-Full",
        W=W, adj=adj, cost=cost, cfg=cfg, seed=42,
    )
    err2, _, _ = run_optimization(
        N=N, d=d, T=T, alpha=0.05, fault_config=fault_cfg, method="RPS-Full",
        W=W, adj=adj, cost=cost, cfg=cfg, seed=42,
    )
    np.testing.assert_array_equal(err1, err2)


def test_different_seed_different_output():
    N, d, T, W, adj, cost, cfg, fault_cfg = _setup()
    err1, _, _ = run_optimization(
        N=N, d=d, T=T, alpha=0.05, fault_config=fault_cfg, method="RPS-Full",
        W=W, adj=adj, cost=cost, cfg=cfg, seed=42,
    )
    err2, _, _ = run_optimization(
        N=N, d=d, T=T, alpha=0.05, fault_config=fault_cfg, method="RPS-Full",
        W=W, adj=adj, cost=cost, cfg=cfg, seed=43,
    )
    # Different seeds should not produce exactly the same trajectory.
    assert not np.allclose(err1, err2)


def test_baseline_methods_reproducible():
    """All non-RPS methods must also reproduce byte-identically."""
    N, d, T, W, adj, cost, cfg, fault_cfg = _setup()
    for method in ("Hard-Threshold", "Uniform-Discount", "Byzantine-Resilient"):
        err1, _, _ = run_optimization(
            N=N, d=d, T=T, alpha=0.05, fault_config=fault_cfg, method=method,
            W=W, adj=adj, cost=cost, cfg=cfg, seed=7,
        )
        err2, _, _ = run_optimization(
            N=N, d=d, T=T, alpha=0.05, fault_config=fault_cfg, method=method,
            W=W, adj=adj, cost=cost, cfg=cfg, seed=7,
        )
        np.testing.assert_array_equal(err1, err2,
            err_msg=f"method {method} not reproducible")
