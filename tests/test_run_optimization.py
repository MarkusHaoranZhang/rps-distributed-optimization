"""``run_optimization`` 的端到端冒烟：每个方法都能正常返回有限值。"""

import numpy as np
import pytest

from config import KNOWN_METHODS, RPSConfig
from costs import LeastSquaresCost, generate_least_squares_data
from distributed_optimization import build_graph
from experiments import recovery_time, resilience_metric, run_optimization


@pytest.fixture(scope="module")
def small_setup():
    N, d, p, T = 8, 4, 3, 120
    W, adj, _ = build_graph(N, seed=0)
    A_list, b_list = generate_least_squares_data(N, d, p, seed=0)
    cost = LeastSquaresCost(A_list, b_list)
    return dict(N=N, d=d, T=T, W=W, adj=adj, cost=cost)


@pytest.mark.parametrize("method", list(KNOWN_METHODS))
def test_each_method_runs(method, small_setup):
    s = small_setup
    cfg = RPSConfig(burn_in=40, window_len=10, top_m=8, diagnose_every=5)
    fault_cfg = ({'onset': s['T'] + 1, 'agents': [], 'type': 'constant',
                  'delta': None}
                 if method == "Ideal"
                 else {'onset': 50, 'agents': [2], 'type': 'drift',
                        'delta': 0.005 * np.ones(s['d']), 'drift_cap': 30})
    err, res, log = run_optimization(
        N=s['N'], d=s['d'], T=s['T'], alpha=0.05,
        fault_config=fault_cfg, method=method,
        W=s['W'], adj=s['adj'], cost=s['cost'], cfg=cfg, seed=0,
    )
    assert err.shape == (s['T'],)
    assert res.shape == (s['T'], s['N'])
    assert np.all(np.isfinite(err)), f"{method} produced non-finite errors"


def test_ideal_converges_close_to_zero(small_setup):
    """无故障 Ideal 应在足够步数后收敛到比初始误差小至少两个数量级。"""
    s = small_setup
    cfg = RPSConfig(burn_in=40, window_len=10, top_m=8, diagnose_every=5)
    T_long = 600
    err_ideal, _, _ = run_optimization(
        N=s['N'], d=s['d'], T=T_long, alpha=0.05,
        fault_config={'onset': T_long + 1, 'agents': [], 'type': 'constant',
                       'delta': None},
        method="Ideal", W=s['W'], adj=s['adj'], cost=s['cost'],
        cfg=cfg, seed=0,
    )
    assert err_ideal[-1] < err_ideal[0] / 100, (
        f"Ideal err did not drop 2 orders: {err_ideal[0]} → {err_ideal[-1]}")


def test_recovery_time_is_finite(small_setup):
    """RPS-Full 在 drift 下应在故障期内有可测的 recovery time（非 nan）。"""
    s = small_setup
    cfg = RPSConfig(burn_in=40, window_len=10, top_m=8, diagnose_every=5)
    fault_cfg = {'onset': 50, 'agents': [2], 'type': 'constant',
                 'delta': 0.005 * np.ones(s['d'])}
    err, _, _ = run_optimization(
        N=s['N'], d=s['d'], T=s['T'], alpha=0.05,
        fault_config=fault_cfg, method="RPS-Full",
        W=s['W'], adj=s['adj'], cost=s['cost'], cfg=cfg, seed=0,
    )
    rt = recovery_time(err, fault_cfg['onset'])
    # 不要求一定在故障内恢复；但要求不是 nan
    assert not (rt != rt), "recovery_time returned nan"


def test_resilience_metric_nonneg(small_setup):
    """resilience_metric ≥ 0。"""
    s = small_setup
    cfg = RPSConfig(burn_in=40, window_len=10, top_m=8, diagnose_every=5)
    fault_cfg = {'onset': 50, 'agents': [2], 'type': 'constant',
                 'delta': 0.005 * np.ones(s['d'])}
    err_ideal, _, _ = run_optimization(
        N=s['N'], d=s['d'], T=s['T'], alpha=0.05,
        fault_config={'onset': 9999, 'agents': [], 'type': 'constant',
                       'delta': None},
        method="Ideal", W=s['W'], adj=s['adj'], cost=s['cost'],
        cfg=cfg, seed=0,
    )
    err, _, _ = run_optimization(
        N=s['N'], d=s['d'], T=s['T'], alpha=0.05,
        fault_config=fault_cfg, method="RPS-Full",
        W=s['W'], adj=s['adj'], cost=s['cost'], cfg=cfg, seed=0,
    )
    r = resilience_metric(err, err_ideal)
    assert r >= 0


def test_unknown_method_raises(small_setup):
    s = small_setup
    cfg = RPSConfig()
    with pytest.raises(ValueError):
        run_optimization(
            N=s['N'], d=s['d'], T=s['T'], alpha=0.05,
            fault_config={'onset': 100, 'agents': [], 'type': 'constant',
                           'delta': None},
            method="DefinitelyNotAMethod",
            W=s['W'], adj=s['adj'], cost=s['cost'], cfg=cfg, seed=0,
        )
