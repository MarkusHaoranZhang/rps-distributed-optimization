"""论文核心论断的回归测试。

论文 Section 1 / 4.5.3 的中心论点：
    在 Gradual drift 故障下，RPS-Full 的最终相对误差显著低于 Hard-Threshold。

如果未来某次重构破坏这一性质（例如改坏 γ 计算、τ 校准、或 PMF 融合），
本测试会立刻失败。

测试用小规模 (N=20, T=400, MC=3) 让 pytest 总耗时控制在 ~30s 内，但
故障幅度足够让 RPS 优势稳定显现（多 seed 平均下不被噪声淹没）。

.. note::
   **此测试仅锁 RPS-Full vs Hard-Threshold**，不锁 RPS-Full vs
   Uniform-Discount/Byzantine-Resilient。原因详见
   ``IMPLEMENTATION_NOTES.md §20``：在 N=30, MC=3 (quick mode) 下，
   Uniform-Discount 因无差别 self-damping 天然占便宜，RPS-Full 排序
   会略低于 UD（~4%）；论文 "40% over next-best" 论断必须在 N=50,
   MC=20 (完整模式) 下才能复现。我们选择**只锁 RPS vs HT** 以避免
   把"论文论断在 quick mode 下不成立"这件事掩盖在测试假阴性里。
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
    # N=30 是 RPS-Full 优势能稳定显现的最小规模（小于 N=20 时 RPS 反而失效，
    # 因为故障智能体被屏蔽后剩余 N-1 个智能体的 misspecification 误差占主导）。
    # 17 秒跑完 RPS-Full 三趟，符合 pytest 的快速回归约束。
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
    """跑 n_trials 次 MC，返回 final error 平均值。"""
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
    """论文 Section 1 中心论点的回归测试。

    若此测试失败，说明某次代码改动破坏了 RPS 在 drift 场景下的核心优势——
    必须先排查再合并。
    """
    rps_full = _final_mean_over_seeds("RPS-Full", drift_setup)
    hard_t = _final_mean_over_seeds("Hard-Threshold", drift_setup)

    # RPS-Full 应至少比 Hard-Threshold 好 5%（quick 测试规模下的保守门槛）
    margin = 0.95
    assert rps_full < hard_t * margin, (
        f"PAPER CORE CLAIM FAILED: RPS-Full ({rps_full:.4e}) is not "
        f"meaningfully better than Hard-Threshold ({hard_t:.4e}) on Gradual "
        f"drift. Required: rps_full < {margin} * hard_t = {hard_t * margin:.4e}. "
        f"This is the central claim of Section 1; investigate recent commits."
    )


def test_rps_full_beats_no_diagnosis_baseline(drift_setup):
    """RPS-Full 应优于 Ideal-with-fault（即什么都不做的基线）。

    Ideal 方法在故障期不做任何折扣；它的 final error 应远高于 RPS-Full。
    """
    # 用同一个故障配置但跑 "Ideal" 方法（即不做任何诊断/折扣）
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
    """RPS-Full 必须在故障期记录 true-fault top1 概率。

    若 diag_log 不再记录此字段，论文 4.4.3 MTCD 指标失去基础。
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
        "MTCD metric (paper §4.4.3) cannot be computed."
    )


def test_gamma_history_recorded_during_fault(drift_setup):
    """γ 矩阵历史必须在故障期被记录到 diag_log。

    若 gamma_history 缺失，detection rate / false alarm rate 指标
    （论文 §4.4.3）无法计算。
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
        "detection/false-alarm rate metrics (paper §4.4.3) cannot be computed."
    )
