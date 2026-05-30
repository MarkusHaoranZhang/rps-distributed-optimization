"""
主运行脚本：执行所有实验并生成论文 8 张图与 2 张表。

用法
----
::

    python main.py                    # 完整跑全部 8 张图（MC=20，约 2-3 小时）
    python main.py --quick            # 缩减 N/T/MC 快速验证（MC=3，约 8 分钟）
    python main.py --figures 1,2,6    # 只跑 figure 1, 2, 6（代码内编号）
    python main.py --quick --figures 6,7

每张图都被封装成 ``figure_<n>(quick: bool, *args)`` 函数，可以单独跑。
共享的实验上下文（图、代价、统计输出）通过 ``ExperimentContext`` 传递。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from config import RPSConfig
from costs import LeastSquaresCost, generate_least_squares_data
from distributed_optimization import build_graph, simulate_symmetric_packet_loss
from experiments import (
    detection_and_false_alarm_rates,
    mean_time_to_correct_diagnosis,
    recovery_time,
    resilience_metric,
    run_optimization,
)
from figures import (
    plot_figure1,
    plot_figure2,
    plot_figure3,
    plot_figure4,
    plot_figure5,
    plot_figure6,
    plot_figure7,
    plot_figure8,
)
from statistics_utils import cohens_d, holm_bonferroni, wilcoxon_pvalue

# ---------------------------------------------------------------------------
# Default experiment parameters
# ---------------------------------------------------------------------------

D_SYN = 10           # 论文 Section 4.4.1
P_SYN = 5


@dataclass
class _Sizes:
    """Quick / full 模式下各 figure 的规模设置。"""
    fig2_N: int
    fig2_T: int
    fig2_mc: int
    fig3_N: int
    fig3_T: int
    fig4_N: int
    fig4_npts: int
    fig5_N: int
    fig5_mc: int
    fig6_N: int
    fig6_T: int
    fig6_mc: int
    fig7_mc: int
    fig8_Ns: tuple

    @classmethod
    def quick(cls) -> "_Sizes":
        # quick 模式：fig2_T=600, onset=200。``mc_run`` 会再把 burn_in 裁剪到
        # min(cfg.burn_in, onset-10) = min(100, 190) = 100，正好让 burn-in 期
        # 累积 80 步 (window_len=20 之后开始采样) 的熵，τ 分位估计样本足够。
        return cls(fig2_N=30, fig2_T=600, fig2_mc=3,
                   fig3_N=20, fig3_T=400,
                   fig4_N=15, fig4_npts=36,
                   fig5_N=20, fig5_mc=3,
                   fig6_N=30, fig6_T=600, fig6_mc=3,
                   fig7_mc=3,
                   fig8_Ns=(30, 80))

    @classmethod
    def full(cls) -> "_Sizes":
        # 论文 Section 4.4.5 声明 20 次 Monte Carlo。完整模式按论文复现。
        # 跑全套约 2-3 小时；要更快可用 ``--quick`` 或 ``--mc N`` 覆盖。
        return cls(fig2_N=50, fig2_T=1000, fig2_mc=20,
                   fig3_N=30, fig3_T=500,
                   fig4_N=20, fig4_npts=64,
                   fig5_N=30, fig5_mc=20,
                   fig6_N=50, fig6_T=600, fig6_mc=20,
                   fig7_mc=20,
                   fig8_Ns=(50, 200))

    def with_mc(self, mc: int) -> "_Sizes":
        """覆盖所有 MC 字段为指定值。用于 ``--mc N`` 参数。"""
        from dataclasses import replace
        return replace(self, fig2_mc=mc, fig5_mc=mc, fig6_mc=mc, fig7_mc=mc)


# ---------------------------------------------------------------------------
# Shared run context
# ---------------------------------------------------------------------------

@dataclass
class ExperimentContext:
    """跨 figure 共享的状态：base RPS config、统计累积、JSON 输出。"""
    sizes: _Sizes
    base_cfg: RPSConfig
    dataset: str = "synthetic"
    fig2_finals: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    fig2_iters: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    fig2_curves: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    fig2_mtcd: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    fig2_recovery: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    fig2_resilience: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    fig2_detection: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    fig2_false_alarm: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    ablation: Dict[str, np.ndarray] = field(default_factory=dict)
    kappa_emp: float = float('nan')
    kappa_theo: float = float('nan')
    fault_scenarios: Dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mc_run(method: str, *, N: int, T: int, alpha: float, fault_cfg: dict,
            W: np.ndarray, adj: np.ndarray, cost, n_trials: int,
            seed_base: int, cfg: RPSConfig,
            return_logs: bool = False):
    """对一个 (method, fault_cfg) 跑 ``n_trials`` 次 Monte Carlo。

    .. note::
       如果 ``cfg.burn_in`` 被故障 onset 强制裁剪（裁剪发生在 onset 不够大、
       不够留给 burn-in 时），会通过 ``warnings`` 发出一条 UserWarning。
       这是为了让用户在配置 ``cfg.burn_in`` 时不会被静默改写。
    """
    d = cost.problem_dim()
    burn_in = cfg.burn_in
    if fault_cfg.get('onset') is not None:
        # burn_in 必须严格小于 onset，且至少留 30 步给故障前的统计稳定
        # （window_len=20 + 缓冲 10）。否则 τ 校准样本太少。
        max_burn_in = max(cfg.window_len + 10, fault_cfg['onset'] - 10)
        if burn_in > max_burn_in:
            import warnings
            warnings.warn(
                f"mc_run: cfg.burn_in={burn_in} clipped to {max_burn_in} "
                f"because fault onset={fault_cfg['onset']} leaves too little "
                f"room (need ≥ window_len+10 = {cfg.window_len + 10}). "
                f"This affects τ calibration sample size.",
                stacklevel=2,
            )
            burn_in = max_burn_in
    cfg_use = cfg.replace(burn_in=burn_in)
    all_err = np.zeros((n_trials, T))
    all_logs: List[dict] = []
    for k in range(n_trials):
        err, _, log = run_optimization(
            N=N, d=d, T=T, alpha=alpha, fault_config=fault_cfg, method=method,
            W=W, adj=adj, cost=cost, cfg=cfg_use, seed=seed_base + 13 * k,
        )
        all_err[k] = err
        if return_logs:
            all_logs.append(log)
    return (all_err, all_logs) if return_logs else all_err


def _ideal_cfg(T: int) -> dict:
    """Ideal 方法专用故障配置（onset 大于 T 等价无故障）。"""
    return {'onset': T + 1, 'agents': [], 'type': 'constant', 'delta': None}


def _make_cost_and_graph(dataset: str, N: int, seed: int):
    """根据数据集名构造 (W, adj, cost, d)。

    - "synthetic": 合成最小二乘 (论文 4.4.1 主基准)
    - "mnist"    : MNIST 非 IID 多项 logistic 回归 (Section 4.4.1)
    - "ieee39"   : IEEE 39-bus 经济调度 (Section 4.4.1)；强制 N=39
    """
    cost: Any  # 三个分支返回不同的 cost 子类，避开 mypy 单分支类型锁定
    if dataset == "synthetic":
        W, adj, _ = build_graph(N, seed=seed)
        A_list, b_list = generate_least_squares_data(N, D_SYN, P_SYN, seed=seed)
        cost = LeastSquaresCost(A_list, b_list)
        return W, adj, cost, D_SYN
    if dataset == "mnist":
        from datasets import make_mnist_noniid
        cost = make_mnist_noniid(N=N, seed=seed)
        W, adj, _ = build_graph(N, seed=seed)
        return W, adj, cost, cost.problem_dim()
    if dataset == "ieee39":
        from datasets import make_ieee39_dispatch
        cost = make_ieee39_dispatch(seed=seed)
        # IEEE 39-bus 固定 39 个 generator
        W, adj, _ = build_graph(39, seed=seed)
        return W, adj, cost, cost.problem_dim()
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def figure_1(ctx: ExperimentContext) -> None:
    print("\n[1/8] Figure 1: Residual evolution...")
    t0 = time.time()
    N1 = 10
    W1, adj1, _ = build_graph(N1, seed=42)
    A_list, b_list = generate_least_squares_data(N1, D_SYN, P_SYN, seed=42)
    cost1 = LeastSquaresCost(A_list, b_list)
    fault_cfg = {'onset': 500, 'agents': [0], 'type': 'constant',
                  'delta': 0.01 * np.ones(D_SYN)}
    cfg = ctx.base_cfg.replace(burn_in=400)
    _, residuals, _ = run_optimization(
        N=N1, d=D_SYN, T=600, alpha=0.05,
        fault_config=fault_cfg, method="Ideal",
        W=W1, adj=adj1, cost=cost1, cfg=cfg, seed=42,
    )
    nb1 = set(np.where(adj1[0] > 0)[0])
    nb2: set = set()
    for n in nb1:
        nb2.update(np.where(adj1[n] > 0)[0])
    nb2 -= nb1; nb2 -= {0}
    direct_idx = next(iter(nb1)) if nb1 else 1
    twohop_idx = next(iter(nb2)) if nb2 else 3
    plot_figure1(residuals, faulty_idx=0, direct_idx=direct_idx,
                 twohop_idx=twohop_idx, fault_onset=500)
    print(f"  -> fig_preliminary.pdf saved. ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Figure 2 (also fills ctx.fig2_* for downstream reuse by figure 7 / tables)
# ---------------------------------------------------------------------------

_FIG2_METHODS = ("Ideal", "Hard-Threshold", "Uniform-Discount",
                 "Byzantine-Resilient", "RPS-Symmetric", "RPS-Full")

# 跨进程稳定的种子偏移（避免依赖 PYTHONHASHSEED-randomized hash）
_SCENARIO_SEED_OFFSET = {
    "Constant bias": 100,
    "Gradual drift": 200,
    "Intermittent": 300,
}


def figure_2(ctx: ExperimentContext) -> None:
    s = ctx.sizes
    print(f"\n[2/8] Figure 2: Comparative convergence "
          f"(N={s.fig2_N}, T={s.fig2_T}, MC={s.fig2_mc}, dataset={ctx.dataset})...")
    t0 = time.time()
    W, adj, cost, d_actual = _make_cost_and_graph(ctx.dataset, s.fig2_N, seed=0)
    N_actual = W.shape[0]   # ieee39 强制 39

    onset = s.fig2_T // 3
    ctx.fault_scenarios = {
        "Constant bias": {'onset': onset, 'agents': [4], 'type': 'constant',
                          'delta': 0.01 * np.ones(d_actual)},
        "Gradual drift": {'onset': onset, 'agents': [4], 'type': 'drift',
                          'delta': 0.002 * np.ones(d_actual), 'drift_cap': 40},
        "Intermittent": {'onset': onset, 'agents': [4], 'type': 'intermittent',
                         'delta': 0.03 * np.ones(d_actual), 'prob': 0.3},
    }

    for sc, fault_cfg in ctx.fault_scenarios.items():
        ctx.fig2_curves[sc] = {}
        ctx.fig2_finals[sc] = {}
        ctx.fig2_iters[sc] = {}
        ctx.fig2_mtcd[sc] = {}
        ctx.fig2_recovery[sc] = {}
        ctx.fig2_resilience[sc] = {}
        ctx.fig2_detection[sc] = {}
        ctx.fig2_false_alarm[sc] = {}
        for m in _FIG2_METHODS:
            cfg_use = _ideal_cfg(s.fig2_T) if m == "Ideal" else fault_cfg
            t1 = time.time()
            err_mat, logs = mc_run(
                method=m, N=N_actual, T=s.fig2_T, alpha=0.05,
                fault_cfg=cfg_use, W=W, adj=adj, cost=cost,
                n_trials=s.fig2_mc,
                seed_base=10 + _SCENARIO_SEED_OFFSET.get(sc, 0),
                cfg=ctx.base_cfg, return_logs=True,
            )
            ctx.fig2_curves[sc][m] = err_mat.mean(axis=0)
            ctx.fig2_finals[sc][m] = err_mat[:, -1]

            its = []
            for trial in range(s.fig2_mc):
                hit = np.where(err_mat[trial] < 1e-3)[0]
                its.append(int(hit[0]) if len(hit) else s.fig2_T)
            ctx.fig2_iters[sc][m] = np.array(its)

            if m in ("RPS-Full", "RPS-Symmetric", "RPS-NoOrder", "Hard-Threshold"):
                # MTCD 用论文 4.4.3 定义：true-fault top1 概率首次 ≥ 0.95 的迭代数
                mtcd_vals = []
                det_vals: list = []
                fa_vals: list = []
                for log in logs:
                    v = mean_time_to_correct_diagnosis(log, fault_cfg['onset'],
                                                        prob_threshold=0.95)
                    if not np.isnan(v):
                        mtcd_vals.append(v)
                    # 故障检测率 / 误报率（论文 4.4.3）
                    det, fa = detection_and_false_alarm_rates(
                        log.get("gamma_history", []),
                        fault_cfg.get('agents', []),
                        N=N_actual, adj=adj,
                    )
                    if not np.isnan(det):
                        det_vals.append(det)
                    if not np.isnan(fa):
                        fa_vals.append(fa)
                ctx.fig2_mtcd[sc][m] = (np.array(mtcd_vals) if mtcd_vals
                                         else np.array([np.nan]))
                ctx.fig2_detection[sc][m] = (np.array(det_vals) if det_vals
                                              else np.array([np.nan]))
                ctx.fig2_false_alarm[sc][m] = (np.array(fa_vals) if fa_vals
                                                 else np.array([np.nan]))

            ideal_curve = ctx.fig2_curves[sc].get("Ideal", err_mat.mean(axis=0))
            ctx.fig2_recovery[sc][m] = np.array([
                recovery_time(err_mat[k], fault_cfg['onset']) for k in range(s.fig2_mc)
            ])
            ctx.fig2_resilience[sc][m] = np.array([
                resilience_metric(err_mat[k], ideal_curve) for k in range(s.fig2_mc)
            ])
            print(f"    {sc:>16s} | {m:22s} "
                  f"final={err_mat[:, -1].mean():.3e}  ({time.time()-t1:.1f}s)",
                  flush=True)

    plot_figure2(ctx.fig2_curves, list(_FIG2_METHODS),
                  list(ctx.fault_scenarios.keys()))
    print(f"  -> fig_comparative.pdf saved. ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

def figure_3(ctx: ExperimentContext) -> None:
    s = ctx.sizes
    print(f"\n[3/8] Figure 3: Parameter sensitivity (N={s.fig3_N}, T={s.fig3_T})...")
    t0 = time.time()
    W, adj, _ = build_graph(s.fig3_N, seed=1)
    A_list, b_list = generate_least_squares_data(s.fig3_N, D_SYN, P_SYN, seed=1)
    cost = LeastSquaresCost(A_list, b_list)
    base_fault = {'onset': s.fig3_T // 3, 'agents': [5], 'type': 'drift',
                  'delta': 0.002 * np.ones(D_SYN), 'drift_cap': 40}
    onset_for_burnin = int(s.fig3_T // 3)  # 与 base_fault['onset'] 同值，但 mypy 能识别其为 int

    def run_one(**overrides) -> float:
        cfg = ctx.base_cfg.replace(burn_in=max(50, onset_for_burnin - 50))
        cfg = cfg.replace(**overrides)
        err, _, _ = run_optimization(
            N=s.fig3_N, d=D_SYN, T=s.fig3_T, alpha=0.05,
            fault_config=base_fault, method="RPS-Full",
            W=W, adj=adj, cost=cost, cfg=cfg, seed=0,
        )
        return float(err[-1])

    s_vals = np.array([5, 10, 20, 30, 50])
    eta_vals = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    tau_vals = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    h_vals = np.array([1, 2, 3])

    fig3_data = {
        "Window length": (s_vals,
                          np.array([run_one(window_len=int(v)) for v in s_vals])),
        "Temperature":   (eta_vals,
                          np.array([run_one(eta=float(v)) for v in eta_vals])),
        "Confidence threshold": (tau_vals,
                                  np.array([run_one(tau=float(v)) for v in tau_vals])),
        "Hop count":     (h_vals,
                          np.array([run_one(h_hop=int(v)) for v in h_vals])),
    }
    plot_figure3(fig3_data)
    print(f"  -> fig_sensitivity.pdf saved. ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Figure 4 — stability phase diagram
# ---------------------------------------------------------------------------

def figure_4(ctx: ExperimentContext) -> None:
    s = ctx.sizes
    print(f"\n[4/8] Figure 4: Stability phase diagram (N={s.fig4_N}, npts={s.fig4_npts})...")
    t0 = time.time()
    W, adj, _ = build_graph(s.fig4_N, seed=2)
    A_list, b_list = generate_least_squares_data(s.fig4_N, D_SYN, P_SYN, seed=2)
    cost = LeastSquaresCost(A_list, b_list)
    fault_cfg = {'onset': 100, 'agents': [3], 'type': 'constant',
                  'delta': 0.02 * np.ones(D_SYN)}
    alphas = np.logspace(-2.5, -0.7, s.fig4_npts)
    etas_inv = np.logspace(-0.3, 0.7, s.fig4_npts)
    conv_mask = np.zeros(s.fig4_npts, dtype=bool)

    for i, (a, e_inv) in enumerate(zip(alphas, etas_inv)):
        cfg = ctx.base_cfg.replace(burn_in=80, eta=1.0 / float(e_inv))
        err, _, _ = run_optimization(
            N=s.fig4_N, d=D_SYN, T=300, alpha=float(a),
            fault_config=fault_cfg, method="RPS-Full",
            W=W, adj=adj, cost=cost, cfg=cfg, seed=i,
        )
        conv_mask[i] = bool(np.isfinite(err[-1]) and err[-1] < 0.5)

    if conv_mask.any():
        ratios = alphas[conv_mask] / etas_inv[conv_mask]
        ctx.kappa_emp = float(ratios.max() * 1.05)
    else:
        ctx.kappa_emp = 1e-3

    # 论文 Theorem 1 的理论 κ：μ·λ₂(L) / (c₁·L_OPT·L²·Δ)
    # 用合成 LS 上可观测的量算出 κ_theo 作为对比
    H_global = sum(A_list[i].T @ A_list[i] / P_SYN for i in range(s.fig4_N))
    eigs_H = np.linalg.eigvalsh(H_global)
    mu_global = float(max(eigs_H.min(), 1e-6))
    L_global = float(eigs_H.max())
    # 算法 Laplacian 的 λ₂ (Fiedler 值)
    deg = np.diag(adj.sum(axis=1))
    Lap = deg - adj
    eigs_L = np.linalg.eigvalsh(Lap)
    eigs_L.sort()
    lambda2 = float(eigs_L[1]) if len(eigs_L) >= 2 else 1.0
    Delta = float(np.linalg.norm(np.asarray(fault_cfg['delta'])))
    # c1 与 L_OPT 是论文常数；取保守值 c1 = 1, L_OPT = 1
    c1, L_OPT = 1.0, 1.0
    kappa_theo = (mu_global * lambda2) / (c1 * L_OPT * L_global**2 * Delta + 1e-12)
    ctx.kappa_theo = kappa_theo

    plot_figure4(alphas, etas_inv, conv_mask, ctx.kappa_emp,
                  kappa_theo=kappa_theo)
    print(f"  -> fig_stability.pdf saved. κ_emp ≈ {ctx.kappa_emp:.3e}, "
          f"κ_theo ≈ {kappa_theo:.3e} ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Figure 5 — stress tests
# ---------------------------------------------------------------------------

def figure_5(ctx: ExperimentContext) -> None:
    s = ctx.sizes
    print(f"\n[5/8] Figure 5: Stress tests (N={s.fig5_N}, MC={s.fig5_mc})...")
    t0 = time.time()
    W, adj, _ = build_graph(s.fig5_N, seed=3)
    A_list, b_list = generate_least_squares_data(s.fig5_N, D_SYN, P_SYN, seed=3)
    cost = LeastSquaresCost(A_list, b_list)

    deltas = np.linspace(0.02, 0.20, 10)
    gaps = []
    for delta in deltas:
        fc = {'onset': 150, 'agents': [0], 'type': 'constant',
              'delta': float(delta) * np.ones(D_SYN)}
        e_rps = mc_run("RPS-Full", N=s.fig5_N, T=400, alpha=0.05, fault_cfg=fc,
                        W=W, adj=adj, cost=cost, n_trials=s.fig5_mc,
                        seed_base=200, cfg=ctx.base_cfg)
        e_ht = mc_run("Hard-Threshold", N=s.fig5_N, T=400, alpha=0.05, fault_cfg=fc,
                       W=W, adj=adj, cost=cost, n_trials=s.fig5_mc,
                       seed_base=200, cfg=ctx.base_cfg)
        gaps.append((e_ht[:, -1].mean() - e_rps[:, -1].mean()) /
                    (e_ht[:, -1].mean() + 1e-12) * 100)

    loss_rates = np.array([0, 5, 10, 15, 20, 25, 35, 50, 60, 75])
    perf_retain: list = []
    drift_cfg = {'onset': 150, 'agents': [2], 'type': 'drift',
                  'delta': 0.002 * np.ones(D_SYN), 'drift_cap': 40}
    for loss in loss_rates:
        rng = np.random.RandomState(int(loss) + 1)
        # 对称丢包模拟见 ``distributed_optimization.simulate_symmetric_packet_loss``
        # 的 docstring 与 ``tests/test_figure5_packet_loss.py`` 守门测试。
        W_mod = simulate_symmetric_packet_loss(W, loss / 100.0, rng)
        e = mc_run("RPS-Full", N=s.fig5_N, T=400, alpha=0.05,
                    fault_cfg=drift_cfg, W=W_mod, adj=adj, cost=cost,
                    n_trials=max(2, s.fig5_mc // 2), seed_base=300,
                    cfg=ctx.base_cfg)
        perf_retain.append(e[:, -1].mean())

    nfaults_arr = np.arange(1, 6)
    accs: list = []
    for nf in nfaults_arr:
        fc = {'onset': 150, 'agents': list(range(int(nf))), 'type': 'constant',
              'delta': 0.01 * np.ones(D_SYN)}
        e = mc_run("RPS-Full", N=s.fig5_N, T=400, alpha=0.05, fault_cfg=fc,
                    W=W, adj=adj, cost=cost,
                    n_trials=max(2, s.fig5_mc // 2), seed_base=400,
                    cfg=ctx.base_cfg)
        accs.append(e[:, -1].mean())

    plot_figure5(deltas, np.array(gaps), loss_rates,
                  np.array(perf_retain), nfaults_arr, np.array(accs))
    print(f"  -> fig_stress.pdf saved. ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Figure 6 — ablation
# ---------------------------------------------------------------------------

def figure_6(ctx: ExperimentContext) -> None:
    s = ctx.sizes
    print(f"\n[6/8] Figure 6: Ablation (MC={s.fig6_mc}, T={s.fig6_T})...")
    t0 = time.time()
    W, adj, _ = build_graph(s.fig6_N, seed=5)
    A_list, b_list = generate_least_squares_data(s.fig6_N, D_SYN, P_SYN, seed=5)
    cost = LeastSquaresCost(A_list, b_list)
    fault_cfg = {'onset': s.fig6_T // 3, 'agents': [4], 'type': 'drift',
                  'delta': 0.002 * np.ones(D_SYN), 'drift_cap': 40}

    def _final(method: str) -> np.ndarray:
        return mc_run(method, N=s.fig6_N, T=s.fig6_T, alpha=0.05,
                      fault_cfg=fault_cfg, W=W, adj=adj, cost=cost,
                      n_trials=s.fig6_mc, seed_base=500,
                      cfg=ctx.base_cfg)[:, -1]

    full_err = _final("RPS-Full")
    no_err = _final("RPS-NoOrder")
    sym_err = _final("RPS-Symmetric")
    ctx.ablation = {"RPS-Full": full_err,
                    "RPS-NoOrder": no_err,
                    "RPS-Symmetric": sym_err}

    plot_figure6(full_err, no_err, sym_err)
    print(f"  RPS-Full mean = {full_err.mean():.3e}")
    print(f"  RPS-NoOrder mean = {no_err.mean():.3e}  "
          f"(d vs full = {cohens_d(no_err, full_err):.2f})")
    print(f"  RPS-Sym  mean = {sym_err.mean():.3e}  "
          f"(d vs full = {cohens_d(sym_err, full_err):.2f})")
    print(f"  -> fig_ablation.pdf saved. ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Figure 7 — diagnostic delay (consumes fig2_mtcd)
# ---------------------------------------------------------------------------

def figure_7(ctx: ExperimentContext) -> None:
    s = ctx.sizes
    print(f"\n[7/8] Figure 7: Diagnostic delay (MC={s.fig7_mc})...")
    t0 = time.time()
    if "Gradual drift" not in ctx.fig2_mtcd:
        # 没跑 figure 2 的情况下退化为单条占位
        T_minus_onset = s.fig2_T - s.fig2_T // 3
        ht_mtcd = np.array([T_minus_onset])
        rps_mtcd = np.array([T_minus_onset])
    else:
        T_minus_onset = s.fig2_T - ctx.fault_scenarios["Gradual drift"]['onset']
        ht_mtcd = ctx.fig2_mtcd["Gradual drift"].get("Hard-Threshold",
                                                       np.array([T_minus_onset]))
        rps_mtcd = ctx.fig2_mtcd["Gradual drift"].get("RPS-Full",
                                                        np.array([T_minus_onset]))
        if len(ht_mtcd) == 0 or np.all(np.isnan(ht_mtcd)):
            ht_mtcd = np.array([T_minus_onset])
        if len(rps_mtcd) == 0 or np.all(np.isnan(rps_mtcd)):
            rps_mtcd = np.array([T_minus_onset])

    plot_figure7(ht_mtcd, rps_mtcd)
    print(f"  Hard-Threshold MTCD: {np.nanmean(ht_mtcd):.1f} ± {np.nanstd(ht_mtcd):.1f}")
    print(f"  RPS-Full      MTCD: {np.nanmean(rps_mtcd):.1f} ± {np.nanstd(rps_mtcd):.1f}")
    # 同时打印故障检测率 / 误报率（论文 4.4.3）
    if "Gradual drift" in ctx.fig2_detection:
        for m_name in ("Hard-Threshold", "RPS-Full"):
            det = ctx.fig2_detection["Gradual drift"].get(m_name, np.array([np.nan]))
            fa = ctx.fig2_false_alarm["Gradual drift"].get(m_name, np.array([np.nan]))
            print(f"  {m_name:14s} detection={np.nanmean(det):.3f}, "
                  f"false-alarm={np.nanmean(fa):.3f}")
    print(f"  -> fig_diagnostic.pdf saved. ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Figure 8 — scale invariance
# ---------------------------------------------------------------------------

def figure_8(ctx: ExperimentContext) -> None:
    print("\n[8/8] Figure 8: Scale invariance...")
    t0 = time.time()
    curves: Dict[int, np.ndarray] = {}
    for N_val in ctx.sizes.fig8_Ns:
        W, adj, _ = build_graph(N_val, seed=0)
        A_, b_ = generate_least_squares_data(N_val, D_SYN, P_SYN, seed=0)
        cost = LeastSquaresCost(A_, b_)
        fault_cfg = {'onset': 200, 'agents': [4], 'type': 'drift',
                      'delta': 0.002 * np.ones(D_SYN), 'drift_cap': 40}
        cfg = ctx.base_cfg.replace(burn_in=150)
        err, _, _ = run_optimization(
            N=N_val, d=D_SYN, T=600, alpha=0.05,
            fault_config=fault_cfg, method="RPS-Full",
            W=W, adj=adj, cost=cost, cfg=cfg, seed=0,
        )
        curves[N_val] = err
    plot_figure8(curves)
    print(f"  -> fig_scaling.pdf saved. ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Tables and summary
# ---------------------------------------------------------------------------

def print_tables(ctx: ExperimentContext) -> None:
    if ctx.ablation:
        print("\n=== Table 1: Ablation summary ===")
        print(f"{'Variant':<18s}{'Final err (x10^-3)':>22s}")
        for name, vals in ctx.ablation.items():
            print(f"  {name:<16s}  {vals.mean()*1e3:>10.2f} ± {vals.std()*1e3:>5.2f}")

    if not ctx.fig2_finals:
        return

    print("\n=== Table 2: Comparative results ===")
    for sc in ctx.fault_scenarios:
        print(f"\n{sc}:")
        print(f"  {'Method':<22s}{'Final err (x10^-3)':>22s}{'Iter to 1e-3':>18s}")
        rps_vals = ctx.fig2_finals[sc].get("RPS-Full", np.array([]))
        for m in _FIG2_METHODS:
            f = ctx.fig2_finals[sc][m]
            it = ctx.fig2_iters[sc][m]
            print(f"  {m:<22s}  {f.mean()*1e3:>10.2f} ± {f.std()*1e3:>5.2f}"
                  f"  {it.mean():>10.0f} ± {it.std():>5.0f}")
        if len(rps_vals) >= 2:
            comparisons = []
            for m in _FIG2_METHODS:
                if m == "RPS-Full":
                    continue
                other = ctx.fig2_finals[sc][m]
                d = cohens_d(other, rps_vals)
                p = wilcoxon_pvalue(rps_vals, other)
                comparisons.append((m, p, d))
            ps_adj = holm_bonferroni([c[1] for c in comparisons])
            print("  Wilcoxon + Holm vs RPS-Full:")
            for (m, _, d), p_adj in zip(comparisons, ps_adj):
                print(f"    {m:<22s}  p_adj={p_adj:.4f}  Cohen's d={d:.2f}")


def save_results_json(ctx: ExperimentContext, path: str = "results.json") -> None:
    def _clean(v):
        """把 nan / inf 转成 None 让严格 JSON parser 也能读。"""
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, list):
            return [_clean(x) for x in v]
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        return v

    out: Dict[str, Any] = {"kappa_emp": _clean(ctx.kappa_emp),
                            "kappa_theo": _clean(ctx.kappa_theo)}
    if ctx.fig2_finals:
        diag_methods = ("RPS-Full", "RPS-Symmetric", "RPS-NoOrder", "Hard-Threshold")
        out["fig2_finals"] = {sc: {m: ctx.fig2_finals[sc][m].tolist()
                                    for m in _FIG2_METHODS}
                              for sc in ctx.fig2_finals}
        out["fig2_iters"] = {sc: {m: ctx.fig2_iters[sc][m].tolist()
                                   for m in _FIG2_METHODS}
                              for sc in ctx.fig2_iters}
        out["fig2_recovery"] = {sc: {m: ctx.fig2_recovery[sc][m].tolist()
                                      for m in _FIG2_METHODS}
                                 for sc in ctx.fig2_recovery}
        out["fig2_resilience"] = {sc: {m: ctx.fig2_resilience[sc][m].tolist()
                                        for m in _FIG2_METHODS}
                                   for sc in ctx.fig2_resilience}
        out["fig2_mtcd"] = {sc: {m: ctx.fig2_mtcd[sc][m].tolist()
                                  for m in diag_methods if m in ctx.fig2_mtcd[sc]}
                            for sc in ctx.fig2_mtcd}
        out["fig2_detection_rate"] = {
            sc: {m: ctx.fig2_detection[sc][m].tolist()
                 for m in diag_methods if m in ctx.fig2_detection[sc]}
            for sc in ctx.fig2_detection
        }
        out["fig2_false_alarm_rate"] = {
            sc: {m: ctx.fig2_false_alarm[sc][m].tolist()
                 for m in diag_methods if m in ctx.fig2_false_alarm[sc]}
            for sc in ctx.fig2_false_alarm
        }
    if ctx.ablation:
        out["ablation"] = {k: v.tolist() for k, v in ctx.ablation.items()}
    # 对所有数值字段做 NaN/Inf → None 转换，使 results.json 在严格 JSON parser
    # （如 Python 默认配置外的 jq、Java Jackson）下可读
    out = _clean(out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

_FIGURE_FNS = {
    1: figure_1, 2: figure_2, 3: figure_3, 4: figure_4,
    5: figure_5, 6: figure_6, 7: figure_7, 8: figure_8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="缩减 N、T、MC trials 以快速验证管线（约 8 分钟）")
    p.add_argument("--mc", type=int, default=None,
                   help="覆盖默认的 MC trial 数（论文 = 20，快速验证可设 3-5）")
    p.add_argument("--figures", type=str, default=None,
                   help="逗号分隔的 figure 编号，如 '1,2,6'；省略则跑全部")
    p.add_argument("--dataset", type=str, default="synthetic",
                   choices=["synthetic", "mnist", "ieee39"],
                   help="主基准数据集（默认 synthetic）。论文 Section 4.4.1 三档")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("Running experiments" + (" [QUICK MODE]" if args.quick else ""))
    print("=" * 60)

    sizes = _Sizes.quick() if args.quick else _Sizes.full()
    if args.mc is not None:
        if args.mc < 1:
            raise ValueError(f"--mc must be ≥ 1, got {args.mc}")
        sizes = sizes.with_mc(args.mc)
        print(f"  [override] MC trials = {args.mc}")
    base_cfg = RPSConfig()
    ctx = ExperimentContext(sizes=sizes, base_cfg=base_cfg, dataset=args.dataset)
    if args.dataset != "synthetic":
        print(f"  [dataset] {args.dataset} (replaces synthetic LS in Figures 2/6)")

    if args.figures:
        chosen = [int(x) for x in args.figures.split(",")]
    else:
        chosen = list(_FIGURE_FNS.keys())

    # Figure 7 依赖 fig2_mtcd；如果用户单独要 fig7 但没要 fig2，自动补上 fig2
    if 7 in chosen and 2 not in chosen:
        print("[note] Figure 7 depends on Figure 2 results; running Figure 2 first.")
        chosen = [2] + [c for c in chosen if c != 2]

    for fig_num in chosen:
        if fig_num not in _FIGURE_FNS:
            print(f"[warn] unknown figure number: {fig_num}, skipping")
            continue
        _FIGURE_FNS[fig_num](ctx)

    print_tables(ctx)
    save_results_json(ctx)
    print("Done.")


if __name__ == "__main__":
    main()
