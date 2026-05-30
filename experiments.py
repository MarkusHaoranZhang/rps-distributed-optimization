"""
实验运行器
============

完整闭环：诊断 → 软折扣 γ → 鲁棒梯度跟踪。

主入口是 ``run_optimization``，按方法名分派到下面三类内部步函数：

  ``_step_hard_threshold`` : 卡方残差检测 + 硬排除（论文 §1 critique 对象）
  ``_step_uniform_discount``: 全网均匀折扣
  ``_step_rps``            : RPS-Full / RPS-Symmetric / RPS-NoOrder

Byzantine-Resilient 不通过 γ 矩阵，而是在 X 步进前直接对 X 做坐标中位过滤；
Ideal 与 Byzantine 都不需要专用 step 函数。

各方法的状态（HT 检测器、RPS 缓存等）封装在 ``_RunState`` 里，主循环只做调度。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from baselines import (
    HardThresholdDetector,
    coordinate_wise_median_aggregate,
    uniform_discount_gamma,
)
from config import KNOWN_METHODS, PMF, RPSConfig, is_rps_method, validate_fault_config
from distributed_optimization import (
    apply_fault_injection,
    compute_local_gradients,
    compute_residuals,
    gradient_tracking_step,
    hop_neighborhood,
)
from rps_diagnosis import (
    build_fault_propagation_matrix,
    compute_pmf,
    confidence_gated_discount,
    directional_fusion,
    noorder_fusion,
    ordered_probability_transformation,
    pmf_entropy,
    pmf_to_singleton_vector,
    project_pmf,
    symmetric_fusion,
)

__all__ = [
    "run_optimization",
    "mean_time_to_correct_diagnosis",
    "recovery_time",
    "resilience_metric",
    "detection_and_false_alarm_rates",
]


# ---------------------------------------------------------------------------
# Per-agent RPS state
# ---------------------------------------------------------------------------

@dataclass
class _AgentRPSState:
    scope: List[int]
    window: deque
    pmf: Optional[PMF] = None


# ---------------------------------------------------------------------------
# Aggregate run state (mutable container shared by step functions)
# ---------------------------------------------------------------------------

@dataclass
class _RunState:
    N: int
    d: int
    T: int
    rng: np.random.RandomState
    F: np.ndarray
    rps_states: Optional[List[_AgentRPSState]] = None
    ht_detector: Optional[HardThresholdDetector] = None
    tau: float = float('inf')
    last_gamma_mat: Optional[np.ndarray] = None
    burnin_entropies: List[float] = field(default_factory=list)
    diag_log: Dict[str, list] = field(default_factory=lambda: {
        "true_fault_top1_prob": [],
        "entropy_history": [],
        "gamma_history": [],
    })


# ---------------------------------------------------------------------------
# Step functions per method family
# ---------------------------------------------------------------------------

def _step_hard_threshold(t: int, st: _RunState, residuals_norm: np.ndarray,
                          fault_config: dict, cfg: RPSConfig) -> Optional[np.ndarray]:
    """卡方残差检测器：每步重新基于 ``res_norm[t]`` 计算 γ。

    .. note::
       检测器没有时间滤波：``residual_norms`` 在共识动态下天然震荡，
       χ² 统计量 ``r²/σ²`` 也随之震荡，导致 γ 在某些步把故障智能体
       屏蔽、另一些步又放开。这正是论文 Section 1 描述的"oscillates
       between inclusion and exclusion" 现象。它会让 ``Iter to 1e-3``
       指标在 trial 间方差很大（论文的反面教材）。
    """
    assert st.ht_detector is not None, "_RunState.ht_detector must be initialized for HT method"
    if t == cfg.burn_in - 1:
        st.ht_detector.calibrate(residuals_norm[:cfg.burn_in])
    if t < cfg.burn_in:
        return None
    gamma_mat = st.ht_detector.gamma_matrix(residuals_norm[t])
    # 把 HT 的检测结果也记录到 diag_log 里，使 Figure 7 的 MTCD 在
    # HT 与 RPS 之间可比（论文 4.4.3 的 MTCD 定义：故障 agent 被
    # 识别为最可疑的概率 ≥ 0.95）。
    #
    # HT 是二值检测器，没有连续概率：γ_{i, tgt} 的列均值要么 < 0.5（被
    # 多数邻居判为故障）要么 ≥ 0.5。我们把"列均值 < 0.5"映射为 1.0，
    # 否则 0.0。这意味着 MTCD 阈值 0.95 在 HT 上等价于"任一时刻多数
    # 邻居判它为故障"——只要这个事件发生过一次，MTCD 就停在那一步。
    #
    # 在 Drift 场景下 HT 的实测 MTCD 接近 T - onset 的最大值，意味着
    # χ² 阈值在整个故障期都没有跌破——这恰好是论文 §1 critique 的
    # "threshold-based detector ... oscillates / fails"现象的数值
    # 证据，不是 RPS 框架的局限。
    if cfg.record_diagnosis and t >= fault_config['onset']:
        tgt = fault_config['agents'][0] if fault_config['agents'] else None
        if tgt is not None:
            ht_top1 = 1.0 if (gamma_mat[:, tgt].mean() < 0.5) else 0.0
        else:
            ht_top1 = 0.0
        st.diag_log["true_fault_top1_prob"].append(ht_top1)
    return gamma_mat


def _step_uniform_discount(t: int, st: _RunState, fault_config: dict,
                            cfg: RPSConfig) -> Optional[np.ndarray]:
    if t >= fault_config['onset']:
        return uniform_discount_gamma(st.N, factor=cfg.uniform_factor)
    return None


def _compute_magnitude_proxy(residuals_norm: np.ndarray, t: int, cfg: RPSConfig,
                              N: int) -> tuple:
    """从残差范数历史构造故障幅度代理。

    .. note::
       当 ``t < cfg.burn_in`` 时（burn-in 期），baseline 用零向量、std_base 用 1
       作为占位。此时生成的 PMF 仅用于收集熵供 τ 校准（不参与 γ 计算），所以
       baseline 不准确不影响最终结果。这是工程上让 burn-in 期 PMF 也能跑出来
       的占位实现，不是论文公式。

    Returns
    -------
    (magnitude_proxy, baseline, Q0_samples)
        magnitude_proxy : (N,) 故障幅度代理
        baseline        : (N,) 无故障基线均值（burn-in 期为零向量占位）
        Q0_samples      : (M,) 基线波动样本
    """
    win_lo = max(0, t - cfg.window_len + 1)
    window = residuals_norm[win_lo: t + 1]
    mean_now = window.mean(axis=0)
    std_now = window.std(axis=0)
    if t >= cfg.burn_in:
        burn = residuals_norm[:cfg.burn_in]
        baseline = burn.mean(axis=0)
        std_base = burn.std(axis=0)
        Q0_samples = (burn - baseline[None, :]).ravel()
    else:
        baseline = np.zeros(N)
        std_base = np.ones(N)
        Q0_samples = residuals_norm[:t + 1].ravel()
    proxy_mean = np.maximum(mean_now - baseline, 0.0)
    proxy_std = np.maximum(std_now - std_base, 0.0)
    magnitude_proxy = np.maximum(proxy_mean, cfg.proxy_std_weight * proxy_std)
    return magnitude_proxy, baseline, Q0_samples


def _generate_local_pmfs(st: _RunState, residuals_norm: np.ndarray, t: int,
                          F: np.ndarray, magnitude_proxy: np.ndarray,
                          Q0_samples: np.ndarray, baseline: np.ndarray,
                          cfg: RPSConfig) -> None:
    """每个智能体根据自己作用域内的残差窗口生成本地 PMF。"""
    assert st.rps_states is not None
    for i in range(st.N):
        state = st.rps_states[i]
        window_arr = np.array(state.window) - baseline[i]
        state.pmf = compute_pmf(
            i, state.scope, cfg.k_trunc, window_arr, F,
            magnitude_proxy, Q0_samples, cfg.eta,
            top_m=cfg.top_m,
            top_agents_k=cfg.top_agents_k,
            proxy_global_weight=cfg.proxy_global_weight,
        )


def _fuse_for_agent(method: str, state: _AgentRPSState,
                     neighbor_pmfs: List[PMF], cfg: RPSConfig) -> PMF:
    assert state.pmf is not None, "agent's local PMF should be computed before fusion"
    if method == "RPS-Full":
        return directional_fusion(state.pmf, neighbor_pmfs, state.scope,
                                   top_m=cfg.top_m)
    if method == "RPS-Symmetric":
        return symmetric_fusion(state.pmf, neighbor_pmfs, state.scope,
                                 top_m=cfg.top_m)
    if method == "RPS-NoOrder":
        return noorder_fusion(state.pmf, neighbor_pmfs, state.scope,
                               top_m=cfg.top_m)
    raise ValueError(f"Unknown RPS method: {method}")


def _record_agent_diagnosis(st: _RunState, fused: PMF, state: _AgentRPSState,
                              entropy: float, fault_config: dict) -> None:
    sing = pmf_to_singleton_vector(fused, state.scope)
    st.diag_log["entropy_history"].append(float(entropy))
    tgt = fault_config['agents'][0] if fault_config['agents'] else None
    if tgt is not None and tgt in state.scope:
        idx = state.scope.index(tgt)
        st.diag_log["true_fault_top1_prob"].append(float(sing[idx]))
    else:
        st.diag_log["true_fault_top1_prob"].append(0.0)


def _step_rps(t: int, st: _RunState, residuals_norm: np.ndarray,
               method: str, fault_config: dict, adj: np.ndarray,
               cfg: RPSConfig) -> Optional[np.ndarray]:
    """RPS-Full / RPS-Symmetric / RPS-NoOrder 共用的诊断步。

    τ 校准（论文 Section 4.3）的真实实现：
      - 在 burn-in 期 (t = window_len .. burn_in-1) 用当前残差窗口生成 PMF
        并收集每个智能体的熵；
      - 在 t == burn_in - 1 时把所有智能体在 burn-in 期内的熵汇总，按
        ``cfg.tau_quantile`` 取分位作为 τ。
    这样 ``cfg.tau_quantile`` 真实影响 τ 取值，Figure 3 的 τ 敏感性曲线才有效。
    """
    assert st.rps_states is not None, "_RunState.rps_states must be initialized for RPS methods"
    # 上面的 assert 是给 mypy 的类型缩窄；进入此函数的前提是
    # ``is_rps_method(method)``，``run_optimization`` 已经为 RPS 分支
    # 初始化了 rps_states，所以运行时一定非空。
    # 更新滑窗
    for i in range(st.N):
        st.rps_states[i].window.append(residuals_norm[t, i])

    if t < cfg.window_len:
        return None

    magnitude_proxy, baseline, Q0_samples = _compute_magnitude_proxy(
        residuals_norm, t, cfg, st.N)

    # burn-in 期：在故障未到时就生成 PMF 并收集熵，用于 τ 校准
    in_burnin = (t < fault_config['onset']) and (t < cfg.burn_in)
    if in_burnin:
        _generate_local_pmfs(st, residuals_norm, t, st.F, magnitude_proxy,
                              Q0_samples, baseline, cfg)
        # 收集每个智能体的熵
        for state in st.rps_states:
            if state.pmf is not None and not state.pmf.is_empty:
                st.burnin_entropies.append(pmf_entropy(state.pmf))
        return None

    # τ 校准（一次，发生在 burn-in 末或刚进故障期时）
    if st.tau == float('inf'):
        if cfg.tau is not None:
            # 显式给出 τ：直接用（适合 Figure 3 的 τ 敏感性扫描）
            st.tau = float(cfg.tau)
        elif st.burnin_entropies:
            st.tau = float(np.quantile(np.asarray(st.burnin_entropies),
                                        cfg.tau_quantile))
        else:
            st.tau = math.log(cfg.top_m)

    # burn-in 之后但还没到故障期（onset > burn_in 的情形）
    if t < fault_config['onset']:
        return None

    # 节流：每 cfg.diagnose_every 步触发一次诊断重计算
    diag_step = ((t - fault_config['onset']) % max(1, cfg.diagnose_every) == 0)
    if (not diag_step) and st.last_gamma_mat is not None:
        return st.last_gamma_mat

    # 1) 每智能体生成本地 PMF
    _generate_local_pmfs(st, residuals_norm, t, st.F, magnitude_proxy,
                          Q0_samples, baseline, cfg)

    # 2) 邻居 PMF 融合 + γ 计算
    gamma_mat = np.ones((st.N, st.N))
    for i in range(st.N):
        state = st.rps_states[i]
        neighbor_pmfs: List[PMF] = []
        for j in np.where(adj[i] > 0)[0]:
            nb_pmf = st.rps_states[j].pmf
            if nb_pmf is None or nb_pmf.is_empty:
                continue
            projected = project_pmf(nb_pmf, state.scope)
            if not projected.is_empty:
                neighbor_pmfs.append(projected)

        fused = _fuse_for_agent(method, state, neighbor_pmfs, cfg)

        opt = ordered_probability_transformation(fused, state.scope)
        H = pmf_entropy(fused)
        gammas = confidence_gated_discount(opt, H, st.tau, gain=cfg.gain)
        for j in np.where(adj[i] > 0)[0]:
            gamma_mat[i, j] = gammas.get(j, 1.0)
        gamma_mat[i, i] = 1.0

        if cfg.record_diagnosis and i == cfg.record_agent_idx:
            _record_agent_diagnosis(st, fused, state, H, fault_config)

    st.last_gamma_mat = gamma_mat
    return gamma_mat


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_optimization(N: int, d: int, T: int, alpha: float,
                      fault_config: dict, method: str,
                      W: np.ndarray, adj: np.ndarray, *,
                      cost,
                      cfg: Optional[RPSConfig] = None,
                      seed: int = 0):
    """运行一次完整闭环优化。

    Parameters
    ----------
    N, d, T, alpha : 优化主循环参数（智能体数 / 决策维度 / 步数 / 步长）
    fault_config   : 故障配置 dict（schema 见 ``config.FaultConfig``）
    method         : ``KNOWN_METHODS`` 之一
    W, adj         : 共识权重矩阵和邻接矩阵
    cost           : 实现 ``grad_fns()`` / ``global_optimum()`` /
                    ``problem_dim()`` 的代价对象
    cfg            : RPSConfig 实例；为 None 时取默认值
    seed           : 随机种子

    Returns
    -------
    (errors, residuals_norm, diag_log)
        errors          : (T,) 平均相对误差
        residuals_norm  : (T, N) 残差范数历史
        diag_log        : dict, 含 MTCD / γ_history / 诊断概率历史
    """
    if method not in KNOWN_METHODS:
        raise ValueError(f"Unknown method '{method}'. Known: {KNOWN_METHODS}")
    validate_fault_config(fault_config, d=d)

    if cfg is None:
        cfg = RPSConfig()

    rng = np.random.RandomState(seed)
    grad_fns = cost.grad_fns()
    x_opt = cost.global_optimum()

    X = rng.randn(N, d) * 0.1
    grad_old = compute_local_gradients(X, grad_fns,
                                        np.zeros(N, dtype=bool),
                                        np.zeros((N, d)))
    Y = grad_old.copy()

    errors = np.zeros(T)
    residuals_norm = np.zeros((T, N))

    F = build_fault_propagation_matrix(W)
    use_rps = is_rps_method(method)
    rps_states = (
        [_AgentRPSState(scope=hop_neighborhood(adj, i, cfg.h_hop),
                          window=deque(maxlen=cfg.window_len))
         for i in range(N)] if use_rps else None
    )
    ht_detector = (HardThresholdDetector(d=d, confidence=cfg.chi2_confidence)
                   if method == "Hard-Threshold" else None)
    st = _RunState(N=N, d=d, T=T, rng=rng, F=F,
                   rps_states=rps_states, ht_detector=ht_detector)

    norm_opt = np.linalg.norm(x_opt) + 1e-12

    for t in range(T):
        faulty_mask, delta = apply_fault_injection(t, fault_config, N, d, rng)

        # 1) 梯度
        grad_new = compute_local_gradients(X, grad_fns, faulty_mask, delta)

        # 2) 残差
        res_norm, _ = compute_residuals(grad_new, W)
        residuals_norm[t] = res_norm

        # 3) γ 矩阵（按方法分派）
        gamma_mat: Optional[np.ndarray] = None
        if method == "Hard-Threshold":
            gamma_mat = _step_hard_threshold(t, st, residuals_norm,
                                              fault_config, cfg)
        elif method == "Uniform-Discount":
            gamma_mat = _step_uniform_discount(t, st, fault_config, cfg)
        elif use_rps:
            gamma_mat = _step_rps(t, st, residuals_norm, method,
                                   fault_config, adj, cfg)

        # 4) Byzantine 例外：不走 γ，直接坐标中位过滤 X
        if method == "Byzantine-Resilient" and t >= fault_config['onset']:
            X = coordinate_wise_median_aggregate(X, adj)

        # 5) 记录 γ 历史（用于 MTCD）
        if cfg.record_diagnosis and t >= fault_config['onset'] and gamma_mat is not None:
            st.diag_log["gamma_history"].append(gamma_mat.copy())

        # 6) 一步梯度跟踪
        X, Y = gradient_tracking_step(X, Y, grad_old, grad_new, W, alpha,
                                        gamma=gamma_mat)
        grad_old = grad_new

        errors[t] = float(np.mean(np.linalg.norm(X - x_opt[None, :], axis=1)) / norm_opt)

    return errors, residuals_norm, st.diag_log


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def mean_time_to_correct_diagnosis(diag_log: dict, fault_onset: int,
                                     prob_threshold: float = 0.95) -> float:
    """从故障发生到真故障 top-1 概率超过阈值所需的迭代数。"""
    probs = diag_log.get("true_fault_top1_prob", [])
    if not probs:
        return float('nan')
    for k, p in enumerate(probs):
        if p >= prob_threshold:
            return k
    return len(probs)


def recovery_time(errors: np.ndarray, fault_onset: int,
                   base_factor: float = 1.10) -> float:
    """恢复到 1.1 × 故障前误差所需迭代数。"""
    if fault_onset <= 0 or fault_onset >= len(errors):
        return float('nan')
    pre = errors[max(0, fault_onset - 50): fault_onset]
    if len(pre) == 0:
        return float('nan')
    target = base_factor * pre.mean()
    for k in range(fault_onset, len(errors)):
        if errors[k] <= target:
            return k - fault_onset
    return len(errors) - fault_onset


def resilience_metric(errors_method: np.ndarray,
                       errors_ideal: np.ndarray) -> float:
    """方法相对 ideal 的累积超出量；越小越韧性。"""
    L = min(len(errors_method), len(errors_ideal))
    return float(np.sum(np.maximum(errors_method[:L] - errors_ideal[:L], 0.0)))


def detection_and_false_alarm_rates(gamma_history: list,
                                      faulty_agents: list, N: int,
                                      adj: Optional[np.ndarray] = None,
                                      gamma_threshold: float = 0.5) -> tuple:
    """评估诊断器的故障检测率与误报率（论文 Section 4.4.3）。

    判决粒度是"邻居均值"：对每个 j，看 j 的邻居们在该步给出的 γ_{*, j}
    的均值，均值 < ``gamma_threshold`` 时判 "j 在该步被诊断为故障"。

    - detection rate = TP / (TP + FN)
        TP: faulty agent j 在某步被它的邻居判为故障；
        FN: faulty agent 没被它的邻居判为故障。
    - false alarm rate = FP / (FP + TN)
        FP: healthy agent j 被某邻居错判为故障；
        TN: healthy agent 没被任何邻居判为故障。

    Parameters
    ----------
    gamma_history : list of (N, N) ``γ`` 矩阵；只取故障期之后的步。
    faulty_agents : 真实故障 agent 全局索引列表（评估时使用，不进入诊断）。
    N : 智能体总数
    adj : 邻接矩阵；若给出则只对 j 的邻居行求均值，否则对除 j 外所有行求均值。
    gamma_threshold : 邻居 γ 均值低于此值认为 "j 被判为故障"。

    Returns
    -------
    (detection_rate, false_alarm_rate) : 两个 [0, 1] 区间内的浮点
        如果 ``gamma_history`` 为空或 ``faulty_agents`` 为空，返回 (nan, nan)。
    """
    if not gamma_history or not faulty_agents:
        return float('nan'), float('nan')

    faulty_set = set(faulty_agents)

    tp = fn = fp = tn = 0
    for G in gamma_history:
        for j in range(N):
            if adj is not None:
                neighbors = np.where(adj[:, j] > 0)[0]
                if len(neighbors) == 0:
                    continue
                # j 被认为故障 = 它的邻居们的平均 γ 低于阈值
                judged_faulty = float(G[neighbors, j].mean()) < gamma_threshold
            else:
                mask = np.ones(N, dtype=bool); mask[j] = False
                judged_faulty = float(G[mask, j].mean()) < gamma_threshold

            if j in faulty_set:
                if judged_faulty:
                    tp += 1
                else:
                    fn += 1
            else:
                if judged_faulty:
                    fp += 1
                else:
                    tn += 1

    detection = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    return detection, false_alarm
