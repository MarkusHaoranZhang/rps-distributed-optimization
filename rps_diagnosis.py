"""
RPS 诊断模块（向量化高性能版）
================================

实现论文 Section 4.1-4.3 的算法：

- 能量距离 D(R, Q0)
- 期望残差幅度 (Eq. 7)，故障幅度由残差范数代理（不读真 δ）
- 支持度得分 (Eq. 8) —— 工程上用 z-score 替代能量距离（见
  ``IMPLEMENTATION_NOTES.md``）
- 截断 PMF / softmax (Eq. 9)
- 左交集 / 左正交和 LOS (Eq. 10) —— 位掩码加速
- 邻居可靠性：JS 散度
- 有序概率变换 OPT (Eq. 11)
- 置信门控软折扣 (Eq. 12) —— 论文公式的连续化

PMF 数据结构详见 ``config.PMF``。
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Iterable, List, Optional, Sequence

import numpy as np

from config import PMF

# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------

def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """两个样本集之间的能量距离 (Cramér-style)。

    ``D(X, Y) = 2 E|X-Y| - E|X-X'| - E|Y-Y'|``。
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if X.shape[0] == 0 or Y.shape[0] == 0:  # type: ignore[index]
        return 0.0
    if X.shape[1] == 1 and Y.shape[1] == 1:  # type: ignore[index]
        x = X.ravel(); y = Y.ravel()
        EXY = np.abs(x[:, None] - y[None, :]).mean()
        EXX = np.abs(x[:, None] - x[None, :]).mean() if x.size > 1 else 0.0
        EYY = np.abs(y[:, None] - y[None, :]).mean() if y.size > 1 else 0.0
        return float(2.0 * EXY - EXX - EYY)
    from scipy.spatial.distance import cdist
    xy = cdist(X, Y).mean()
    xx = cdist(X, X).mean()
    yy = cdist(Y, Y).mean()
    return float(2.0 * xy - xx - yy)


def estimate_nominal_distribution(residual_norms_history: np.ndarray) -> np.ndarray:
    """从无故障 burn-in 期残差范数历史构造 Q0 样本（一维展开）。"""
    return np.asarray(residual_norms_history, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Fault propagation matrix
# ---------------------------------------------------------------------------

def build_fault_propagation_matrix(W: np.ndarray) -> np.ndarray:
    """残差对偏差的线性灵敏度：``r = (I - W) δ``，故 ``F = |I - W|``。"""
    return np.abs(np.eye(W.shape[0]) - W)  # type: ignore[index]


# ---------------------------------------------------------------------------
# Bitmask utilities
# ---------------------------------------------------------------------------

def _to_mask_array(masks_list: Sequence[int]) -> np.ndarray:
    """位掩码 ≤ 63 位时用 int64，否则用 object（容纳 Python 大整数）。"""
    try:
        return np.array(list(masks_list), dtype=np.int64)
    except OverflowError:
        # Python 整数超出 int64 范围（例如 N>63 时位掩码可能 ≥ 2^63）
        return np.array(list(masks_list), dtype=object)


def _bit_index_map(scope: Sequence[int]):
    a2b = {a: k for k, a in enumerate(scope)}
    b2a = list(scope)
    return a2b, b2a


def _mask_for(perm: Iterable[int], agent_to_bit: dict) -> int:
    m = 0
    for a in perm:
        m |= (1 << agent_to_bit[a])
    return m


# ---------------------------------------------------------------------------
# Hypothesis evaluation (Eqs. 7, 8, 9) - vectorized
# ---------------------------------------------------------------------------

def _enumerate_events(scope: Sequence[int], k_trunc: int) -> List[tuple]:
    """完整截断排列事件空间：r = 1..k 的所有排列。

    .. note::
       这是论文 Eq.(9) 的**参考实现**（PES_k(Θ_i) 完整枚举）。生产路径
       ``compute_pmf`` 走 ``_enumerate_events_topk`` 做先验稀疏化，对照公式
       逐字理解时使用此函数。当 ``top_agents == scope`` 时
       ``_enumerate_events_topk`` 退化为本函数（由
       ``test_enumerate_events_full_equals_topk_with_full_topagents`` 守门）。
    """
    events: List[tuple] = []
    for r in range(1, min(k_trunc, len(scope)) + 1):
        for perm in permutations(scope, r):
            events.append(tuple(perm))
    return events


def _enumerate_events_topk(scope: Sequence[int], k_trunc: int,
                            top_agents: Sequence[int]) -> List[tuple]:
    """两阶段枚举：r=1 用全 scope；r ≥ 2 只在 ``top_agents`` 里排列。

    这是 PMF 的「先验稀疏化」：贝叶斯诊断的质量集中在概率最高的少数智能体上，
    将 r ≥ 2 的排列空间限制到 ``top_agents`` 后，事件总数从
    O(|Θ|^k) 降到 O(|Θ| + |T|^k)。
    """
    events: List[tuple] = [(a,) for a in scope]
    base = list(top_agents)
    for r in range(2, min(k_trunc, len(base)) + 1):
        for perm in permutations(base, r):
            events.append(tuple(perm))
    return events


def expected_residual_at(self_idx: int, assumed_faulty_agents: Sequence[int],
                          F: np.ndarray, magnitude_proxy: np.ndarray) -> float:
    """E[r_i | A] 标量值 (Eq. 7)。

    .. note::
       这是论文 Eq.(7) 的**参考实现**，对照公式时使用。``compute_pmf``
       的高性能路径用向量化的 ``H @ contrib_in_scope`` 计算所有事件的
       期望残差，不调用此函数。

       参数 ``assumed_faulty_agents`` 是**假设的**故障 agent 索引列表（即
       排列事件 A 的元素），不是 ground-truth 故障配置——切勿与
       ``apply_fault_injection(fault_config, ...)`` 里的 ``fault_config: dict``
       混淆。
    """
    return float(sum(F[self_idx, j] * magnitude_proxy[j] for j in assumed_faulty_agents))


def compute_support_score(window: np.ndarray, expected_value: float,
                           Q0_samples: np.ndarray) -> float:
    """支持度 (Eq. 8): ``s_A = -log D(R - E[r|A], Q_0)``。

    .. note::
       这是论文 Eq.(8) 的**参考实现**，对照公式时使用。``compute_pmf``
       的高性能路径用 z-score 近似（见 IMPL §2），不调用此函数。
       保留是为了让读者能把代码与公式逐字对照。
    """
    diff = np.asarray(window, dtype=float).ravel() - float(expected_value)
    if diff.size == 0:
        return 0.0
    dist = energy_distance(diff.reshape(-1, 1), Q0_samples.reshape(-1, 1))
    return -math.log(abs(dist) + 1e-10)


def compute_pmf(self_idx: int, scope: Sequence[int], k_trunc: int,
                 residual_window: np.ndarray, F: np.ndarray,
                 magnitude_proxy: np.ndarray, Q0_samples: np.ndarray,
                 eta: float, *,
                 top_m: int = 16,
                 top_agents_k: int = 5,
                 proxy_global_weight: float = 0.5) -> PMF:
    """构造截断 PMF (Eq. 9)。

    在 small-fault regime 下，残差分布的均值漂移主导（论文假设）。支持度用
    z-score 计算 ``s_A = -|mean(R) - c_A| / sigma_0``：与能量距离在大样本下
    等价，但单步开销 O(E) 而非 O(E·s·M)。

    Parameters
    ----------
    self_idx : 本智能体全局索引
    scope    : 本智能体可诊断作用域（h-hop 邻域）
    k_trunc  : 截断阶数 k
    residual_window : 本智能体残差范数滑窗 (s,)
    F        : 故障传播矩阵 (N, N)
    magnitude_proxy : 故障幅度代理 (N,)
    Q0_samples : 名义分布样本（一维），仅用于估计 σ_0；不再做子采样。
    eta      : softmax 温度
    top_m    : PMF 保留的 top 事件数
    top_agents_k : 两阶段枚举里候选 agent 数
    proxy_global_weight : ``contrib + w · proxy_in_scope`` 的混合系数

    Notes
    -----
    H 矩阵构造里有一个 O(E·|A|) 的 Python 双层循环。``top_m=16`` 与
    ``k_trunc=3`` 让总赋值次数约 50，配合 ``diagnose_every=5`` 节流，实测
    不是热点；保留可读写法，没有向量化。
    """
    a2b, _ = _bit_index_map(scope)

    F_row = F[self_idx]
    contrib = F_row * magnitude_proxy
    scope_arr = np.asarray(scope, dtype=int)
    contrib_in_scope = contrib[scope_arr]

    # 全局 proxy 也加进每智能体局部信号：故障 agent 自身残差增量
    # 反映 (1-W_jj)·||δ_j||，不通过 F[i, :] 滤波也是有用信号。
    proxy_in_scope = magnitude_proxy[scope_arr]
    combined_signal = contrib_in_scope + proxy_global_weight * proxy_in_scope

    thr = max(combined_signal.max() * 0.05, 1e-12)
    sig_idx = np.where(combined_signal >= thr)[0]
    if len(sig_idx) == 0:
        sig_idx = np.array([int(np.argmax(combined_signal))])
    n_top = min(top_agents_k, len(sig_idx))
    sorted_sig = sig_idx[np.argsort(-combined_signal[sig_idx])][:n_top]
    top_agents = [int(scope_arr[i]) for i in sorted_sig]

    events = _enumerate_events_topk(scope, k_trunc, top_agents)
    if not events:
        return PMF.empty()

    # 每事件的 c_A：用稀疏指示矩阵 H 一次性算
    n_scope = len(scope)
    a2b_arr = np.full(F.shape[0], -1, dtype=np.int64)
    for kk, a in enumerate(scope):
        a2b_arr[a] = kk
    H = np.zeros((len(events), n_scope))
    for ee, A in enumerate(events):
        for a in A:
            H[ee, a2b_arr[a]] = 1.0
    c_arr = H @ contrib_in_scope                          # (E,)

    X = np.asarray(residual_window, dtype=float).ravel()
    Y = np.asarray(Q0_samples, dtype=float).ravel()
    if X.size == 0 or Y.size == 0:
        m = 1.0 / len(events)
        masks = _to_mask_array([_mask_for(A, a2b) for A in events])
        return PMF(events=tuple(events), mass=np.full(len(events), m), masks=masks)

    mean_R = float(X.mean())
    sigma_0 = float(Y.std()) + 1e-6
    scores = -np.abs(mean_R - c_arr) / sigma_0           # (E,)
    scores -= scores.max()
    weights = np.exp(eta * scores)
    s = weights.sum()
    if s <= 0 or not np.isfinite(s):
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / s

    if len(events) > top_m:
        keep = np.argpartition(weights, -top_m)[-top_m:]
        keep = keep[np.argsort(-weights[keep])]
        events = [events[i] for i in keep]
        mass = weights[keep]
        mass = mass / mass.sum()
    else:
        mass = weights

    masks = _to_mask_array([_mask_for(A, a2b) for A in events])
    return PMF(events=tuple(events), mass=mass, masks=masks)


# ---------------------------------------------------------------------------
# JS divergence and singleton vector
# ---------------------------------------------------------------------------

def pmf_to_singleton_vector(pmf: PMF, scope: Sequence[int]) -> np.ndarray:
    """边缘化 PMF 到各 singleton 概率向量。

    一个事件 ``A = (a_1, ..., a_r)`` 把它的质量等分给所有成员，再归一。
    """
    if pmf.is_empty:
        return np.zeros(len(scope))
    a2b = {a: k for k, a in enumerate(scope)}
    p = np.zeros(len(scope))
    for k_evt, A in enumerate(pmf.events):
        m = pmf.mass[k_evt]
        share = m / len(A)
        for a in A:
            bb = a2b.get(a)
            if bb is not None:
                p[bb] += share
    s = p.sum()
    if s > 0:
        p /= s
    return p


def js_divergence(pmf1: PMF, pmf2: PMF, scope: Sequence[int], *,
                   sing1: Optional[np.ndarray] = None,
                   sing2: Optional[np.ndarray] = None) -> float:
    """两个 PMF 的 Jensen-Shannon 散度（边缘 singleton 向量上的）。"""
    p = sing1 if sing1 is not None else pmf_to_singleton_vector(pmf1, scope)
    q = sing2 if sing2 is not None else pmf_to_singleton_vector(pmf2, scope)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    # Defensive normalization：sing1/sing2 由调用方传入时不一定归一化
    # （例如 _order_by_js 缓存的中间结果）；clip 之后也可能不再 sum 到 1。
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


# ---------------------------------------------------------------------------
# Combination rules: LOS (Eq. 10) and DS (set intersection)
# ---------------------------------------------------------------------------

def _uniform_singleton(scope: Sequence[int]) -> PMF:
    a2b, _ = _bit_index_map(scope)
    events = tuple((a,) for a in scope)
    n = len(events)
    if n == 0:
        return PMF.empty()
    mass = np.full(n, 1.0 / n)
    masks = _to_mask_array([1 << a2b[a] for a in scope])
    return PMF(events=events, mass=mass, masks=masks)


def left_intersection(A: tuple, B: tuple) -> tuple:
    """A∩B (left)：保留 A 的顺序，剔除不在 B 中的元素。"""
    Bset = set(B)
    return tuple(x for x in A if x in Bset)


def _combine_with_intersection(pmf_a: PMF, pmf_b: PMF, scope: Sequence[int],
                                top_m: int, ordered: bool) -> PMF:
    """LOS（ordered=True）与 DS（ordered=False）的统一实现。

    两者唯一区别在交集元组的构造：
        LOS: 保留 ``pmf_a`` 中事件 A 的顺序
        DS : 集合交集后按全局升序排列
    其它步骤（外积质量、冲突归一化、top_m 截断）相同。
    """
    if pmf_a.is_empty or pmf_b.is_empty:
        return pmf_a if not pmf_a.is_empty else pmf_b

    a2b, _ = _bit_index_map(scope)
    M = pmf_a.mass[:, None] * pmf_b.mass[None, :]
    inter_masks = pmf_a.masks[:, None] & pmf_b.masks[None, :]
    nonempty = inter_masks != 0
    K = float(M[~nonempty].sum())
    if K >= 1.0 - 1e-12:
        return _uniform_singleton(scope)

    new: dict = {}
    nz_a, nz_b = np.where(nonempty)
    inter_arr = inter_masks[nz_a, nz_b]
    M_arr = M[nz_a, nz_b]
    cache: dict = {}
    for k_idx in range(nz_a.size):
        ia = int(nz_a[k_idx])
        mask = int(inter_arr[k_idx])
        cache_key = (ia, mask) if ordered else mask
        C = cache.get(cache_key)
        if C is None:
            if ordered:
                A_evt = pmf_a.events[ia]
                C = tuple(x for x in A_evt if (mask >> a2b[x]) & 1)
            else:
                members = [x for x in scope if (mask >> a2b[x]) & 1]
                C = tuple(sorted(members))
            cache[cache_key] = C
        if C:
            new[C] = new.get(C, 0.0) + float(M_arr[k_idx])

    if not new:
        return _uniform_singleton(scope)

    factor = 1.0 / (1.0 - K)
    keys = list(new.keys())
    vals = np.array([new[k] for k in keys]) * factor

    if len(keys) > top_m:
        keep = np.argpartition(vals, -top_m)[-top_m:]
        keep = keep[np.argsort(-vals[keep])]
        keys = [keys[i] for i in keep]
        vals = vals[keep]
    vals = vals / vals.sum()

    masks = _to_mask_array([_mask_for(C, a2b) for C in keys])
    return PMF(events=tuple(keys), mass=vals, masks=masks)


def left_orthogonal_sum(pmf_a: PMF, pmf_b: PMF, scope: Sequence[int],
                         top_m: int = 16) -> PMF:
    """论文 Eq.(10) LOS：``pmf_a`` 是更可靠源，结果保留 A 的事件顺序。"""
    return _combine_with_intersection(pmf_a, pmf_b, scope, top_m, ordered=True)


def dempster_shafer_combination(pmf_a: PMF, pmf_b: PMF, scope: Sequence[int],
                                 top_m: int = 16) -> PMF:
    """对称的 Dempster-Shafer 组合（用于 ``RPS-NoOrder`` 消融）。"""
    return _combine_with_intersection(pmf_a, pmf_b, scope, top_m, ordered=False)


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

def _order_by_js(pmf_self: PMF, neighbor_pmfs: Sequence[PMF],
                 scope: Sequence[int]) -> List[int]:
    sing_self = pmf_to_singleton_vector(pmf_self, scope)
    sings = [pmf_to_singleton_vector(p, scope) for p in neighbor_pmfs]
    js = [js_divergence(p, pmf_self, scope, sing1=s, sing2=sing_self)
          for p, s in zip(neighbor_pmfs, sings)]
    return list(np.argsort(js))


def directional_fusion(pmf_self: PMF, neighbor_pmfs: Sequence[PMF],
                        scope: Sequence[int], top_m: int = 16) -> PMF:
    """方向性融合（论文 Eq.10 + Section 4.2）：以 ``pmf_self`` 为锚，按 JS 散
    度升序依次做 LOS。"""
    if not neighbor_pmfs:
        return pmf_self
    fused = pmf_self
    for idx in _order_by_js(pmf_self, neighbor_pmfs, scope):
        fused = left_orthogonal_sum(fused, neighbor_pmfs[idx], scope, top_m=top_m)
    return fused


def _collapse_to_unordered(pmf: PMF, scope: Sequence[int]) -> PMF:
    """把 PMF 的有序排列事件合并为无序集合事件。

    论文 Section 4.4.2 RPS-NoOrder 定义："collapsing ordered tuples to
    unordered sets before fusion"。``(a, b)`` 与 ``(b, a)`` 合并成单一的
    无序键 ``(min, max)``，质量相加。
    """
    if pmf.is_empty:
        return pmf
    new_mass: dict = {}
    for A, m in zip(pmf.events, pmf.mass):
        key = tuple(sorted(A))
        new_mass[key] = new_mass.get(key, 0.0) + float(m)
    keys = list(new_mass.keys())
    vals = np.array([new_mass[k] for k in keys])
    s = vals.sum()
    if s > 0:
        vals = vals / s
    a2b, _ = _bit_index_map(scope)
    masks = _to_mask_array([_mask_for(k, a2b) for k in keys])
    return PMF(events=tuple(keys), mass=vals, masks=masks)


def symmetric_fusion(pmf_self: PMF, neighbor_pmfs: Sequence[PMF],
                      scope: Sequence[int], top_m: int = 16) -> PMF:
    """RPS-Symmetric 消融（论文 Section 4.4.2 字面定义）。

    论文文字："directional LOS fusion is replaced by **symmetric
    Dempster-Shafer combination**"。整条融合链都用 DS，**不使用 LOS**——
    self 不作为 ordering anchor。DS 是对称运算，源的顺序不影响结果。
    """
    if not neighbor_pmfs:
        return pmf_self
    fused = pmf_self
    for nb_pmf in neighbor_pmfs:
        if nb_pmf.is_empty:
            continue
        fused = dempster_shafer_combination(fused, nb_pmf, scope, top_m=top_m)
    return fused


def noorder_fusion(pmf_self: PMF, neighbor_pmfs: Sequence[PMF],
                    scope: Sequence[int], top_m: int = 16) -> PMF:
    """RPS-NoOrder 消融（论文 Section 4.4.2 字面定义）。

    论文文字："the permutation structure is removed by **collapsing
    ordered tuples to unordered sets before fusion**"。

    实现：所有 PMF 在融合前先 collapse 为无序集合事件，再按 DS 链合并。
    输出的事件键是排序后的 tuple，确保 OPT 端无法从顺序里"恢复"任何信息。
    """
    self_unord = _collapse_to_unordered(pmf_self, scope)
    if not neighbor_pmfs:
        return self_unord
    fused = self_unord
    for nb in neighbor_pmfs:
        if nb.is_empty:
            continue
        nb_unord = _collapse_to_unordered(nb, scope)
        if nb_unord.is_empty:
            continue
        fused = dempster_shafer_combination(fused, nb_unord, scope, top_m=top_m)
    return fused


# ---------------------------------------------------------------------------
# PMF projection / flattening
# ---------------------------------------------------------------------------

def project_pmf(pmf: PMF, target_scope: Sequence[int]) -> PMF:
    """把 PMF 投影到子作用域：保留事件中位于 ``target_scope`` 的元素，归一化。"""
    if pmf.is_empty:
        return PMF.empty()
    target_set = set(target_scope)
    new: dict = {}
    for A, m in zip(pmf.events, pmf.mass):
        C = tuple(x for x in A if x in target_set)
        if C:
            new[C] = new.get(C, 0.0) + float(m)
    if not new:
        return PMF.empty()
    keys = list(new.keys())
    vals = np.array([new[k] for k in keys])
    vals = vals / vals.sum()
    a2b, _ = _bit_index_map(target_scope)
    masks = _to_mask_array([_mask_for(k, a2b) for k in keys])
    return PMF(events=tuple(keys), mass=vals, masks=masks)


# ---------------------------------------------------------------------------
# OPT, entropy, gating, tau calibration
# ---------------------------------------------------------------------------

def ordered_probability_transformation(pmf: PMF, scope: Sequence[int]) -> dict:
    """OPT (论文 Eq.11)：终端元素不分配，前缀均分。"""
    p = {a: 0.0 for a in scope}
    for A, m in zip(pmf.events, pmf.mass):
        if len(A) == 1:
            p[A[0]] += float(m)
        else:
            share = float(m) / (len(A) - 1)
            for a in A[:-1]:
                p[a] += share
    return p


def pmf_entropy(pmf: PMF) -> float:
    if pmf.is_empty:
        return 0.0
    m = np.clip(pmf.mass, 1e-12, 1.0)
    return float(-(m * np.log(m)).sum())


def confidence_gated_discount(opt_probs: dict, entropy: float, tau: float, *,
                               gain: float = 4.0,
                               base_keep: float = 1.0) -> dict:
    """连续软折扣 ``γ_ij = base_keep · exp(-effective_gain · P_OPT(j))``。

    论文 Eq.(12) 是分段函数（高熵→1，低熵→ ``P / max P``）。这里改为单调连续：

    - ``P_OPT(j) = 0`` (无嫌疑) → γ ≈ ``base_keep``（≈1，不折扣）
    - ``P_OPT(j) → 1`` (确认故障) → γ → 0（强力压制）
    - 高熵（H ≥ τ，证据弱）时 ``effective_gain`` 减半，避免不确定时过度反应。

    参数说明见 ``IMPLEMENTATION_NOTES.md``。
    """
    eff_gain = 0.5 * gain if entropy >= tau else gain
    return {a: float(base_keep * np.exp(-eff_gain * p))
            for a, p in opt_probs.items()}


def calibrate_tau(entropy_history: Sequence[float], quantile: float = 0.95) -> float:
    """把 τ 校准为无故障期 PMF 熵分布的指定分位（论文 Section 4.3）。"""
    if len(entropy_history) == 0:
        return float('inf')
    return float(np.quantile(np.asarray(entropy_history, dtype=float), quantile))
