"""
分布式优化核心模块
====================

实现论文方法论中的：
- 随机几何图 + Metropolis-Hastings 双随机权重 (Section 4.4)
- 软故障注入：constant / drift / intermittent (Eq. 3)
- 残差 r_i^{(k)} = ∇f_i(x_i) - Σ_j a_ij ∇f_j(x_j) (Eq. 4)
- 梯度跟踪更新 (Eq. 2) 和带 γ 的鲁棒更新 (Eq. 6)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Graph and consensus weights
# ---------------------------------------------------------------------------

def build_graph(N, radius=None, seed=0, max_attempts=20):
    """生成无向随机几何图，确保强连通（无向图等价于连通）。

    radius 缺省时取 ``1.2 * sqrt(log(N+1)/N)``——RGG 连通性阈值的标准估计加
    1.2 安全系数；若初次尝试不连通，按 ``r ← r * (1 + 0.05·attempt)`` 渐增半径
    直到连通。这是对论文 4.4.4 "minimum radius ensuring strong connectivity"
    的工程近似（理论下界 + 自适应增长），不是严格意义上的"最小半径"。

    返回 (W, adj, pos)，其中 W 是 Metropolis-Hastings 双随机权重矩阵。
    """
    rng = np.random.RandomState(seed)
    base_radius = radius if radius is not None else 1.2 * np.sqrt(np.log(N + 1) / N)

    for attempt in range(max_attempts):
        pos = rng.rand(N, 2)
        r = base_radius * (1.0 + 0.05 * attempt)  # 渐增半径直到连通
        adj = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(pos[i] - pos[j]) < r:
                    adj[i, j] = adj[j, i] = 1.0
        if _is_connected(adj):
            break
    else:
        raise RuntimeError(f"Failed to build connected graph after {max_attempts} attempts.")

    deg = adj.sum(axis=1)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if adj[i, j]:
                W[i, j] = 1.0 / (max(deg[i], deg[j]) + 1)
        W[i, i] = 1.0 - W[i, :].sum()
    return W, adj, pos


def _is_connected(adj):
    """连通性检查（DFS）。"""
    N = adj.shape[0]
    visited = np.zeros(N, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in np.where(adj[u] > 0)[0]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    return bool(visited.all())


def hop_neighborhood(adj, i, h):
    """返回智能体 i 的 h-hop 邻域（包含 i 自己）作为升序索引列表。"""
    visited = {i}
    frontier = {i}
    for _ in range(h):
        new_frontier = set()
        for u in frontier:
            for v in np.where(adj[u] > 0)[0]:
                if v not in visited:
                    new_frontier.add(int(v))
        visited |= new_frontier
        frontier = new_frontier
        if not frontier:
            break
    return sorted(visited)


# ---------------------------------------------------------------------------
# Faults (Eq. 3) and residuals (Eq. 4)
# ---------------------------------------------------------------------------

def apply_fault_injection(t, fault_config, N, d, rng):
    """根据故障配置返回 (faulty_mask, delta)，delta 形状 (N, d)。

    论文 Section 4.4 假设 small-fault regime：偏差有界。
    - constant   : 故障 onset 起恒定偏差 base_delta。
    - drift      : 以速率 base_delta 线性渐增，饱和于 drift_cap。
                   这是物理上的传感器渐变漂移，渐进至稳态偏差。论文实验
                   中所有 drift 调用点都显式设置 ``drift_cap=40``（让稳态
                   故障量级 0.002 × 40 = 0.08 落在 small-fault regime 内）；
                   函数级默认 100 仅作 schema fallback，不应直接依赖。
    - intermittent : 故障期内以概率 prob 出现一次幅度为 base_delta 的偏差。

    .. note::
       ``rng`` 由 ``run_optimization`` 主循环传入，与初始 X 采样、共识扰动等共
       享。这意味着如果未来 RPS 路径里加入随机性（例如 Monte Carlo PMF
       估计），上游的 rng 消耗会改变 intermittent 触发序列。当前所有 RPS 路径
       都是确定性的，故没问题；如要修改 RPS 内部用到 rng 时，建议为故障注入
       分配独立 ``RandomState(seed + 1)`` 以保护 intermittent 序列的可复现性。
    """
    faulty_mask = np.zeros(N, dtype=bool)
    delta = np.zeros((N, d))
    if t < fault_config['onset']:
        return faulty_mask, delta

    ftype = fault_config['type']
    base_delta = np.asarray(fault_config['delta'], dtype=float) if fault_config.get('delta') is not None else None

    for ag in fault_config['agents']:
        if ftype == 'constant':
            faulty_mask[ag] = True
            delta[ag] = base_delta
        elif ftype == 'drift':
            faulty_mask[ag] = True
            elapsed = t - fault_config['onset'] + 1
            cap = float(fault_config.get('drift_cap', 100.0))  # 单位为 base_delta 的倍数
            ramp = min(elapsed, cap)
            delta[ag] = base_delta * ramp
        elif ftype == 'intermittent':
            if rng.rand() < fault_config['prob']:
                faulty_mask[ag] = True
                delta[ag] = base_delta
        else:
            raise ValueError(f"Unknown fault type: {ftype}")
    return faulty_mask, delta


def compute_local_gradients(X, grad_fn_list, faulty_mask, delta):
    """每个智能体计算自己的（含故障的）局部梯度，返回 (N, d) 矩阵。"""
    N, d = X.shape
    grad = np.zeros_like(X)
    for i in range(N):
        grad[i] = grad_fn_list[i](X[i])
        if faulty_mask[i]:
            grad[i] = grad[i] + delta[i]
    return grad


def compute_residuals(grad, W):
    """残差 r_i^{(k)} = ∇f_i(x_i) - Σ_j a_ij ∇f_j(x_j)，向量化实现。

    返回标量残差范数 (N,) 与残差向量 (N, d)。
    """
    nb_avg = W @ grad
    res_vec = grad - nb_avg
    res_norm = np.linalg.norm(res_vec, axis=1)
    return res_norm, res_vec


# ---------------------------------------------------------------------------
# Gradient tracking updates
# ---------------------------------------------------------------------------

def gradient_tracking_step(X, Y, grad_old, grad_new, W, alpha, gamma=None):
    """一步梯度跟踪更新。

    - 当 ``gamma is None`` 时执行论文 Eq. (2)：
        X_{k+1} = W X_k − α Y_k
        Y_{k+1} = W Y_k + (∇f(x_{k+1}) − ∇f(x_k))

    - 当 ``gamma`` 给出 (N, N) 矩阵时执行论文 Eq. (6) 的扩展：
        x_i^{(k+1)} = Σ_j ā_ij x_j^{(k)} − α y_i^{(k)}
        y_i^{(k+1)} = Σ_j ā_ij y_j^{(k)} + (∇f_i^{new} − ∇f_i^{old})

      其中 ``ā_ij = γ_ij W_ij + (1−γ_ij)·δ_ij``，即把折扣的邻居权重转移
      给自身，保持每行之和为 1（双随机性的局部修复），从而避免被屏蔽
      agent 的状态污染共识、同时维持收敛常数。

      ``gamma=None`` 等价于 ``ā = W``，两条路径数值上一致（由
      ``test_gradient_tracking_step_actually_uses_gamma`` 守门）。
    """
    if gamma is None:
        A_eff = W
    else:
        A_eff = gamma * W
        row_sum = A_eff.sum(axis=1)
        deficit = 1.0 - row_sum
        A_eff = A_eff + np.diag(deficit)
    X_new = A_eff @ X - alpha * Y
    Y_new = A_eff @ Y + (grad_new - grad_old)
    return X_new, Y_new


# ---------------------------------------------------------------------------
# Communication degradation models (for figure_5 stress test)
# ---------------------------------------------------------------------------

def simulate_symmetric_packet_loss(W: np.ndarray, loss_rate: float,
                                    rng: np.random.RandomState) -> np.ndarray:
    """对称丢包后的双随机权重矩阵（论文 4.5.3 的"通信退化"模拟）。

    丢包模型：
    - 以概率 ``loss_rate`` 同时丢 ``W[i, j]`` 与 ``W[j, i]``（对称丢包）。
    - 把每行被丢失的权重之和加到对角，恢复行和=1。
    - 因 drop 与对角修补都对称，结果矩阵对称；对称 + 行和=1 ⇒ 列和=1，
      仍是合法的双随机矩阵。

    为什么必须双随机：
    - X 步进 ``W X − α Y`` 需要**行随机**（共识收敛）；
    - Y 步进 ``W Y + Δgrad`` 需要**列随机**（梯度平均不变量）。
    - 只修行不修列会让 Y-tracking 失效，figure_5 测的就不再是丢包鲁棒性。

    .. note::
       本简化把丢包建模为 W 上的"永久删边"（一次实验内固定）。论文 4.5.3
       描述的是 per-iteration packet drops（时变 W）；要让时变 W 仍维持
       双随机性需要 Sinkhorn 迭代，成本远高于本简化。读者应理解 figure_5
       中间子图反映的是"网络稀疏化后 RPS 鲁棒性"，而非"逐步丢包下 RPS
       鲁棒性"。

    Parameters
    ----------
    W         : Metropolis-Hastings 双随机权重矩阵 (N, N)
    loss_rate : 丢包率，[0, 1]
    rng       : 随机数发生器

    Returns
    -------
    np.ndarray
        丢包后的 W_mod，仍是双随机的对称矩阵。
    """
    if not (0.0 <= loss_rate <= 1.0):
        raise ValueError(f"loss_rate must be in [0, 1], got {loss_rate}")
    W_mod = W.copy()
    if loss_rate <= 0:
        return W_mod
    N = W.shape[0]
    tri_mask = np.triu(rng.rand(N, N) < loss_rate, k=1)
    sym_mask = tri_mask | tri_mask.T
    lost_row_sum = (W_mod * sym_mask).sum(axis=1)
    W_mod[sym_mask] = 0.0
    W_mod[np.arange(N), np.arange(N)] += lost_row_sum
    return W_mod
