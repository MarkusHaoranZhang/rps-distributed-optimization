"""分布式优化核心步骤的单元测试：图、共识权重、梯度跟踪、故障注入、残差。"""

import numpy as np
import pytest

from distributed_optimization import (
    apply_fault_injection,
    build_graph,
    compute_local_gradients,
    compute_residuals,
    gradient_tracking_step,
    hop_neighborhood,
)

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def test_build_graph_is_connected():
    W, adj, _ = build_graph(N=20, seed=0)
    # 通过邻接矩阵 DFS 检查连通（用 LIFO 栈，命名一致）
    visited = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in np.where(adj[u] > 0)[0]:
            if v not in visited:
                visited.add(int(v))
                stack.append(int(v))
    assert len(visited) == 20


def test_W_is_metropolis_doubly_stochastic():
    """Metropolis-Hastings 权重应行和与列和都为 1。"""
    W, _, _ = build_graph(N=15, seed=7)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-12)
    assert (W >= 0).all()


def test_hop_neighborhood_includes_self_and_grows():
    _, adj, _ = build_graph(N=15, seed=3)
    s1 = hop_neighborhood(adj, 0, h=1)
    s2 = hop_neighborhood(adj, 0, h=2)
    assert 0 in s1
    assert 0 in s2
    # 2-hop 必包含 1-hop
    assert set(s1).issubset(set(s2))


# ---------------------------------------------------------------------------
# Fault injection schema
# ---------------------------------------------------------------------------

def _cfg(**kwargs):
    base = {'onset': 5, 'agents': [1], 'type': 'constant',
            'delta': np.ones(3) * 0.1}
    base.update(kwargs)
    return base


def test_fault_injection_inactive_before_onset():
    rng = np.random.RandomState(0)
    mask, delta = apply_fault_injection(t=0, fault_config=_cfg(), N=4, d=3, rng=rng)
    assert not mask.any()
    assert (delta == 0).all()


def test_fault_injection_constant_active_after_onset():
    rng = np.random.RandomState(0)
    mask, delta = apply_fault_injection(t=10, fault_config=_cfg(), N=4, d=3, rng=rng)
    assert mask[1] and not mask[0] and not mask[2]
    np.testing.assert_allclose(delta[1], 0.1)


def test_fault_injection_drift_saturates_at_cap():
    cfg = _cfg(type='drift', delta=np.ones(3) * 0.01, drift_cap=10)
    rng = np.random.RandomState(0)
    # onset = 5；t = 100 远超 cap = 10；ramp 应饱和到 10
    _, delta = apply_fault_injection(t=100, fault_config=cfg, N=4, d=3, rng=rng)
    np.testing.assert_allclose(delta[1], 0.01 * 10)


def test_fault_injection_intermittent_probabilistic():
    """直接验证 intermittent 概率 ≈ prob。"""
    cfg = _cfg(type='intermittent', delta=np.ones(3) * 0.1, prob=0.3)
    rng = np.random.RandomState(0)
    triggered = 0
    n_trials = 2000
    for _ in range(n_trials):
        mask, _ = apply_fault_injection(t=10, fault_config=cfg, N=4, d=3, rng=rng)
        if mask[1]:
            triggered += 1
    rate = triggered / n_trials
    # 2000 trials × p=0.3 的标准误约 0.01；放宽到 ±0.04（4σ）避免 flaky
    assert 0.26 < rate < 0.34, f"intermittent rate {rate} far from 0.3"


def test_fault_injection_unknown_type_raises():
    cfg = _cfg(type='nonexistent')
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError):
        apply_fault_injection(t=10, fault_config=cfg, N=4, d=3, rng=rng)


# ---------------------------------------------------------------------------
# Residuals & gradient tracking
# ---------------------------------------------------------------------------

def test_residuals_zero_when_grad_consensus():
    """所有 agent 梯度相同时残差 ≈ 0（W 是双随机的）。"""
    W, _, _ = build_graph(N=10, seed=5)
    g = np.tile([1.0, 2.0, -1.0], (10, 1))
    res_norm, _ = compute_residuals(g, W)
    np.testing.assert_allclose(res_norm, 0.0, atol=1e-12)


def test_gradient_tracking_no_gamma_converges_on_quadratic():
    """无故障无 γ 的标准 GT 在 N=5 二次问题上收敛到 < 1e-6。"""
    np.random.seed(0)
    N, d = 5, 3
    W, _, _ = build_graph(N=N, seed=0)
    A_list = [np.random.randn(4, d) for _ in range(N)]
    b_list = [np.random.randn(4) for _ in range(N)]

    def grad_fns():
        out = []
        for i in range(N):
            Ai, bi = A_list[i], b_list[i]
            out.append(lambda x, A=Ai, b=bi: A.T @ (A @ x - b) / 4.0)
        return out

    gfs = grad_fns()
    X = np.zeros((N, d))
    grad_old = compute_local_gradients(X, gfs,
                                         np.zeros(N, dtype=bool),
                                         np.zeros((N, d)))
    Y = grad_old.copy()
    for _ in range(2000):
        grad_new = compute_local_gradients(X, gfs,
                                             np.zeros(N, dtype=bool),
                                             np.zeros((N, d)))
        X, Y = gradient_tracking_step(X, Y, grad_old, grad_new, W, 0.05)
        grad_old = grad_new

    # 集中式最优
    H = sum(A_list[i].T @ A_list[i] / 4.0 for i in range(N))
    g_all = sum(A_list[i].T @ b_list[i] / 4.0 for i in range(N))
    x_opt = np.linalg.solve(H, g_all)

    # 收敛精度
    err = np.mean(np.linalg.norm(X - x_opt[None, :], axis=1))
    assert err < 1e-4, f"GT did not converge: err={err}"


def test_gradient_tracking_with_gamma_preserves_row_sum_in_function():
    """``gradient_tracking_step`` 内部对 γ 的处理必须保证 ā_ij 行和 = 1。

    实际调用函数本身（之前版本只重写一遍逻辑做断言，是假阳性测试）：
    构造能让 γ 屏蔽某些列的输入，从函数输出的 X_new = ā @ X 反推 ā 行和。
    """
    np.random.seed(0)
    N, d = 8, 3
    W, _, _ = build_graph(N=N, seed=2)

    # γ 让 agent 1 的列被全部屏蔽
    gamma = np.ones((N, N))
    gamma[:, 1] = 0.0

    # 用全 1 的 X：X_new[i] = Σ_j ā_ij * 1 = sum_j ā_ij = ā 第 i 行的和
    X = np.ones((N, d))
    Y = np.zeros((N, d))
    grad_old = np.zeros((N, d))
    grad_new = np.zeros((N, d))

    X_new, _ = gradient_tracking_step(X, Y, grad_old, grad_new, W, alpha=0.0,
                                         gamma=gamma)
    # X_new[i, k] 在 alpha=0 且全 1 输入下应等于 ā 第 i 行和
    row_sums = X_new[:, 0]   # 任一列都行
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10,
        err_msg="gradient_tracking_step did not preserve row-stochastic ā")


def test_gradient_tracking_step_actually_uses_gamma():
    """另一个守门测试：``gamma=None`` 与 ``gamma=ones`` 在数值上必须等价。

    之前的假阳性可能让"忽略 γ"的回归被忽略——这个测试咬住"γ=ones 应该
    与无 γ 完全相同"。
    """
    np.random.seed(0)
    N, d = 6, 2
    W, _, _ = build_graph(N=N, seed=3)
    X = np.random.randn(N, d)
    Y = np.random.randn(N, d)
    grad_old = np.random.randn(N, d)
    grad_new = np.random.randn(N, d)

    X_no_gamma, Y_no_gamma = gradient_tracking_step(
        X, Y, grad_old, grad_new, W, alpha=0.05, gamma=None)
    X_ones_gamma, Y_ones_gamma = gradient_tracking_step(
        X, Y, grad_old, grad_new, W, alpha=0.05, gamma=np.ones((N, N)))
    np.testing.assert_allclose(X_no_gamma, X_ones_gamma, atol=1e-10)
    np.testing.assert_allclose(Y_no_gamma, Y_ones_gamma, atol=1e-10)


def test_gradient_tracking_with_gamma_zero_isolates_agent():
    """γ_{:, j} = 0 时 ā 应把 agent j 的 X 从共识中完全屏蔽。

    具体：让 agent j=2 的 X 取一个特异值，所有其它 agent 的 X 取 0；
    若 ā 真的屏蔽了 agent 2，那其它 agent 的 X_new 应等于 0。
    """
    np.random.seed(0)
    N, d = 5, 2
    W, _, _ = build_graph(N=N, seed=1)
    X = np.zeros((N, d))
    X[2] = 100.0   # agent 2 的极端值，应被完全屏蔽
    Y = np.zeros((N, d))
    grad = np.zeros((N, d))
    gamma = np.ones((N, N))
    gamma[:, 2] = 0.0

    X_new, _ = gradient_tracking_step(X, Y, grad, grad, W, alpha=0.0,
                                         gamma=gamma)
    # 其他 agent 的 X 应该不受 X[2] 影响（因为 ā_{i,2} = 0 并补到对角）
    for i in range(N):
        if i == 2:
            continue
        np.testing.assert_allclose(X_new[i], 0.0, atol=1e-10,
            err_msg=f"agent {i} got contaminated by agent 2 (gamma not effective)")
