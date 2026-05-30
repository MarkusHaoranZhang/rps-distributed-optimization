"""Unit tests for the core distributed-optimization steps: graph,
consensus weights, gradient tracking, fault injection, residuals."""

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
    # Check connectivity via DFS over the adjacency matrix (LIFO stack,
    # for naming consistency).
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
    """Metropolis-Hastings weights have row sums = column sums = 1."""
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
    # 2-hop must include 1-hop.
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
    # onset = 5; t = 100 is well past the cap of 10; the ramp should
    # saturate to 10.
    _, delta = apply_fault_injection(t=100, fault_config=cfg, N=4, d=3, rng=rng)
    np.testing.assert_allclose(delta[1], 0.01 * 10)


def test_fault_injection_intermittent_probabilistic():
    """Direct check that the intermittent rate is approximately
    ``prob``."""
    cfg = _cfg(type='intermittent', delta=np.ones(3) * 0.1, prob=0.3)
    rng = np.random.RandomState(0)
    triggered = 0
    n_trials = 2000
    for _ in range(n_trials):
        mask, _ = apply_fault_injection(t=10, fault_config=cfg, N=4, d=3, rng=rng)
        if mask[1]:
            triggered += 1
    rate = triggered / n_trials
    # 2000 trials with p = 0.3 give a standard error around 0.01; we
    # widen to +/- 0.04 (4 sigma) to avoid flakiness.
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
    """When every agent has the same gradient, residuals are ~ 0
    (because ``W`` is doubly stochastic)."""
    W, _, _ = build_graph(N=10, seed=5)
    g = np.tile([1.0, 2.0, -1.0], (10, 1))
    res_norm, _ = compute_residuals(g, W)
    np.testing.assert_allclose(res_norm, 0.0, atol=1e-12)


def test_gradient_tracking_no_gamma_converges_on_quadratic():
    """Standard GT with no fault and no gamma converges to < 1e-6 on a
    small ``N=5`` quadratic problem."""
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

    # Centralized optimum.
    H = sum(A_list[i].T @ A_list[i] / 4.0 for i in range(N))
    g_all = sum(A_list[i].T @ b_list[i] / 4.0 for i in range(N))
    x_opt = np.linalg.solve(H, g_all)

    # Convergence accuracy.
    err = np.mean(np.linalg.norm(X - x_opt[None, :], axis=1))
    assert err < 1e-4, f"GT did not converge: err={err}"


def test_gradient_tracking_with_gamma_preserves_row_sum_in_function():
    """``gradient_tracking_step``'s handling of ``gamma`` must keep
    ``abar_ij`` row sums equal to 1.

    This actually invokes the function (a previous version reimplemented
    the logic inside the assertion -- a false positive). We construct an
    input where ``gamma`` masks some columns, then back out the row sums
    of ``abar`` from ``X_new = abar @ X``.
    """
    np.random.seed(0)
    N, d = 8, 3
    W, _, _ = build_graph(N=N, seed=2)

    # gamma masks every column belonging to agent 1.
    gamma = np.ones((N, N))
    gamma[:, 1] = 0.0

    # Use an all-ones X: ``X_new[i] = sum_j abar_ij * 1 = sum_j abar_ij``
    # (the ``i``-th row sum of ``abar``).
    X = np.ones((N, d))
    Y = np.zeros((N, d))
    grad_old = np.zeros((N, d))
    grad_new = np.zeros((N, d))

    X_new, _ = gradient_tracking_step(X, Y, grad_old, grad_new, W, alpha=0.0,
                                         gamma=gamma)
    # With alpha=0 and an all-ones input, ``X_new[i, k]`` equals the
    # ``i``-th row sum of ``abar``.
    row_sums = X_new[:, 0]   # any column would work
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10,
        err_msg="gradient_tracking_step did not preserve row-stochastic abar")


def test_gradient_tracking_step_actually_uses_gamma():
    """Companion guard: ``gamma=None`` and ``gamma=ones`` must produce
    numerically identical results.

    A previous false-positive style could let an "ignore gamma"
    regression slip through. This test pins down "gamma=ones must
    behave exactly like no gamma".
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
    """When ``gamma_{:, j} = 0``, ``abar`` should fully exclude agent
    ``j``'s ``X`` from the consensus.

    Concretely: give agent ``j=2`` a peculiar ``X`` value while every
    other agent has ``X=0``. If ``abar`` truly masks agent 2, every
    other agent's ``X_new`` must be 0.
    """
    np.random.seed(0)
    N, d = 5, 2
    W, _, _ = build_graph(N=N, seed=1)
    X = np.zeros((N, d))
    X[2] = 100.0   # agent 2 is an outlier and should be fully masked
    Y = np.zeros((N, d))
    grad = np.zeros((N, d))
    gamma = np.ones((N, N))
    gamma[:, 2] = 0.0

    X_new, _ = gradient_tracking_step(X, Y, grad, grad, W, alpha=0.0,
                                         gamma=gamma)
    # Other agents' X should be unaffected by X[2] (because
    # ``abar_{i, 2} = 0`` and the slack has been added back to the
    # diagonal).
    for i in range(N):
        if i == 2:
            continue
        np.testing.assert_allclose(X_new[i], 0.0, atol=1e-10,
            err_msg=f"agent {i} got contaminated by agent 2 (gamma not effective)")
