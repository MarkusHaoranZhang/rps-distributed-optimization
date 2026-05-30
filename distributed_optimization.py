"""
Distributed-optimization core module
=====================================

Implements the methodology from the paper:
- Random geometric graph + Metropolis-Hastings doubly-stochastic weights
  (Section 4.4)
- Soft-fault injection: constant / drift / intermittent (Eq. 3)
- Residual r_i^{(k)} = grad f_i(x_i) - sum_j a_ij grad f_j(x_j) (Eq. 4)
- Gradient-tracking update (Eq. 2) and the gamma-discounted robust update
  (Eq. 6).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Graph and consensus weights
# ---------------------------------------------------------------------------

def build_graph(N, radius=None, seed=0, max_attempts=20):
    """Build an undirected random geometric graph that is guaranteed to be
    strongly connected (which, for an undirected graph, is the same as
    connected).

    When ``radius`` is unspecified we use ``1.2 * sqrt(log(N+1)/N)`` -- a
    standard estimate of the RGG connectivity threshold with a 1.2 safety
    factor. If the first attempt is not connected we grow the radius via
    ``r <- r * (1 + 0.05 * attempt)`` until connectivity is achieved. This is
    an engineering approximation of the "minimum radius ensuring strong
    connectivity" mentioned in Section 4.4.4 (theoretical lower bound +
    adaptive growth), not a strict "minimum radius" in the formal sense.

    Returns ``(W, adj, pos)`` where ``W`` is the Metropolis-Hastings
    doubly-stochastic weight matrix.
    """
    rng = np.random.RandomState(seed)
    base_radius = radius if radius is not None else 1.2 * np.sqrt(np.log(N + 1) / N)

    for attempt in range(max_attempts):
        pos = rng.rand(N, 2)
        r = base_radius * (1.0 + 0.05 * attempt)  # grow radius until connected
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
    """Connectivity check (DFS)."""
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
    """Return the ``h``-hop neighborhood of agent ``i`` (including ``i``
    itself) as a sorted list of indices."""
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
    """Return ``(faulty_mask, delta)`` for the current step, with ``delta``
    of shape ``(N, d)``.

    Section 4.4 of the paper assumes the small-fault regime: bounded offsets.

    - ``constant``    : starting at the onset, a constant offset
                        ``base_delta``.
    - ``drift``       : linear ramp at rate ``base_delta`` per step,
                        saturating at ``drift_cap``. This models a physical
                        sensor's gradual drift toward a steady-state offset.
                        Every drift call site in the paper experiments
                        explicitly sets ``drift_cap=40`` (so the steady-state
                        magnitude 0.002 * 40 = 0.08 still lies within the
                        small-fault regime); the function-level default of
                        100 is only a schema fallback and should not be
                        relied on directly.
    - ``intermittent`` : during the fault period, with probability ``prob``
                        a single offset of magnitude ``base_delta`` is
                        applied.

    .. note::
       ``rng`` is passed in by the main loop in ``run_optimization`` and is
       shared with initial-X sampling, consensus perturbations, etc. This
       means that if randomness is ever added to the RPS path (e.g. Monte
       Carlo PMF estimation), upstream consumption of the rng would change
       the intermittent trigger sequence. All current RPS paths are
       deterministic, so this is fine; if rng is added inside RPS in the
       future, allocate a separate ``RandomState(seed + 1)`` for fault
       injection to protect the reproducibility of the intermittent
       sequence.
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
            cap = float(fault_config.get('drift_cap', 100.0))  # in units of base_delta
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
    """Each agent computes its own (possibly faulty) local gradient.
    Returns an ``(N, d)`` matrix."""
    N, d = X.shape
    grad = np.zeros_like(X)
    for i in range(N):
        grad[i] = grad_fn_list[i](X[i])
        if faulty_mask[i]:
            grad[i] = grad[i] + delta[i]
    return grad


def compute_residuals(grad, W):
    """Residual ``r_i^{(k)} = grad f_i(x_i) - sum_j a_ij grad f_j(x_j)``,
    vectorized.

    Returns the scalar residual norms ``(N,)`` and the residual vectors
    ``(N, d)``.
    """
    nb_avg = W @ grad
    res_vec = grad - nb_avg
    res_norm = np.linalg.norm(res_vec, axis=1)
    return res_norm, res_vec


# ---------------------------------------------------------------------------
# Gradient tracking updates
# ---------------------------------------------------------------------------

def gradient_tracking_step(X, Y, grad_old, grad_new, W, alpha, gamma=None):
    """One gradient-tracking step.

    - When ``gamma is None``, performs the paper's Eq. (2):
        X_{k+1} = W X_k - alpha Y_k
        Y_{k+1} = W Y_k + (grad f(x_{k+1}) - grad f(x_k))

    - When ``gamma`` is an ``(N, N)`` matrix, performs the extended Eq. (6):
        x_i^{(k+1)} = sum_j abar_ij x_j^{(k)} - alpha y_i^{(k)}
        y_i^{(k+1)} = sum_j abar_ij y_j^{(k)} + (grad f_i^{new} - grad f_i^{old})

      where ``abar_ij = gamma_ij * W_ij + (1 - gamma_ij) * delta_ij``: the
      mass that was discounted away from a neighbor is returned to the
      diagonal so each row still sums to 1 (a local repair of double
      stochasticity). This avoids contaminating the consensus state with
      masked-out agents while preserving the convergence constants.

      ``gamma=None`` is exactly equivalent to ``abar = W`` -- the two paths
      are numerically identical (guarded by
      ``test_gradient_tracking_step_actually_uses_gamma``).
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
    """Return a doubly-stochastic weight matrix after symmetric packet
    loss ("communication degradation" simulation in Section 4.5.3).

    Loss model:
    - With probability ``loss_rate``, both ``W[i, j]`` and ``W[j, i]`` are
      dropped (symmetric loss).
    - For each row, the dropped weight mass is added to the diagonal so row
      sums return to 1.
    - Because both the drop and the diagonal repair are symmetric, the
      result is still symmetric; symmetric + row-sum 1 implies column-sum 1,
      i.e. the result is still a valid doubly-stochastic matrix.

    Why double stochasticity is required:
    - The X update ``W X - alpha Y`` needs **row stochasticity** (consensus
      convergence).
    - The Y update ``W Y + delta_grad`` needs **column stochasticity** (the
      gradient-average invariant).
    - Repairing rows but not columns would break Y-tracking and figure_5
      would no longer measure packet-loss robustness.

    .. note::
       This simplification models packet loss as a "permanent edge removal"
       on W (fixed within a single experiment). Section 4.5.3 of the paper
       describes per-iteration packet drops (time-varying W); keeping
       time-varying W doubly stochastic would require Sinkhorn iterations,
       at much higher cost than this simplification. Readers should
       interpret the middle panel of figure_5 as "RPS robustness under a
       sparsified network", not "RPS robustness under per-iteration packet
       drops".

    Parameters
    ----------
    W         : Metropolis-Hastings doubly-stochastic weight matrix ``(N, N)``
    loss_rate : packet-loss rate, in ``[0, 1]``
    rng       : random number generator

    Returns
    -------
    np.ndarray
        ``W_mod`` after losses; still a doubly-stochastic symmetric matrix.
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
