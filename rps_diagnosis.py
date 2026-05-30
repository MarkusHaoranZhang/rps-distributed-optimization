"""
RPS diagnosis module (vectorized high-performance implementation)
==================================================================

Implements the algorithms in Sections 4.1-4.3 of the paper:

- Energy distance D(R, Q0)
- Expected residual magnitude (Eq. 7); fault magnitude is proxied by the
  residual norm (we never read the true delta).
- Support score (Eq. 8) -- engineered as a z-score in the production path
  (see ``IMPLEMENTATION_NOTES.md``).
- Truncated PMF / softmax (Eq. 9).
- Left intersection / left orthogonal sum LOS (Eq. 10) -- bitmask-accelerated.
- Neighbor reliability via JS divergence.
- Ordered probability transformation OPT (Eq. 11).
- Confidence-gated soft discount (Eq. 12) -- a continuous relaxation of the
  paper's piecewise rule.

The PMF data structure lives in ``config.PMF``.
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
    """Energy distance (Cramer-style) between two sample sets.

    ``D(X, Y) = 2 E|X-Y| - E|X-X'| - E|Y-Y'|``.
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
    """Build Q0 samples (flattened to 1-D) from the residual-norm history of
    the fault-free burn-in phase."""
    return np.asarray(residual_norms_history, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Fault propagation matrix
# ---------------------------------------------------------------------------

def build_fault_propagation_matrix(W: np.ndarray) -> np.ndarray:
    """Linear sensitivity of residuals to offsets: ``r = (I - W) delta``,
    so ``F = |I - W|``."""
    return np.abs(np.eye(W.shape[0]) - W)  # type: ignore[index]


# ---------------------------------------------------------------------------
# Bitmask utilities
# ---------------------------------------------------------------------------

def _to_mask_array(masks_list: Sequence[int]) -> np.ndarray:
    """Use ``int64`` for bitmasks of <= 63 bits; fall back to ``object``
    (which holds Python big ints) otherwise."""
    try:
        return np.array(list(masks_list), dtype=np.int64)
    except OverflowError:
        # Python int exceeds int64 range (e.g. when N > 63 the bitmask may
        # be >= 2^63).
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
    """Full truncated permutation event space: every permutation for
    ``r = 1..k``.

    .. note::
       This is the **reference implementation** of Eq. (9)'s
       ``PES_k(Theta_i)`` (full enumeration). The production path,
       ``compute_pmf``, uses ``_enumerate_events_topk`` for prior-driven
       sparsification. Use this function when you want a literal
       comparison with the equation. When ``top_agents == scope``,
       ``_enumerate_events_topk`` reduces to this function (guarded by
       ``test_enumerate_events_full_equals_topk_with_full_topagents``).
    """
    events: List[tuple] = []
    for r in range(1, min(k_trunc, len(scope)) + 1):
        for perm in permutations(scope, r):
            events.append(tuple(perm))
    return events


def _enumerate_events_topk(scope: Sequence[int], k_trunc: int,
                            top_agents: Sequence[int]) -> List[tuple]:
    """Two-stage enumeration: full ``scope`` for ``r=1``, but for ``r >= 2``
    only permutations within ``top_agents``.

    This is the PMF's "prior-driven sparsification": Bayesian diagnosis
    concentrates mass on the few agents with the highest probability, so
    restricting the ``r >= 2`` permutation space to ``top_agents`` shrinks
    the event count from ``O(|Theta|^k)`` to ``O(|Theta| + |T|^k)``.
    """
    events: List[tuple] = [(a,) for a in scope]
    base = list(top_agents)
    for r in range(2, min(k_trunc, len(base)) + 1):
        for perm in permutations(base, r):
            events.append(tuple(perm))
    return events


def expected_residual_at(self_idx: int, assumed_faulty_agents: Sequence[int],
                          F: np.ndarray, magnitude_proxy: np.ndarray) -> float:
    """Scalar ``E[r_i | A]`` (Eq. 7).

    .. note::
       This is the **reference implementation** of Eq. (7); use it when
       comparing with the equation. The high-performance path inside
       ``compute_pmf`` computes the expected residual for every event in
       one shot via the vectorized ``H @ contrib_in_scope``, and does not
       call this function.

       The argument ``assumed_faulty_agents`` is the list of **assumed**
       faulty-agent indices (the elements of permutation event ``A``); it
       is *not* the ground-truth fault configuration. Do not confuse it
       with the ``fault_config: dict`` consumed by
       ``apply_fault_injection(fault_config, ...)``.
    """
    return float(sum(F[self_idx, j] * magnitude_proxy[j] for j in assumed_faulty_agents))


def compute_support_score(window: np.ndarray, expected_value: float,
                           Q0_samples: np.ndarray) -> float:
    """Support (Eq. 8): ``s_A = -log D(R - E[r|A], Q_0)``.

    .. note::
       This is the **reference implementation** of Eq. (8); use it when
       comparing with the equation. The high-performance path in
       ``compute_pmf`` uses a z-score approximation (see IMPL Sec. 2) and
       does not call this function. We keep this helper around so the
       reader can match code to formula line-by-line.
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
    """Construct a truncated PMF (Eq. 9).

    In the small-fault regime, mean-shift dominates the residual
    distribution (paper's assumption). The support score is therefore
    computed as a z-score, ``s_A = -|mean(R) - c_A| / sigma_0``. This is
    asymptotically equivalent to the energy-distance form, but with O(E)
    per-step cost instead of O(E * s * M).

    Parameters
    ----------
    self_idx : global index of this agent
    scope    : this agent's diagnosable scope (h-hop neighborhood)
    k_trunc  : truncation order ``k``
    residual_window : sliding window of residual norms for this agent ``(s,)``
    F        : fault-propagation matrix ``(N, N)``
    magnitude_proxy : per-agent fault-magnitude proxy ``(N,)``
    Q0_samples : nominal-distribution samples (1-D); only used to estimate
                 ``sigma_0``. No sub-sampling is performed.
    eta      : softmax temperature
    top_m    : number of top events kept by the PMF
    top_agents_k : number of candidate agents kept by the two-stage
                   enumeration
    proxy_global_weight : mixing coefficient
                          ``contrib + w * proxy_in_scope``

    Notes
    -----
    Building the H matrix involves an O(E * |A|) double Python loop. With
    ``top_m=16`` and ``k_trunc=3`` the total assignment count is around 50,
    and combined with the ``diagnose_every=5`` throttle it does not show
    up as a hot spot in profiling, so we keep the readable form rather
    than vectorizing it.
    """
    a2b, _ = _bit_index_map(scope)

    F_row = F[self_idx]
    contrib = F_row * magnitude_proxy
    scope_arr = np.asarray(scope, dtype=int)
    contrib_in_scope = contrib[scope_arr]

    # Inject the global proxy into the per-agent local signal as well: a
    # faulty agent's own residual increment reflects (1 - W_jj) * ||delta_j||
    # without going through F[i, :], which is a useful complementary signal.
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

    # Per-event c_A: compute everything at once via the sparse indicator
    # matrix H.
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
    """Marginalize a PMF to a singleton-probability vector.

    An event ``A = (a_1, ..., a_r)`` distributes its mass equally among
    its members; the resulting vector is then normalized.
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
    """Jensen-Shannon divergence between two PMFs (over their marginal
    singleton vectors)."""
    p = sing1 if sing1 is not None else pmf_to_singleton_vector(pmf1, scope)
    q = sing2 if sing2 is not None else pmf_to_singleton_vector(pmf2, scope)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    # Defensive normalization: the caller-supplied sing1 / sing2 are not
    # guaranteed normalized (e.g. cached intermediates from
    # ``_order_by_js``), and the clip step can also break sum-to-1.
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
    """Left intersection ``A inter B``: keep the order of ``A`` and drop
    elements not in ``B``."""
    Bset = set(B)
    return tuple(x for x in A if x in Bset)


def _combine_with_intersection(pmf_a: PMF, pmf_b: PMF, scope: Sequence[int],
                                top_m: int, ordered: bool) -> PMF:
    """Unified implementation of LOS (``ordered=True``) and DS
    (``ordered=False``).

    The only difference is how the intersection tuple is built:
        LOS: keep the order of event ``A`` from ``pmf_a``;
        DS : take the set intersection and sort by global index.
    Every other step (outer-product mass, conflict normalization, top_m
    truncation) is shared.
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
    """Paper Eq. (10), LOS: ``pmf_a`` is the more reliable source; the
    output keeps the event order of ``A``."""
    return _combine_with_intersection(pmf_a, pmf_b, scope, top_m, ordered=True)


def dempster_shafer_combination(pmf_a: PMF, pmf_b: PMF, scope: Sequence[int],
                                 top_m: int = 16) -> PMF:
    """Symmetric Dempster-Shafer combination (used by the ``RPS-NoOrder``
    ablation)."""
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
    """Directional fusion (paper Eq. 10 + Section 4.2): ``pmf_self`` is
    the anchor; combine with neighbors in order of increasing JS
    divergence using LOS."""
    if not neighbor_pmfs:
        return pmf_self
    fused = pmf_self
    for idx in _order_by_js(pmf_self, neighbor_pmfs, scope):
        fused = left_orthogonal_sum(fused, neighbor_pmfs[idx], scope, top_m=top_m)
    return fused


def _collapse_to_unordered(pmf: PMF, scope: Sequence[int]) -> PMF:
    """Collapse a PMF's ordered permutation events into unordered set
    events.

    Section 4.4.2 of the paper defines RPS-NoOrder as "collapsing ordered
    tuples to unordered sets before fusion". Here ``(a, b)`` and
    ``(b, a)`` are merged into the single unordered key ``(min, max)``,
    with masses summed.
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
    """RPS-Symmetric ablation (literal definition from Section 4.4.2).

    The paper says: "directional LOS fusion is replaced by **symmetric
    Dempster-Shafer combination**". The whole fusion chain therefore uses
    DS, **never LOS** -- self does not act as an ordering anchor. DS is a
    symmetric operation, so the order of the sources does not matter.
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
    """RPS-NoOrder ablation (literal definition from Section 4.4.2).

    The paper says: "the permutation structure is removed by **collapsing
    ordered tuples to unordered sets before fusion**".

    Implementation: every PMF is collapsed to unordered set events before
    fusion, then combined along a DS chain. Output event keys are sorted
    tuples, so the OPT step cannot "recover" any information from order.
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
    """Project a PMF onto a sub-scope: keep only the elements of each
    event that lie in ``target_scope``, then renormalize."""
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
    """OPT (paper Eq. 11): the terminal element of an event receives no
    mass; the prefix splits the mass equally."""
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
    """Continuous soft discount
    ``gamma_ij = base_keep * exp(-effective_gain * P_OPT(j))``.

    Eq. (12) of the paper is piecewise (high entropy -> 1, low entropy ->
    ``P / max P``). Here we replace it by a monotone continuous form:

    - ``P_OPT(j) = 0`` (no suspicion) -> ``gamma ~ base_keep`` (close to 1,
      no discount).
    - ``P_OPT(j) -> 1`` (confirmed fault) -> ``gamma -> 0`` (strong
      suppression).
    - At high entropy (``H >= tau``, weak evidence) ``effective_gain`` is
      halved to avoid overreaction when uncertain.

    See ``IMPLEMENTATION_NOTES.md`` for the parameter description.
    """
    eff_gain = 0.5 * gain if entropy >= tau else gain
    return {a: float(base_keep * np.exp(-eff_gain * p))
            for a, p in opt_probs.items()}


def calibrate_tau(entropy_history: Sequence[float], quantile: float = 0.95) -> float:
    """Calibrate tau as the requested quantile of the fault-free PMF
    entropy distribution (paper Section 4.3)."""
    if len(entropy_history) == 0:
        return float('inf')
    return float(np.quantile(np.asarray(entropy_history, dtype=float), quantile))
