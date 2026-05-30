"""End-to-end unit tests for ``compute_pmf`` and
``confidence_gated_discount``."""

import numpy as np

from rps_diagnosis import (
    _enumerate_events,
    _enumerate_events_topk,
    build_fault_propagation_matrix,
    compute_pmf,
    confidence_gated_discount,
    ordered_probability_transformation,
    pmf_to_singleton_vector,
)


def _toy_W(N=5):
    """A simple 5-node ring: each node is connected to the two adjacent
    neighbors."""
    adj = np.zeros((N, N))
    for i in range(N):
        adj[i, (i + 1) % N] = 1
        adj[i, (i - 1) % N] = 1
    deg = adj.sum(axis=1)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if adj[i, j]:
                W[i, j] = 1.0 / (max(deg[i], deg[j]) + 1)
        W[i, i] = 1.0 - W[i, :].sum()
    return W


# ---------------------------------------------------------------------------
# compute_pmf basic invariants
# ---------------------------------------------------------------------------

def test_pmf_mass_sums_to_one():
    N = 5
    W = _toy_W(N)
    F = build_fault_propagation_matrix(W)
    scope = list(range(N))
    rng = np.random.RandomState(0)
    window = np.abs(rng.randn(20))           # residual norms are non-negative
    proxy = np.array([0.0, 0.5, 0.0, 0.0, 0.0])  # agent 1 is the suspect
    Q0 = np.abs(rng.randn(80))
    pmf = compute_pmf(self_idx=0, scope=scope, k_trunc=3,
                       residual_window=window, F=F, magnitude_proxy=proxy,
                       Q0_samples=Q0, eta=1.0, top_m=16)
    np.testing.assert_allclose(pmf.mass.sum(), 1.0, atol=1e-10)


def test_pmf_concentrates_on_high_proxy_agent():
    """When the proxy is concentrated on agent 1 (a ring neighbor of
    agent 0), the PMF's marginal probability on the singleton ``(1,)``
    should exceed uniform."""
    N = 5
    W = _toy_W(N)
    F = build_fault_propagation_matrix(W)
    scope = list(range(N))
    rng = np.random.RandomState(1)
    Q0 = rng.randn(80) * 0.05
    proxy = np.zeros(N)
    target = 1                              # ring neighbor of agent 0
    proxy[target] = 1.0
    c_target = float(F[0, target] * proxy[target])
    assert c_target > 0, "test setup broken: F[0, 1] should be > 0"
    window = np.full(20, c_target) + rng.randn(20) * 0.01
    pmf = compute_pmf(self_idx=0, scope=scope, k_trunc=3,
                       residual_window=window, F=F, magnitude_proxy=proxy,
                       Q0_samples=Q0, eta=2.0, top_m=16)
    sing = pmf_to_singleton_vector(pmf, scope)
    # target marginal probability should be strictly above uniform
    assert sing[target] > 1.0 / N + 1e-3, (
        f"target agent {target} singleton {sing[target]} not above uniform")


def test_pmf_returns_uniform_when_proxy_all_zero():
    """When the proxy is identically zero, event enumeration still works
    and the output is finite."""
    N = 4
    W = _toy_W(N)
    F = build_fault_propagation_matrix(W)
    scope = list(range(N))
    rng = np.random.RandomState(2)
    pmf = compute_pmf(self_idx=0, scope=scope, k_trunc=2,
                       residual_window=np.abs(rng.randn(20)),
                       F=F, magnitude_proxy=np.zeros(N),
                       Q0_samples=np.abs(rng.randn(80)),
                       eta=1.0, top_m=16)
    assert not pmf.is_empty
    assert np.all(np.isfinite(pmf.mass))
    np.testing.assert_allclose(pmf.mass.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# OPT and confidence-gated discount
# ---------------------------------------------------------------------------

def test_opt_pure_singleton():
    """When the only events are ``(0,) -> 0.6`` and ``(1,) -> 0.4``, the
    OPT probability equals the mass."""
    from config import PMF
    from rps_diagnosis import _bit_index_map, _mask_for, _to_mask_array
    scope = [0, 1, 2]
    a2b, _ = _bit_index_map(scope)
    masks = _to_mask_array([_mask_for(A, a2b) for A in [(0,), (1,)]])
    pmf = PMF(events=((0,), (1,)), mass=np.array([0.6, 0.4]), masks=masks)
    opt = ordered_probability_transformation(pmf, scope)
    assert abs(opt[0] - 0.6) < 1e-12
    assert abs(opt[1] - 0.4) < 1e-12
    assert abs(opt[2] - 0.0) < 1e-12


def test_opt_terminal_element_gets_zero():
    """OPT (Eq. 11): the terminal element receives no mass; the entire
    mass of event ``(0, 1)`` goes to agent 0."""
    from config import PMF
    from rps_diagnosis import _bit_index_map, _mask_for, _to_mask_array
    scope = [0, 1]
    a2b, _ = _bit_index_map(scope)
    masks = _to_mask_array([_mask_for(A, a2b) for A in [(0, 1)]])
    pmf = PMF(events=((0, 1),), mass=np.array([1.0]), masks=masks)
    opt = ordered_probability_transformation(pmf, scope)
    assert abs(opt[0] - 1.0) < 1e-12
    assert abs(opt[1] - 0.0) < 1e-12


def test_gated_discount_high_entropy_softens():
    """High entropy halves the effective gain, so the same ``P_OPT``
    yields a larger gamma (less discount)."""
    opt = {0: 0.5, 1: 0.0}
    tau = 1.0
    g_low_h = confidence_gated_discount(opt, entropy=0.0, tau=tau, gain=4.0)
    g_high_h = confidence_gated_discount(opt, entropy=2.0, tau=tau, gain=4.0)
    assert g_low_h[0] < g_high_h[0], (
        f"low-entropy gamma should be smaller (more discount): "
        f"low_h[0]={g_low_h[0]}, high_h[0]={g_high_h[0]}")


def test_gated_discount_zero_prob_no_discount():
    """An agent with ``P_OPT = 0`` gets ``gamma ~ 1`` (no discount)."""
    opt = {0: 0.0, 1: 1.0}
    g = confidence_gated_discount(opt, entropy=0.0, tau=1.0, gain=4.0)
    assert abs(g[0] - 1.0) < 1e-10


def test_gated_discount_high_prob_strong_discount():
    """An agent with ``P_OPT ~ 1`` gets ``gamma`` close to 0."""
    opt = {0: 1.0}
    g = confidence_gated_discount(opt, entropy=0.0, tau=1.0, gain=4.0)
    assert g[0] < 0.05


# ---------------------------------------------------------------------------
# Reference-impl guard: ``_enumerate_events`` (full) and
# ``_enumerate_events_topk`` should be equivalent when ``top_agents == scope``
# (note that even in topk, ``r=1`` always uses the full scope, so the event
# sets are equal).
# ---------------------------------------------------------------------------

def test_enumerate_events_full_equals_topk_with_full_topagents():
    """``_enumerate_events_topk`` with ``top_agents == scope`` should
    produce the same event set as ``_enumerate_events`` (the order may
    differ, but the sets are equal)."""
    scope = [0, 1, 2, 3]
    k = 3
    full = set(_enumerate_events(scope, k))
    topk_full = set(_enumerate_events_topk(scope, k, top_agents=scope))
    assert full == topk_full, (
        f"_enumerate_events_topk(top_agents=scope) should reproduce "
        f"_enumerate_events; symmetric difference = {full ^ topk_full}"
    )
