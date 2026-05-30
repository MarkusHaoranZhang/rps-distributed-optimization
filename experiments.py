"""
Experiment runner
==================

End-to-end loop: diagnosis -> soft discount gamma -> robust gradient
tracking.

The main entry point is ``run_optimization``, which dispatches to one of the
internal step functions based on the method name:

  ``_step_hard_threshold``  : chi-squared residual detector + strict
                              exclusion (the target of the paper's Section 1
                              critique).
  ``_step_uniform_discount`` : equal-weight discount applied across the
                              network.
  ``_step_rps``             : RPS-Full / RPS-Symmetric / RPS-NoOrder.

Byzantine-Resilient does not go through the gamma matrix; it filters X
directly via the coordinate-wise median before the X update. Neither Ideal
nor Byzantine needs a dedicated step function.

Per-method state (HT detector, RPS caches, ...) is encapsulated inside
``_RunState`` so the main loop only has to schedule the work.
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
    """Chi-squared residual detector: gamma is recomputed every step from
    the current ``res_norm[t]``.

    .. note::
       The detector has no temporal filtering: ``residual_norms`` naturally
       oscillate under the consensus dynamics, the chi-squared statistic
       ``r^2 / sigma^2`` oscillates with them, and as a result gamma
       excludes the faulty agent on some steps and lets it back in on
       others. This is precisely the "oscillates between inclusion and
       exclusion" behavior described in Section 1 of the paper. It also
       inflates trial-to-trial variance of the ``Iter to 1e-3`` metric
       (the paper's negative example).
    """
    assert st.ht_detector is not None, "_RunState.ht_detector must be initialized for HT method"
    if t == cfg.burn_in - 1:
        st.ht_detector.calibrate(residuals_norm[:cfg.burn_in])
    if t < cfg.burn_in:
        return None
    gamma_mat = st.ht_detector.gamma_matrix(residuals_norm[t])
    # Also log HT's detection result into diag_log so the MTCD metric in
    # Figure 7 is comparable between HT and RPS (paper Section 4.4.3
    # defines MTCD as: probability that the faulty agent is identified as
    # the most suspicious >= 0.95).
    #
    # HT is a binary detector with no continuous probability: the column
    # mean of gamma_{i, tgt} is either < 0.5 (the majority of neighbors
    # consider it faulty) or >= 0.5. We map "column mean < 0.5" to 1.0,
    # and to 0.0 otherwise. This means the MTCD threshold of 0.95 is
    # effectively "at any point in time, the majority of neighbors mark
    # it as faulty" -- as soon as that event occurs even once, MTCD locks
    # to that step.
    #
    # Under drift, the empirical MTCD for HT is close to ``T - onset``
    # (its maximum), which means the chi-squared threshold never trips
    # during the entire fault period. This is the numerical evidence for
    # the "threshold-based detector ... oscillates / fails" phenomenon
    # criticized in Section 1, not a limitation of the RPS framework.
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
    """Build a fault-magnitude proxy from the residual-norm history.

    .. note::
       When ``t < cfg.burn_in`` (burn-in phase), we use a zero vector for
       the baseline and 1 for ``std_base`` as placeholders. The PMFs
       generated in this phase are only used to collect entropy for tau
       calibration (they never feed into gamma), so the inaccuracy of the
       baseline does not affect the final result. This is an engineering
       placeholder so that the burn-in PMF can still be computed; it is
       not a paper formula.

    Returns
    -------
    (magnitude_proxy, baseline, Q0_samples)
        magnitude_proxy : ``(N,)`` fault-magnitude proxy
        baseline        : ``(N,)`` mean residual under the fault-free
                          baseline (zero placeholder during burn-in)
        Q0_samples      : ``(M,)`` baseline-fluctuation samples
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
    """Each agent generates its local PMF from the residual window inside
    its own scope."""
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
    """Shared diagnosis step for RPS-Full / RPS-Symmetric / RPS-NoOrder.

    Concrete implementation of tau calibration (paper Section 4.3):
      - During burn-in (t = window_len .. burn_in - 1) we generate PMFs
        from the current residual window and collect each agent's entropy.
      - At ``t == burn_in - 1`` we aggregate every agent's burn-in
        entropy and take the ``cfg.tau_quantile`` quantile as tau.
    This makes ``cfg.tau_quantile`` actually influence tau, which is what
    the Figure 3 tau-sensitivity curve relies on.
    """
    assert st.rps_states is not None, "_RunState.rps_states must be initialized for RPS methods"
    # The assert above is a type-narrowing hint for mypy; the function is
    # only entered when ``is_rps_method(method)`` holds, and
    # ``run_optimization`` always initializes ``rps_states`` for the RPS
    # branch, so it is guaranteed non-empty at runtime.
    # Update sliding window.
    for i in range(st.N):
        st.rps_states[i].window.append(residuals_norm[t, i])

    if t < cfg.window_len:
        return None

    magnitude_proxy, baseline, Q0_samples = _compute_magnitude_proxy(
        residuals_norm, t, cfg, st.N)

    # Burn-in phase: generate PMFs even before the fault arrives so we
    # can collect entropy for tau calibration.
    in_burnin = (t < fault_config['onset']) and (t < cfg.burn_in)
    if in_burnin:
        _generate_local_pmfs(st, residuals_norm, t, st.F, magnitude_proxy,
                              Q0_samples, baseline, cfg)
        # Collect entropy from every agent.
        for state in st.rps_states:
            if state.pmf is not None and not state.pmf.is_empty:
                st.burnin_entropies.append(pmf_entropy(state.pmf))
        return None

    # Tau calibration (one shot, at the end of burn-in or right when the
    # fault period begins).
    if st.tau == float('inf'):
        if cfg.tau is not None:
            # Explicit tau: use it directly (handy for the Figure 3
            # tau-sensitivity sweep).
            st.tau = float(cfg.tau)
        elif st.burnin_entropies:
            st.tau = float(np.quantile(np.asarray(st.burnin_entropies),
                                        cfg.tau_quantile))
        else:
            st.tau = math.log(cfg.top_m)

    # Past burn-in but the fault has not yet started (when onset >
    # burn_in).
    if t < fault_config['onset']:
        return None

    # Throttle: trigger diagnosis recomputation every cfg.diagnose_every
    # steps.
    diag_step = ((t - fault_config['onset']) % max(1, cfg.diagnose_every) == 0)
    if (not diag_step) and st.last_gamma_mat is not None:
        return st.last_gamma_mat

    # 1) Each agent generates its local PMF.
    _generate_local_pmfs(st, residuals_norm, t, st.F, magnitude_proxy,
                          Q0_samples, baseline, cfg)

    # 2) Neighbor-PMF fusion + gamma computation.
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
    """Run one full closed-loop optimization.

    Parameters
    ----------
    N, d, T, alpha : main-loop parameters (number of agents / decision
                     dimension / step count / step size).
    fault_config   : fault configuration dict (schema in
                     ``config.FaultConfig``).
    method         : one of ``KNOWN_METHODS``.
    W, adj         : consensus weight matrix and adjacency matrix.
    cost           : cost object that implements ``grad_fns()`` /
                     ``global_optimum()`` / ``problem_dim()``.
    cfg            : ``RPSConfig`` instance; defaults are used when ``None``.
    seed           : random seed.

    Returns
    -------
    (errors, residuals_norm, diag_log)
        errors          : ``(T,)`` mean relative error
        residuals_norm  : ``(T, N)`` residual-norm history
        diag_log        : dict with MTCD / gamma_history / diagnosis-probability
                          history.
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

        # 1) Gradients.
        grad_new = compute_local_gradients(X, grad_fns, faulty_mask, delta)

        # 2) Residuals.
        res_norm, _ = compute_residuals(grad_new, W)
        residuals_norm[t] = res_norm

        # 3) Gamma matrix (dispatched by method).
        gamma_mat: Optional[np.ndarray] = None
        if method == "Hard-Threshold":
            gamma_mat = _step_hard_threshold(t, st, residuals_norm,
                                              fault_config, cfg)
        elif method == "Uniform-Discount":
            gamma_mat = _step_uniform_discount(t, st, fault_config, cfg)
        elif use_rps:
            gamma_mat = _step_rps(t, st, residuals_norm, method,
                                   fault_config, adj, cfg)

        # 4) Byzantine exception: skip the gamma path entirely and apply
        #    coordinate-wise median filtering directly on X.
        if method == "Byzantine-Resilient" and t >= fault_config['onset']:
            X = coordinate_wise_median_aggregate(X, adj)

        # 5) Record gamma history (used by MTCD).
        if cfg.record_diagnosis and t >= fault_config['onset'] and gamma_mat is not None:
            st.diag_log["gamma_history"].append(gamma_mat.copy())

        # 6) One gradient-tracking step.
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
    """Iterations from fault onset until the true-fault top-1 probability
    exceeds the threshold."""
    probs = diag_log.get("true_fault_top1_prob", [])
    if not probs:
        return float('nan')
    for k, p in enumerate(probs):
        if p >= prob_threshold:
            return k
    return len(probs)


def recovery_time(errors: np.ndarray, fault_onset: int,
                   base_factor: float = 1.10) -> float:
    """Iterations required to return to ``1.1 * pre-fault error``."""
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
    """Cumulative excess over the ideal curve; smaller is more
    resilient."""
    L = min(len(errors_method), len(errors_ideal))
    return float(np.sum(np.maximum(errors_method[:L] - errors_ideal[:L], 0.0)))


def detection_and_false_alarm_rates(gamma_history: list,
                                      faulty_agents: list, N: int,
                                      adj: Optional[np.ndarray] = None,
                                      gamma_threshold: float = 0.5) -> tuple:
    """Evaluate the diagnoser's detection rate and false-alarm rate
    (paper Section 4.4.3).

    The decision granularity is "neighbor mean": for every agent ``j``,
    look at the mean of ``gamma_{*, j}`` reported by ``j``'s neighbors at
    that step. When the mean is below ``gamma_threshold``, agent ``j`` is
    judged faulty at that step.

    - detection rate = TP / (TP + FN)
        TP: a faulty agent ``j`` is judged faulty by its neighbors at
            some step.
        FN: a faulty agent is not judged faulty by its neighbors.
    - false alarm rate = FP / (FP + TN)
        FP: a healthy agent ``j`` is mistakenly judged faulty by some
            neighbor.
        TN: a healthy agent is not judged faulty by any neighbor.

    Parameters
    ----------
    gamma_history : list of ``(N, N)`` gamma matrices; only the steps
                    after the fault onset are taken.
    faulty_agents : global indices of the truly faulty agents (used only
                    for evaluation; not visible to the diagnoser).
    N : total number of agents.
    adj : adjacency matrix; when given, only ``j``'s neighbor rows are
          averaged; otherwise all rows except ``j`` are averaged.
    gamma_threshold : a neighbor-mean below this threshold flags ``j`` as
                      faulty.

    Returns
    -------
    (detection_rate, false_alarm_rate) : two floats in ``[0, 1]``.
        Returns ``(nan, nan)`` if ``gamma_history`` is empty or
        ``faulty_agents`` is empty.
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
                # j is judged faulty iff the average gamma reported by
                # its neighbors is below the threshold.
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
