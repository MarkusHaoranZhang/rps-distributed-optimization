"""
Centralized configuration + core data structures
=================================================

All tuning knobs are gathered into ``RPSConfig``. ``run_optimization`` accepts
an ``RPSConfig`` instance instead of a long list of loose arguments; anyone who
wants to reproduce a particular experiment only has to save this single object.

The PMF triple has been promoted from ``Tuple[tuple, ndarray, ndarray]`` to a
dataclass, gaining named fields, type checking, and room to grow (e.g. caching
singleton vectors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# Permutation Mass Function — replaces the old (events, mass, masks) tuple
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PMF:
    """Truncated mass function on a Random Permutation Set.

    Attributes
    ----------
    events : tuple[tuple[int, ...], ...]
        Ordered tuple of events; each event is an ordered tuple of agent
        indices.
    mass : np.ndarray
        Shape ``(E,)``. Mass associated with each event; sums to 1 unless the
        PMF is empty.
    masks : np.ndarray
        Shape ``(E,)``. Bitmask of event members. ``int64`` when the scope
        size is ≤ 63, otherwise falls back to ``object`` dtype to hold Python
        big integers.
    """

    events: tuple
    mass: np.ndarray
    masks: np.ndarray

    def __len__(self) -> int:
        return len(self.events)

    @property
    def is_empty(self) -> bool:
        return len(self.events) == 0

    @classmethod
    def empty(cls) -> "PMF":
        return cls(events=(), mass=np.empty(0),
                   masks=np.empty(0, dtype=np.int64))


# ---------------------------------------------------------------------------
# Fault configuration schema
# ---------------------------------------------------------------------------

class FaultConfig(TypedDict, total=False):
    """Schema for the fault configuration accepted by ``apply_fault_injection``.

    .. note::
       This ``TypedDict`` is **for documentation only**. At runtime, all call
       sites (``apply_fault_injection`` / ``run_optimization`` /
       ``validate_fault_config``) still type their parameter as ``dict`` because
       fault_config is constructed with literal dicts for convenience.
       ``validate_fault_config`` performs schema checking at runtime (missing
       keys, wrong types, out-of-range values), so schema violations cannot
       slip through silently.

       Switching every call site to ``FaultConfig`` would give mypy-level
       enforcement, but it is not needed for the paper companion code; runtime
       validation already covers every failure mode.

    Required
    --------
    onset : int
        Iteration step at which fault injection starts.
    agents : list[int]
        Global indices of faulty agents (may be empty, meaning fault-free).
    type : str
        One of ``"constant"`` / ``"drift"`` / ``"intermittent"``.

    Optional / type-specific
    ------------------------
    delta : np.ndarray | None
        Base offset of the fault, shape ``(d,)``. Used by ``constant`` /
        ``drift`` / ``intermittent``.
    drift_cap : float
        Used only when ``type="drift"``: drift saturates after growing to this
        multiple (default 100.0). The small-fault regime assumed in
        Section 4.4 of the paper requires bounded offsets; this is the
        engineering trick that turns an unbounded linear drift into a bounded
        ramp.
    prob : float
        Used only when ``type="intermittent"``: per-step probability of
        triggering the fault.
    """

    onset: int
    agents: list
    type: str
    delta: Optional[np.ndarray]
    drift_cap: float
    prob: float


# ---------------------------------------------------------------------------
# RPSConfig — every tuning knob lives here
# ---------------------------------------------------------------------------

@dataclass
class RPSConfig:
    """All tuning parameters for RPS diagnosis + soft discounting.

    Mapping to paper equations
    --------------------------
    - h_hop / k_trunc : diagnosable scope and truncation order from
                        Section 4.1.
    - window_len      : sliding window length s from Section 4.1.
    - eta             : softmax temperature in Eq. (9).
    - top_m           : number of top events kept after PMF truncation;
                        engineering knob.
    - top_agents_k    : two-stage event-enumeration pruning parameter inside
                        ``compute_pmf``; engineering knob.
    - gain            : sensitivity of the continuous soft discount
                        γ = exp(-gain · P_OPT). Continuous relaxation of
                        Eq. (12); ``gain`` is an added engineering parameter.
    - tau_quantile    : Section 4.3 calibrates τ as the requested quantile of
                        the entropy distribution during the fault-free phase.
    - diagnose_every  : engineering throttle: how often diagnosis is
                        recomputed. Not present in the paper equations; the
                        LOS step is the O(E²) bottleneck.
    - proxy_std_weight : magnitude_proxy = max(mean_inc, weight · std_inc).
                        Controls the weight of the std increment in the
                        fault-source proxy; engineering knob.
    - proxy_global_weight : mixing coefficient ``contrib + weight · proxy``
                        inside ``compute_pmf``; injects a global proxy into the
                        per-agent local signal. Not present in the paper.

    None of these "engineering parameters" belong to the paper's core
    methodology, but they are necessary under our experimental setup. See
    ``IMPLEMENTATION_NOTES.md`` for details.
    """

    # ----- core (paper) -----
    h_hop: int = 2
    k_trunc: int = 3
    window_len: int = 20
    eta: float = 1.0
    burn_in: int = 100
    tau: Optional[float] = None
    """Explicit τ; ``None`` means "calibrate from the entropy distribution at
    the end of burn-in using ``tau_quantile``"."""
    tau_quantile: float = 0.95
    """τ calibration: when ``tau`` is None, take this quantile of the
    fault-free entropy distribution (paper Section 4.3)."""

    # ----- engineering knobs -----
    top_m: int = 16
    top_agents_k: int = 5
    gain: float = 4.0
    diagnose_every: int = 5
    proxy_std_weight: float = 2.0
    proxy_global_weight: float = 0.5

    # ----- baseline-related -----
    uniform_factor: float = 0.9      # fixed discount used by Uniform-Discount
    chi2_confidence: float = 0.99    # chi-square confidence for Hard-Threshold

    # ----- ablation switches -----
    record_diagnosis: bool = True
    record_agent_idx: int = 0        # which agent's view the diagnosis log records

    def __post_init__(self) -> None:
        """Validate parameters at construction time so invalid configs fail
        at the source rather than mid-run."""
        problems: list[str] = []
        if self.h_hop < 1:
            problems.append(f"h_hop must be >= 1, got {self.h_hop}")
        if self.k_trunc < 1:
            problems.append(f"k_trunc must be >= 1, got {self.k_trunc}")
        if self.window_len < 1:
            problems.append(f"window_len must be >= 1, got {self.window_len}")
        if self.eta <= 0:
            problems.append(f"eta must be > 0, got {self.eta}")
        if self.burn_in < self.window_len:
            problems.append(
                f"burn_in ({self.burn_in}) must be >= window_len ({self.window_len}) "
                f"(otherwise no full sliding window is available for calibration "
                f"before the fault period)"
            )
        if not (0.0 < self.tau_quantile < 1.0):
            problems.append(f"tau_quantile must be in (0, 1), got {self.tau_quantile}")
        if self.top_m < 1:
            problems.append(f"top_m must be >= 1, got {self.top_m}")
        if self.top_agents_k < 1:
            problems.append(f"top_agents_k must be >= 1, got {self.top_agents_k}")
        if self.gain < 0:
            problems.append(f"gain must be >= 0, got {self.gain}")
        if self.diagnose_every < 1:
            problems.append(f"diagnose_every must be >= 1, got {self.diagnose_every}")
        if self.proxy_std_weight < 0:
            problems.append(f"proxy_std_weight must be >= 0, got {self.proxy_std_weight}")
        if self.proxy_global_weight < 0:
            problems.append(f"proxy_global_weight must be >= 0, got {self.proxy_global_weight}")
        if not (0.0 <= self.uniform_factor <= 1.0):
            problems.append(f"uniform_factor must be in [0, 1], got {self.uniform_factor}")
        if not (0.0 < self.chi2_confidence < 1.0):
            problems.append(f"chi2_confidence must be in (0, 1), got {self.chi2_confidence}")
        if self.record_agent_idx < 0:
            problems.append(f"record_agent_idx must be >= 0, got {self.record_agent_idx}")
        if problems:
            raise ValueError("Invalid RPSConfig:\n  - " + "\n  - ".join(problems))

    def replace(self, **changes) -> "RPSConfig":
        """Return a copy with the given fields overridden (handy for
        parameter sensitivity sweeps)."""
        from dataclasses import replace
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

KNOWN_METHODS: tuple = (
    "Ideal",
    "Hard-Threshold",
    "Uniform-Discount",
    "Byzantine-Resilient",
    "RPS-Symmetric",
    "RPS-NoOrder",
    "RPS-Full",
)
"""All supported method names (paper Section 4.4.2).

When adding / renaming / reordering, the following must be kept in sync:

- ``main._FIG2_METHODS`` (the subset listed in Figure 2 / Table 2; currently
  excludes RPS-NoOrder because it appears separately in Figure 6).
- The method dispatch in step 3 of ``experiments.run_optimization``
  (HT / Uniform / RPS / Byzantine).
- ``tests/test_run_optimization.py::test_each_method_runs`` covers everything
  automatically via ``parametrize(list(KNOWN_METHODS))``, so new methods are
  picked up by the smoke test for free.
"""

RPS_METHODS: tuple = ("RPS-Full", "RPS-Symmetric", "RPS-NoOrder")


def is_rps_method(method: str) -> bool:
    return method in RPS_METHODS


# ---------------------------------------------------------------------------
# Runtime validation for the fault_config dict
# ---------------------------------------------------------------------------

_VALID_FAULT_TYPES = ("constant", "drift", "intermittent")


def validate_fault_config(fault_config: dict, *, d: Optional[int] = None) -> None:
    """Check that ``fault_config`` matches the ``FaultConfig`` schema.

    Raises ``ValueError`` with a list of problems if any required key is
    missing, has the wrong type, or is out of range. Called at the entry
    point of ``run_optimization`` so format errors surface at the source.

    Parameters
    ----------
    fault_config : dict to validate
    d            : if given, also checks ``delta.shape == (d,)``
    """
    problems: list[str] = []
    if 'onset' not in fault_config:
        problems.append("missing required key 'onset'")
    elif not isinstance(fault_config['onset'], int) or fault_config['onset'] < 0:
        problems.append(f"'onset' must be non-negative int, got {fault_config['onset']!r}")

    if 'agents' not in fault_config:
        problems.append("missing required key 'agents'")
    elif not isinstance(fault_config['agents'], (list, tuple)):
        problems.append(f"'agents' must be list/tuple, got {type(fault_config['agents']).__name__}")

    if 'type' not in fault_config:
        problems.append("missing required key 'type'")
    elif fault_config['type'] not in _VALID_FAULT_TYPES:
        problems.append(
            f"'type' must be one of {_VALID_FAULT_TYPES}, got {fault_config['type']!r}"
        )

    ftype = fault_config.get('type')
    if ftype in ('constant', 'drift', 'intermittent'):
        if fault_config.get('agents') and fault_config.get('delta') is None:
            problems.append(f"'delta' must be provided for type='{ftype}' with agents")
        elif d is not None and fault_config.get('delta') is not None:
            delta = np.asarray(fault_config['delta'])
            if delta.shape != (d,):
                problems.append(
                    f"'delta' shape {delta.shape} does not match decision dim ({d},)"
                )
    if ftype == 'intermittent':
        if 'prob' not in fault_config:
            problems.append("type='intermittent' requires 'prob' key")
        elif not (0.0 <= float(fault_config['prob']) <= 1.0):
            problems.append(f"'prob' must be in [0, 1], got {fault_config['prob']!r}")
    if ftype == 'drift' and 'drift_cap' in fault_config:
        if float(fault_config['drift_cap']) <= 0:
            problems.append(f"'drift_cap' must be > 0, got {fault_config['drift_cap']!r}")

    if problems:
        raise ValueError("Invalid fault_config:\n  - " + "\n  - ".join(problems))
