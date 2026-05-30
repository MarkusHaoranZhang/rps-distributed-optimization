"""Unit tests for the PMF data structure and basic PMF operations."""

import numpy as np
import pytest

from config import PMF
from rps_diagnosis import (
    _bit_index_map,
    _mask_for,
    _to_mask_array,
    pmf_entropy,
    pmf_to_singleton_vector,
    project_pmf,
)


def _make_pmf(events, mass, scope):
    """Test helper: build a PMF from ``events`` and ``mass``."""
    a2b, _ = _bit_index_map(scope)
    masks = _to_mask_array([_mask_for(A, a2b) for A in events])
    return PMF(events=tuple(events), mass=np.asarray(mass), masks=masks)


# ---------------------------------------------------------------------------
# Basic dataclass behavior
# ---------------------------------------------------------------------------

def test_empty_pmf():
    p = PMF.empty()
    assert p.is_empty
    assert len(p) == 0


def test_pmf_is_immutable():
    """``frozen=True`` should prevent field assignment."""
    import dataclasses
    p = PMF(events=((0,),), mass=np.array([1.0]),
            masks=np.array([1], dtype=np.int64))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.events = ((1,),)   # type: ignore[misc]


# ---------------------------------------------------------------------------
# Singleton marginalization
# ---------------------------------------------------------------------------

def test_singleton_vector_pure_singleton():
    """When every event is a singleton, the singleton probability
    equals the mass."""
    scope = [0, 1, 2]
    p = _make_pmf([(0,), (1,), (2,)], [0.5, 0.3, 0.2], scope)
    v = pmf_to_singleton_vector(p, scope)
    np.testing.assert_allclose(v, [0.5, 0.3, 0.2], atol=1e-12)


def test_singleton_vector_multi_event():
    """Events ``(0, 1) -> 0.4`` and ``(2,) -> 0.6`` give the singleton
    vector ``[0.2, 0.2, 0.6]``."""
    scope = [0, 1, 2]
    p = _make_pmf([(0, 1), (2,)], [0.4, 0.6], scope)
    v = pmf_to_singleton_vector(p, scope)
    # ``(0, 1)`` splits 0.4 evenly -> 0.2 each; ``(2,)`` gives 0.6 to
    # agent 2.
    np.testing.assert_allclose(v, [0.2, 0.2, 0.6], atol=1e-12)


def test_singleton_vector_sums_to_one():
    scope = [0, 1, 2, 3]
    p = _make_pmf([(0, 1, 2), (1, 3), (2,)], [0.5, 0.3, 0.2], scope)
    v = pmf_to_singleton_vector(p, scope)
    np.testing.assert_allclose(v.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def test_entropy_uniform():
    """A 4-event uniform PMF has entropy ``log 4``."""
    scope = [0, 1, 2, 3]
    p = _make_pmf([(0,), (1,), (2,), (3,)], [0.25] * 4, scope)
    assert abs(pmf_entropy(p) - np.log(4)) < 1e-10


def test_entropy_certain():
    """A singleton-event PMF has entropy 0."""
    scope = [0, 1]
    p = _make_pmf([(0,)], [1.0], scope)
    assert pmf_entropy(p) < 1e-10


def test_entropy_nonneg():
    scope = [0, 1, 2]
    p = _make_pmf([(0,), (1, 2)], [0.7, 0.3], scope)
    assert pmf_entropy(p) > 0


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def test_project_drops_outside_target():
    """When projecting to a sub-scope, elements outside the target are
    dropped."""
    scope = [0, 1, 2, 3]
    p = _make_pmf([(0, 3), (1,), (2, 3)], [0.4, 0.3, 0.3], scope)
    target = [0, 1, 2]
    pp = project_pmf(p, target)
    # ``(0, 3) -> (0,)``; ``(1,)`` is kept; ``(2, 3) -> (2,)``.
    expected_events = {(0,), (1,), (2,)}
    assert set(pp.events) == expected_events
    np.testing.assert_allclose(pp.mass.sum(), 1.0, atol=1e-10)


def test_project_returns_empty_when_no_overlap():
    scope = [0, 1, 2]
    p = _make_pmf([(0,), (1,)], [0.5, 0.5], scope)
    target = [9, 10]
    pp = project_pmf(p, target)
    assert pp.is_empty
