"""Equation-level checks for LOS (left orthogonal sum) and DS (symmetric
combination)."""

import numpy as np

from config import PMF
from rps_diagnosis import (
    _bit_index_map,
    _mask_for,
    _to_mask_array,
    dempster_shafer_combination,
    left_intersection,
    left_orthogonal_sum,
)


def _make_pmf(events, mass, scope):
    a2b, _ = _bit_index_map(scope)
    masks = _to_mask_array([_mask_for(A, a2b) for A in events])
    return PMF(events=tuple(events), mass=np.asarray(mass), masks=masks)


# ---------------------------------------------------------------------------
# left_intersection primitive
# ---------------------------------------------------------------------------

def test_left_intersection_keeps_a_order():
    assert left_intersection((1, 2, 3), (3, 2)) == (2, 3)
    assert left_intersection((3, 1, 2), (2, 3)) == (3, 2)


def test_left_intersection_empty_when_disjoint():
    assert left_intersection((1, 2), (3, 4)) == ()


# ---------------------------------------------------------------------------
# Identity & idempotence laws
# ---------------------------------------------------------------------------

def test_los_with_self_concentrates_top1():
    """``LOS(A, A)`` concentrates mass on the intersections inside ``A``;
    combining with self should preserve normalization."""
    scope = [0, 1, 2]
    p = _make_pmf([(0, 1), (2,)], [0.6, 0.4], scope)
    out = left_orthogonal_sum(p, p, scope, top_m=16)
    np.testing.assert_allclose(out.mass.sum(), 1.0, atol=1e-10)


def test_los_normalization_after_conflict():
    """LOS output mass must be normalized even when there is conflict."""
    scope = [0, 1]
    a = _make_pmf([(0,), (1,)], [0.5, 0.5], scope)
    b = _make_pmf([(0,), (1,)], [0.7, 0.3], scope)
    out = left_orthogonal_sum(a, b, scope, top_m=16)
    np.testing.assert_allclose(out.mass.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Ordering preservation: LOS keeps A's order; DS doesn't
# ---------------------------------------------------------------------------

def test_los_preserves_order_of_first_argument():
    """A = (2, 0, 1), B = (1, 0, 2) -> left intersection = (2, 0, 1)."""
    scope = [0, 1, 2]
    a = _make_pmf([(2, 0, 1)], [1.0], scope)
    b = _make_pmf([(1, 0, 2)], [1.0], scope)
    out = left_orthogonal_sum(a, b, scope, top_m=16)
    assert out.events[0] == (2, 0, 1)


def test_ds_outputs_sorted_set():
    """DS uses set intersection; the output is sorted in ascending order
    and is independent of ``A``'s order."""
    scope = [0, 1, 2]
    a = _make_pmf([(2, 0, 1)], [1.0], scope)
    b = _make_pmf([(1, 0, 2)], [1.0], scope)
    out = dempster_shafer_combination(a, b, scope, top_m=16)
    assert out.events[0] == (0, 1, 2)


def test_los_vs_ds_differ_when_orders_disagree():
    """For the same input pair, the LOS and DS event tuples should differ
    (order-sensitive vs. order-insensitive)."""
    scope = [0, 1, 2]
    a = _make_pmf([(2, 1)], [1.0], scope)
    b = _make_pmf([(1, 2)], [1.0], scope)
    out_los = left_orthogonal_sum(a, b, scope, top_m=16)
    out_ds = dempster_shafer_combination(a, b, scope, top_m=16)
    assert out_los.events[0] == (2, 1)   # keeps A's order
    assert out_ds.events[0] == (1, 2)    # ascending


# ---------------------------------------------------------------------------
# Conflict handling
# ---------------------------------------------------------------------------

def test_los_falls_back_to_uniform_on_total_conflict():
    """When two PMFs are fully exclusive, LOS falls back to a uniform
    singleton PMF."""
    scope = [0, 1]
    a = _make_pmf([(0,)], [1.0], scope)
    b = _make_pmf([(1,)], [1.0], scope)
    out = left_orthogonal_sum(a, b, scope, top_m=16)
    np.testing.assert_allclose(out.mass.sum(), 1.0, atol=1e-10)
    # Falls back to uniform over ``((0,), (1,))``.
    assert set(out.events) == {(0,), (1,)}


def test_los_with_empty_returns_other():
    scope = [0, 1]
    a = _make_pmf([(0,)], [1.0], scope)
    empty = PMF.empty()
    out1 = left_orthogonal_sum(a, empty, scope, top_m=16)
    out2 = left_orthogonal_sum(empty, a, scope, top_m=16)
    assert out1.events == a.events
    assert out2.events == a.events


# ---------------------------------------------------------------------------
# Top-m truncation
# ---------------------------------------------------------------------------

def test_los_top_m_truncates_low_mass_events():
    """Force LOS to produce many result events and check that ``top_m``
    truncates them and that the mass is renormalized."""
    scope = list(range(5))
    events_a = [(i,) for i in range(5)]
    a = _make_pmf(events_a, [0.2] * 5, scope)
    events_b = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    b = _make_pmf(events_b, [0.2] * 5, scope)
    out = left_orthogonal_sum(a, b, scope, top_m=3)
    assert len(out) <= 3
    np.testing.assert_allclose(out.mass.sum(), 1.0, atol=1e-10)
