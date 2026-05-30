"""Core-property tests for the three fusion strategies.

These map to paper Section 4.2 (directional fusion via LOS) and
Section 4.4.2 (the RPS-Symmetric / RPS-NoOrder ablation pair).

Coverage gap closed: ``directional_fusion`` / ``symmetric_fusion`` /
``noorder_fusion`` previously had no unit tests. This file fills that in.
"""

import numpy as np
import pytest

from config import PMF
from rps_diagnosis import (
    _bit_index_map,
    _mask_for,
    _to_mask_array,
    directional_fusion,
    left_orthogonal_sum,
    noorder_fusion,
    symmetric_fusion,
)


def _make(events, mass, scope):
    a2b, _ = _bit_index_map(scope)
    masks = _to_mask_array([_mask_for(A, a2b) for A in events])
    return PMF(events=tuple(events), mass=np.asarray(mass), masks=masks)


# ---------------------------------------------------------------------------
# directional_fusion: it must differ substantively from symmetric / noorder
# ---------------------------------------------------------------------------

def test_directional_distinct_from_symmetric():
    """``directional_fusion`` and ``symmetric_fusion`` should diverge on
    multi-element ordered events.

    With ``self=(2, 1, 3)`` and ``nb=(3, 1, 2)`` (same elements, different
    order):
    - LOS keeps self's order -> ``(2, 1, 3)``.
    - DS uses set intersection -> sorted ``(1, 2, 3)``.
    If both produced the same result, ``directional_fusion`` would have
    silently degenerated into ``symmetric_fusion`` -- the directional
    aggregation in Section 4.2 of the paper would no longer hold.
    """
    scope = [0, 1, 2, 3]
    self_p = _make([(2, 1, 3)], [1.0], scope)
    nb = _make([(3, 1, 2)], [1.0], scope)

    d_out = directional_fusion(self_p, [nb], scope, top_m=16)
    s_out = symmetric_fusion(self_p, [nb], scope, top_m=16)

    assert (2, 1, 3) in d_out.events, (
        f"directional should preserve self order (2,1,3); got {d_out.events}"
    )
    assert (1, 2, 3) in s_out.events, (
        f"symmetric (pure DS) should give sorted (1,2,3); got {s_out.events}"
    )
    assert (2, 1, 3) not in s_out.events, (
        f"symmetric should not preserve self order; got {s_out.events}"
    )


def test_symmetric_uses_pure_ds_no_los():
    """RPS-Symmetric must use **DS only** (literal definition in paper
    Section 4.4.2).

    Construct an input where LOS and DS already differ at the very first
    step: ``self=(2, 1)`` and ``nb=(1, 2)``. LOS would output ``(2, 1)``,
    DS outputs sorted ``(1, 2)``. ``symmetric_fusion`` must produce
    ``(1, 2)``.
    """
    scope = [0, 1, 2]
    self_p = _make([(2, 1)], [1.0], scope)
    nb = _make([(1, 2)], [1.0], scope)
    out = symmetric_fusion(self_p, [nb], scope, top_m=16)
    assert out.events[0] == (1, 2), (
        f"symmetric_fusion should be DS-based (sorted output), got {out.events[0]}"
    )


def test_noorder_collapses_before_fusion():
    """RPS-NoOrder must collapse ``(a, b)`` and ``(b, a)`` into a single
    unordered event before the DS fusion (paper Section 4.4.2).

    ``self`` has both ordered events ``(1, 0)`` and ``(0, 1)``; after
    collapsing they merge into a single ``(0, 1)`` with mass 1.0.
    """
    scope = [0, 1, 2]
    # Two orderings of the same agent pair that should be collapsed
    # into one.
    self_p = _make([(1, 0), (0, 1)], [0.5, 0.5], scope)
    nb = _make([(0, 1)], [1.0], scope)
    out = noorder_fusion(self_p, [nb], scope, top_m=16)
    # After collapsing, ``self`` keeps a single sorted event ``(0, 1)``
    # which matches ``nb``, so DS yields a singleton.
    assert out.events == ((0, 1),)
    np.testing.assert_allclose(out.mass, [1.0], atol=1e-10)


def test_directional_preserves_self_order_in_pure_intersection():
    """``directional_fusion`` must preserve the event order of ``self``.

    With ``self=(1, 0)`` and ``neighbor=(0, 1)``, ``directional`` outputs
    ``(1, 0)`` (self's order), while ``noorder`` outputs ``(0, 1)`` after
    collapsing (the literal "collapsing before fusion" definition in
    Section 4.4.2).
    """
    scope = [0, 1]
    self_p = _make([(1, 0)], [1.0], scope)
    nb = _make([(0, 1)], [1.0], scope)

    d = directional_fusion(self_p, [nb], scope, top_m=16)
    n = noorder_fusion(self_p, [nb], scope, top_m=16)

    assert d.events[0] == (1, 0), (
        f"directional should preserve self order (1,0); got {d.events[0]}"
    )
    assert n.events[0] == (0, 1), (
        f"noorder should output collapsed sorted (0,1); got {n.events[0]}"
    )


def test_directional_with_no_neighbors_returns_self():
    """When the neighbor list is empty, all three fusions return
    ``self`` unchanged."""
    scope = [0, 1, 2]
    self_p = _make([(0, 1), (2,)], [0.6, 0.4], scope)
    for fn in (directional_fusion, symmetric_fusion, noorder_fusion):
        out = fn(self_p, [], scope, top_m=16)
        assert out.events == self_p.events
        np.testing.assert_allclose(out.mass, self_p.mass)


# ---------------------------------------------------------------------------
# Relation between directional and LOS: it is the chain of LOS calls.
# ---------------------------------------------------------------------------

def test_directional_with_one_neighbor_equals_los():
    """``directional_fusion`` with one neighbor is equivalent to
    ``LOS(self, neighbor)``."""
    scope = [0, 1, 2]
    self_p = _make([(0, 1)], [1.0], scope)
    nb = _make([(1, 2)], [1.0], scope)
    d = directional_fusion(self_p, [nb], scope, top_m=16)
    los = left_orthogonal_sum(self_p, nb, scope, top_m=16)
    assert d.events == los.events
    np.testing.assert_allclose(d.mass, los.mass)


# ---------------------------------------------------------------------------
# symmetric_fusion: DS is source-symmetric (pair-wise symmetric).
# ---------------------------------------------------------------------------

def test_symmetric_fusion_pairwise_ds_symmetry():
    """Pair-wise DS is source-symmetric: ``DS(a, b)`` and ``DS(b, a)``
    agree on the ``(event, mass)`` pairs.

    With >= 2 neighbors the DS chain is no longer commutative in the
    neighbor-list order (this is a known property of DS); the
    "symmetric" in RPS-Symmetric refers to pair-wise source symmetry
    and does not require chain order independence. This test pins down
    only the pair-wise property.
    """
    scope = [0, 1, 2]
    self_p = _make([(0,)], [1.0], scope)
    nb = _make([(1, 0)], [1.0], scope)
    out_ab = symmetric_fusion(self_p, [nb], scope, top_m=16)
    out_ba = symmetric_fusion(nb, [self_p], scope, top_m=16)
    # Compare via dicts (event-tuple order is not meaningful as a key).
    d_ab = dict(zip(out_ab.events, out_ab.mass))
    d_ba = dict(zip(out_ba.events, out_ba.mass))
    assert set(d_ab.keys()) == set(d_ba.keys())
    for key in d_ab:
        np.testing.assert_allclose(d_ab[key], d_ba[key], atol=1e-10)


# ---------------------------------------------------------------------------
# noorder_fusion: collapse + DS chain.
# ---------------------------------------------------------------------------

def test_noorder_uses_set_intersection_underneath():
    """On two unambiguous ordered events, ``noorder_fusion`` is
    equivalent to DS."""
    scope = [0, 1, 2]
    self_p = _make([(2, 1)], [1.0], scope)
    nb = _make([(1, 2)], [1.0], scope)
    n = noorder_fusion(self_p, [nb], scope, top_m=16)
    # Both events collapse to ``(1, 2)``; DS leaves it unchanged.
    assert n.events[0] == (1, 2)


# ---------------------------------------------------------------------------
# Mass conservation across all three fusions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fusion_fn", [
    directional_fusion, symmetric_fusion, noorder_fusion,
])
def test_fusion_mass_sums_to_one(fusion_fn):
    """All three fusion outputs must be normalized."""
    scope = [0, 1, 2, 3]
    self_p = _make([(0, 1), (2,), (3,)], [0.5, 0.3, 0.2], scope)
    nb_a = _make([(1, 0), (3,)], [0.6, 0.4], scope)
    nb_b = _make([(2, 3), (1,)], [0.5, 0.5], scope)
    out = fusion_fn(self_p, [nb_a, nb_b], scope, top_m=16)
    assert not out.is_empty
    np.testing.assert_allclose(out.mass.sum(), 1.0, atol=1e-10)
