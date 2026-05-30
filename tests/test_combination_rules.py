"""LOS（左正交和）与 DS（对称组合）的等式验证。"""

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
    """LOS(A, A) 把质量集中到 A 内部交集；自身组合应保持归一性。"""
    scope = [0, 1, 2]
    p = _make_pmf([(0, 1), (2,)], [0.6, 0.4], scope)
    out = left_orthogonal_sum(p, p, scope, top_m=16)
    np.testing.assert_allclose(out.mass.sum(), 1.0, atol=1e-10)


def test_los_normalization_after_conflict():
    """LOS 输出质量必须归一化，即使有冲突。"""
    scope = [0, 1]
    a = _make_pmf([(0,), (1,)], [0.5, 0.5], scope)
    b = _make_pmf([(0,), (1,)], [0.7, 0.3], scope)
    out = left_orthogonal_sum(a, b, scope, top_m=16)
    np.testing.assert_allclose(out.mass.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Ordering preservation: LOS keeps A's order; DS doesn't
# ---------------------------------------------------------------------------

def test_los_preserves_order_of_first_argument():
    """A=(2, 0, 1), B=(1, 0, 2) → left intersection = (2, 0, 1)。"""
    scope = [0, 1, 2]
    a = _make_pmf([(2, 0, 1)], [1.0], scope)
    b = _make_pmf([(1, 0, 2)], [1.0], scope)
    out = left_orthogonal_sum(a, b, scope, top_m=16)
    assert out.events[0] == (2, 0, 1)


def test_ds_outputs_sorted_set():
    """DS 用集合交集，输出按升序排列，与 A 的顺序无关。"""
    scope = [0, 1, 2]
    a = _make_pmf([(2, 0, 1)], [1.0], scope)
    b = _make_pmf([(1, 0, 2)], [1.0], scope)
    out = dempster_shafer_combination(a, b, scope, top_m=16)
    assert out.events[0] == (0, 1, 2)


def test_los_vs_ds_differ_when_orders_disagree():
    """同一对输入 LOS 与 DS 的事件元组应不同（顺序敏感 vs 不敏感）。"""
    scope = [0, 1, 2]
    a = _make_pmf([(2, 1)], [1.0], scope)
    b = _make_pmf([(1, 2)], [1.0], scope)
    out_los = left_orthogonal_sum(a, b, scope, top_m=16)
    out_ds = dempster_shafer_combination(a, b, scope, top_m=16)
    assert out_los.events[0] == (2, 1)   # 保留 A 的顺序
    assert out_ds.events[0] == (1, 2)    # 升序


# ---------------------------------------------------------------------------
# Conflict handling
# ---------------------------------------------------------------------------

def test_los_falls_back_to_uniform_on_total_conflict():
    """两个 PMF 完全互斥时 LOS 退回均匀单点 PMF。"""
    scope = [0, 1]
    a = _make_pmf([(0,)], [1.0], scope)
    b = _make_pmf([(1,)], [1.0], scope)
    out = left_orthogonal_sum(a, b, scope, top_m=16)
    np.testing.assert_allclose(out.mass.sum(), 1.0, atol=1e-10)
    # 退回到 ((0,), (1,)) 均匀
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
    """让 LOS 产生很多结果事件，验证 top_m 截断且质量重新归一。"""
    scope = list(range(5))
    events_a = [(i,) for i in range(5)]
    a = _make_pmf(events_a, [0.2] * 5, scope)
    events_b = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    b = _make_pmf(events_b, [0.2] * 5, scope)
    out = left_orthogonal_sum(a, b, scope, top_m=3)
    assert len(out) <= 3
    np.testing.assert_allclose(out.mass.sum(), 1.0, atol=1e-10)
