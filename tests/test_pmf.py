"""PMF 数据结构与基础 PMF 操作的单元测试。"""

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
    """测试辅助：从 events + mass 构造 PMF。"""
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
    """frozen=True dataclass 应阻止字段赋值。"""
    import dataclasses
    p = PMF(events=((0,),), mass=np.array([1.0]),
            masks=np.array([1], dtype=np.int64))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.events = ((1,),)   # type: ignore[misc]


# ---------------------------------------------------------------------------
# Singleton marginalization
# ---------------------------------------------------------------------------

def test_singleton_vector_pure_singleton():
    """PMF 全是单点事件时，singleton 概率 = mass。"""
    scope = [0, 1, 2]
    p = _make_pmf([(0,), (1,), (2,)], [0.5, 0.3, 0.2], scope)
    v = pmf_to_singleton_vector(p, scope)
    np.testing.assert_allclose(v, [0.5, 0.3, 0.2], atol=1e-12)


def test_singleton_vector_multi_event():
    """事件 (0, 1) 0.4 + (2,) 0.6 → singleton 应为 [0.2, 0.2, 0.6]。"""
    scope = [0, 1, 2]
    p = _make_pmf([(0, 1), (2,)], [0.4, 0.6], scope)
    v = pmf_to_singleton_vector(p, scope)
    # (0, 1) 平分 0.4 → 各 0.2；(2,) 给 2 → 0.6
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
    """均匀 4 事件 PMF 的熵 = log 4。"""
    scope = [0, 1, 2, 3]
    p = _make_pmf([(0,), (1,), (2,), (3,)], [0.25] * 4, scope)
    assert abs(pmf_entropy(p) - np.log(4)) < 1e-10


def test_entropy_certain():
    """单事件 PMF 的熵 = 0。"""
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
    """投影到子作用域时，落在外部的元素被剔除。"""
    scope = [0, 1, 2, 3]
    p = _make_pmf([(0, 3), (1,), (2, 3)], [0.4, 0.3, 0.3], scope)
    target = [0, 1, 2]
    pp = project_pmf(p, target)
    # (0, 3) → (0,)；(1,) 保留；(2, 3) → (2,)
    expected_events = {(0,), (1,), (2,)}
    assert set(pp.events) == expected_events
    np.testing.assert_allclose(pp.mass.sum(), 1.0, atol=1e-10)


def test_project_returns_empty_when_no_overlap():
    scope = [0, 1, 2]
    p = _make_pmf([(0,), (1,)], [0.5, 0.5], scope)
    target = [9, 10]
    pp = project_pmf(p, target)
    assert pp.is_empty
