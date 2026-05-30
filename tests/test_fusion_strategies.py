"""三个 fusion 策略的核心性质测试。

对应论文 Section 4.2 (directional fusion via LOS) 与 Section 4.4.2
(RPS-Symmetric / RPS-NoOrder 消融对照)。

之前覆盖率盲区：``directional_fusion`` / ``symmetric_fusion`` /
``noorder_fusion`` 三个函数都没有单元测试。本文件补齐。
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
# directional_fusion 核心性质：与 symmetric / noorder 实质不同
# ---------------------------------------------------------------------------

def test_directional_distinct_from_symmetric():
    """``directional_fusion`` 与 ``symmetric_fusion`` 在多元有序事件上分化。

    用 self=(2,1,3), nb=(3,1,2) 这种共享元素但顺序不同的输入：
    - LOS 保留 self 顺序 → (2,1,3)
    - DS 用集合交集 → sorted (1,2,3)
    若两者结果相同，说明 directional 实际退化为 symmetric——RPS 论文
    Section 4.2 的 directional aggregation 不成立。
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
    """RPS-Symmetric 必须**完全用 DS**（论文 Section 4.4.2 字面定义）。

    构造让 LOS 与 DS 在第一步就分化的输入：self=(2,1), nb=(1,2)。
    LOS 输出顺序 (2,1)，DS 输出 sorted (1,2)。symmetric_fusion 应给 (1,2)。
    """
    scope = [0, 1, 2]
    self_p = _make([(2, 1)], [1.0], scope)
    nb = _make([(1, 2)], [1.0], scope)
    out = symmetric_fusion(self_p, [nb], scope, top_m=16)
    assert out.events[0] == (1, 2), (
        f"symmetric_fusion should be DS-based (sorted output), got {out.events[0]}"
    )


def test_noorder_collapses_before_fusion():
    """RPS-NoOrder 必须先把 (a,b) 与 (b,a) 折叠，再做 DS（论文 4.4.2）。

    self 含 (1, 0) 与 (0, 1) 两个有序事件 → 折叠后变成单点 (0, 1) 质量 1.0。
    """
    scope = [0, 1, 2]
    # 同一对 agent 的两种顺序，折叠后应合并
    self_p = _make([(1, 0), (0, 1)], [0.5, 0.5], scope)
    nb = _make([(0, 1)], [1.0], scope)
    out = noorder_fusion(self_p, [nb], scope, top_m=16)
    # 折叠后 self 只剩 (0, 1) 一个 sorted 事件，与 nb 一致 → DS 单点
    assert out.events == ((0, 1),)
    np.testing.assert_allclose(out.mass, [1.0], atol=1e-10)


def test_directional_preserves_self_order_in_pure_intersection():
    """``directional_fusion`` 必须保留 self PMF 的事件顺序。

    self=(1,0), neighbor=(0,1) → directional 输出 (1,0)（self 顺序），
    noorder 折叠后输出 (0,1)（论文 4.4.2 字面定义：collapsing before fusion）。
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
    """空邻居列表时 directional/symmetric/noorder 都返回 self。"""
    scope = [0, 1, 2]
    self_p = _make([(0, 1), (2,)], [0.6, 0.4], scope)
    for fn in (directional_fusion, symmetric_fusion, noorder_fusion):
        out = fn(self_p, [], scope, top_m=16)
        assert out.events == self_p.events
        np.testing.assert_allclose(out.mass, self_p.mass)


# ---------------------------------------------------------------------------
# directional 与 LOS 的关系：序贯 LOS 的总和
# ---------------------------------------------------------------------------

def test_directional_with_one_neighbor_equals_los():
    """``directional_fusion`` with 1 neighbor 等价于 ``LOS(self, neighbor)``。"""
    scope = [0, 1, 2]
    self_p = _make([(0, 1)], [1.0], scope)
    nb = _make([(1, 2)], [1.0], scope)
    d = directional_fusion(self_p, [nb], scope, top_m=16)
    los = left_orthogonal_sum(self_p, nb, scope, top_m=16)
    assert d.events == los.events
    np.testing.assert_allclose(d.mass, los.mass)


# ---------------------------------------------------------------------------
# symmetric_fusion 性质：DS 是源对称（pair-wise 对称）
# ---------------------------------------------------------------------------

def test_symmetric_fusion_pairwise_ds_symmetry():
    """二元 DS 对源对称：DS(a, b) 与 DS(b, a) 在 (event, mass) 上集合相同。

    DS 链对邻居列表的顺序在 ≥ 2 邻居时不再交换（这是 DS 的已知性质，
    论文 RPS-Symmetric 的"symmetric"指的是 pair-wise 源对称，不要求
    多邻居链对列表顺序不变）。本测试只锁 pair-wise 源对称。
    """
    scope = [0, 1, 2]
    self_p = _make([(0,)], [1.0], scope)
    nb = _make([(1, 0)], [1.0], scope)
    out_ab = symmetric_fusion(self_p, [nb], scope, top_m=16)
    out_ba = symmetric_fusion(nb, [self_p], scope, top_m=16)
    # 转 dict 比较（事件元组顺序在 dict 里无意义）
    d_ab = dict(zip(out_ab.events, out_ab.mass))
    d_ba = dict(zip(out_ba.events, out_ba.mass))
    assert set(d_ab.keys()) == set(d_ba.keys())
    for key in d_ab:
        np.testing.assert_allclose(d_ab[key], d_ba[key], atol=1e-10)


# ---------------------------------------------------------------------------
# noorder_fusion 性质：折叠 + DS 链
# ---------------------------------------------------------------------------

def test_noorder_uses_set_intersection_underneath():
    """``noorder_fusion`` 在两个无歧义有序事件上等价于 DS。"""
    scope = [0, 1, 2]
    self_p = _make([(2, 1)], [1.0], scope)
    nb = _make([(1, 2)], [1.0], scope)
    n = noorder_fusion(self_p, [nb], scope, top_m=16)
    # 两个事件折叠后都是 (1, 2)，DS 后仍是 (1, 2)
    assert n.events[0] == (1, 2)


# ---------------------------------------------------------------------------
# Mass conservation across all three fusions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fusion_fn", [
    directional_fusion, symmetric_fusion, noorder_fusion,
])
def test_fusion_mass_sums_to_one(fusion_fn):
    """三种 fusion 输出的质量都必须归一化。"""
    scope = [0, 1, 2, 3]
    self_p = _make([(0, 1), (2,), (3,)], [0.5, 0.3, 0.2], scope)
    nb_a = _make([(1, 0), (3,)], [0.6, 0.4], scope)
    nb_b = _make([(2, 3), (1,)], [0.5, 0.5], scope)
    out = fusion_fn(self_p, [nb_a, nb_b], scope, top_m=16)
    assert not out.is_empty
    np.testing.assert_allclose(out.mass.sum(), 1.0, atol=1e-10)
