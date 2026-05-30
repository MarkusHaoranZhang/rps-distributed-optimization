"""``statistics_utils`` 的单元测试。"""

import numpy as np

from statistics_utils import cohens_d, holm_bonferroni, wilcoxon_pvalue


def test_cohens_d_zero_when_identical():
    x = np.array([1.0, 2.0, 3.0])
    # 配对相同 → diff 全为 0；返回 0
    assert cohens_d(x, x) == 0.0


def test_cohens_d_positive_when_y_larger():
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    y = x + 1.0
    d = cohens_d(y, x)
    assert d > 0.5


def test_holm_bonferroni_monotone():
    """Holm 校正后的 p 值序列保持原始 p 的单调性。"""
    raw = [0.001, 0.01, 0.04, 0.05]
    adj = holm_bonferroni(raw)
    # 输入升序时校正值也应非递减（Holm 性质）
    assert all(adj[i] <= adj[i + 1] + 1e-12 for i in range(len(adj) - 1))


def test_holm_bonferroni_caps_at_one():
    raw = [0.5, 0.6, 0.7]
    adj = holm_bonferroni(raw)
    assert all(p <= 1.0 for p in adj)


def test_wilcoxon_returns_one_for_identical():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    p = wilcoxon_pvalue(x, x)
    assert p == 1.0


def test_wilcoxon_small_for_clear_difference():
    rng = np.random.RandomState(0)
    x = rng.randn(15)
    y = x + 2.0
    p = wilcoxon_pvalue(x, y)
    assert p < 0.01
