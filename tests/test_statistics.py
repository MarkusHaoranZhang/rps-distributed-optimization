"""Unit tests for ``statistics_utils``."""

import numpy as np

from statistics_utils import cohens_d, holm_bonferroni, wilcoxon_pvalue


def test_cohens_d_zero_when_identical():
    x = np.array([1.0, 2.0, 3.0])
    # Pairing with itself -> all diffs are 0; return 0.
    assert cohens_d(x, x) == 0.0


def test_cohens_d_positive_when_y_larger():
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    y = x + 1.0
    d = cohens_d(y, x)
    assert d > 0.5


def test_holm_bonferroni_monotone():
    """The Holm-corrected p-values preserve the monotonicity of the
    raw p-values."""
    raw = [0.001, 0.01, 0.04, 0.05]
    adj = holm_bonferroni(raw)
    # When the input is in ascending order, the corrected values are
    # also non-decreasing (a property of Holm's procedure).
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
