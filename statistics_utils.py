"""
Statistical helpers: Wilcoxon signed-rank + Holm-Bonferroni + Cohen's d
=======================================================================
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def cohens_d(x, y) -> float:
    """Paired Cohen's d (using the standard deviation of the differences).

    Edge cases: when the difference variance is 0 -- if the mean is also 0
    we return 0; if the mean is non-zero the effect size is infinite, so
    we return ``+/-inf`` rather than a misleading 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    sd = diff.std(ddof=1)
    mean = diff.mean()
    if sd == 0:
        if mean == 0:
            return 0.0
        return float('inf') if mean > 0 else float('-inf')
    return float(mean / sd)


def wilcoxon_pvalue(x, y) -> float:
    """Two-sided Wilcoxon signed-rank test. ``x`` and ``y`` are paired
    samples of equal length."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.allclose(x, y):
        return 1.0
    try:
        _, p = wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
    except ValueError:
        return 1.0
    return float(p)


def holm_bonferroni(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, returned in the original input
    order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted
