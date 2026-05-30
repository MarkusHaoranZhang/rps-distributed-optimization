"""
统计工具：Wilcoxon signed-rank + Holm-Bonferroni + Cohen's d
============================================================
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def cohens_d(x, y) -> float:
    """配对 Cohen's d（用差值的标准差）。

    边界：差值方差为 0 时——若均值也为 0 返回 0；若均值非零，效应量是无穷，
    返回 ``±inf`` 而不是误导性的 0。
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
    """两侧 Wilcoxon signed-rank 检验。``x``, ``y`` 等长配对样本。"""
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
    """Holm-Bonferroni 校正后的 p 值（保持原顺序返回）。"""
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
