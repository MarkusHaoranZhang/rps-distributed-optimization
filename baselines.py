"""
基线方法（论文 Section 4.4.2）
==============================

- ``HardThresholdDetector``      : 卡方残差检验 + 硬排除（χ² 99% 置信）
- ``uniform_discount_gamma``     : 所有邻居等权折扣
- ``coordinate_wise_median_aggregate`` : Byzantine-resilient 坐标中位聚合

每个基线都不依赖 ground-truth ``faulty_mask`` 或真实 ``δ``。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Hard-Threshold via chi-squared residual detector
# ---------------------------------------------------------------------------

class HardThresholdDetector:
    """卡方残差检测器：``r_i^T Σ^{-1} r_i ~ χ²_d`` 在无故障假设下。

    在 burn-in 阶段估计 ``Σ ≈ diag(σ²)``（用残差范数平方的均值近似 ``d·σ²``）。
    超过 ``chi2.ppf(confidence, d)`` 的智能体被认为故障。
    """

    def __init__(self, d: int, confidence: float = 0.99):
        self.d = d
        self.threshold = chi2.ppf(confidence, d)
        self.var_est: float | None = None

    def calibrate(self, residual_norms_history: np.ndarray) -> None:
        """从无故障历史估计每维方差 ``σ²``。

        采用 ``E[||r||²] = d·σ²``，把 σ² 估计为残差范数平方均值除以维度 d。

        .. note::
           这是把所有 N 个 agent + 所有时间步的残差范数当作同分布的池化
           估计。在共识收敛过程中残差量级早期较大、后期较小，假设并不严格
           成立。**这是有意为之**——HT 是论文 Section 1 critique 的对象，朴素
           估计正好贴合论文要 critique 的"threshold-based detector ...
           oscillates"现象。不要把这里"修得更精细"。

        Parameters
        ----------
        residual_norms_history : shape (T_burn, N)
        """
        sq = residual_norms_history ** 2
        # E[||r||²] = d σ²  →  σ² = mean / d
        self.var_est = max(float(sq.mean()) / self.d, 1e-8)

    def gamma_matrix(self, residual_norms: np.ndarray) -> np.ndarray:
        """返回 ``(N, N)`` 的 γ：每行将故障邻居的列置 0，其余 1。

        ``residual_norms`` 形状 ``(N,)``。
        """
        N = len(residual_norms)
        if self.var_est is None:
            return np.ones((N, N))
        stat = residual_norms ** 2 / self.var_est
        faulty = stat > self.threshold
        gamma = np.ones((N, N))
        gamma[:, faulty] = 0.0
        return gamma


# ---------------------------------------------------------------------------
# Uniform discount
# ---------------------------------------------------------------------------

def uniform_discount_gamma(N: int, factor: float = 0.9) -> np.ndarray:
    return factor * np.ones((N, N))


# ---------------------------------------------------------------------------
# Byzantine-resilient: coordinate-wise median
# ---------------------------------------------------------------------------

def coordinate_wise_median_aggregate(X: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """对每个智能体 i，用其邻居（含自身）的坐标中位替换其状态。"""
    N = X.shape[0]
    X_new = X.copy()
    for i in range(N):
        nb = list(np.where(adj[i] > 0)[0]) + [i]
        X_new[i] = np.median(X[nb, :], axis=0)
    return X_new
