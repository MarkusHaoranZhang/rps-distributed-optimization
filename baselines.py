"""
Baseline methods (paper Section 4.4.2)
=======================================

- ``HardThresholdDetector``           : chi-squared residual test + strict
                                        exclusion (chi^2 99% confidence)
- ``uniform_discount_gamma``          : equal-weight discount over all
                                        neighbors
- ``coordinate_wise_median_aggregate`` : Byzantine-resilient coordinate-wise
                                        median aggregation

None of the baselines rely on the ground-truth ``faulty_mask`` or the true
``delta``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Hard-Threshold via chi-squared residual detector
# ---------------------------------------------------------------------------

class HardThresholdDetector:
    """Chi-squared residual detector: under the fault-free hypothesis,
    ``r_i^T Sigma^{-1} r_i ~ chi^2_d``.

    During burn-in we estimate ``Sigma ~ diag(sigma^2)`` (using the mean of
    squared residual norms to approximate ``d * sigma^2``). Agents whose
    statistic exceeds ``chi2.ppf(confidence, d)`` are flagged as faulty.
    """

    def __init__(self, d: int, confidence: float = 0.99):
        self.d = d
        self.threshold = chi2.ppf(confidence, d)
        self.var_est: float | None = None

    def calibrate(self, residual_norms_history: np.ndarray) -> None:
        """Estimate the per-coordinate variance ``sigma^2`` from the
        fault-free history.

        We use ``E[||r||^2] = d * sigma^2``, i.e. estimate ``sigma^2`` as
        the mean of squared residual norms divided by the dimension ``d``.

        .. note::
           This is a pooled estimator that treats the residual norms over
           all ``N`` agents and all time steps as samples from the same
           distribution. During consensus convergence the residual
           magnitude is larger early and smaller later, so the assumption
           is not strict. **That is on purpose** -- HT is exactly the
           method that Section 1 of the paper criticizes, and the naive
           estimator reproduces the "threshold-based detector ...
           oscillates" phenomenon the paper wants to expose. Do not "fix
           this to be more refined".

        Parameters
        ----------
        residual_norms_history : shape ``(T_burn, N)``
        """
        sq = residual_norms_history ** 2
        # E[||r||^2] = d * sigma^2  =>  sigma^2 = mean / d
        self.var_est = max(float(sq.mean()) / self.d, 1e-8)

    def gamma_matrix(self, residual_norms: np.ndarray) -> np.ndarray:
        """Return the ``(N, N)`` gamma: rows zero out columns for faulty
        neighbors and keep the rest at 1.

        ``residual_norms`` has shape ``(N,)``.
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
    """For each agent ``i``, replace its state with the coordinate-wise
    median of its neighbors (including itself)."""
    N = X.shape[0]
    X_new = X.copy()
    for i in range(N):
        nb = list(np.where(adj[i] > 0)[0]) + [i]
        X_new[i] = np.median(X[nb, :], axis=0)
    return X_new
