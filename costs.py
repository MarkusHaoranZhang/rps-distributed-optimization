"""
分布式优化代价模型
====================

本模块提供论文 Section 4.4.1 三档基准的代价类。每个类都暴露统一接口：

  ``problem_dim() -> int``                              决策变量维度
  ``grad_fns() -> list[Callable[[ndarray], ndarray]]``  每个智能体的局部梯度
  ``global_optimum() -> ndarray``                       集中式 x*

- ``LeastSquaresCost``     : 合成最小二乘 (主基准)
- ``LogRegCost``           : MNIST 非 IID 多项 logistic（数据加载在 datasets.py）
- ``QuadraticDispatchCost``: IEEE 39-bus 经济调度
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic least squares (Section 4.4.1)
# ---------------------------------------------------------------------------

class LeastSquaresCost:
    """``f_i(x) = (1 / (2 p_i)) || A_i x - b_i ||^2``。

    除以 p_i 是论文 Section 4.4.1 的标准归一化，保证局部 Hessian 谱不随
    样本量漂移；从而固定 alpha 在不同 (N, d, p) 下保持稳定。
    """

    def __init__(self, A_list: List[np.ndarray], b_list: List[np.ndarray]):
        self.A_list = A_list
        self.b_list = b_list
        self.N = len(A_list)
        self._p = [Ai.shape[0] for Ai in A_list]
        self._d = A_list[0].shape[1]
        self._x_opt: Optional[np.ndarray] = None

    def problem_dim(self) -> int:
        return self._d

    def grad_fns(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        return [self._make_grad(i) for i in range(self.N)]

    def _make_grad(self, i: int):
        Ai, bi, pi = self.A_list[i], self.b_list[i], self._p[i]

        def g(x: np.ndarray) -> np.ndarray:
            return Ai.T @ (Ai @ x - bi) / pi

        return g

    def global_optimum(self) -> np.ndarray:
        if self._x_opt is not None:
            return self._x_opt
        H = sum(self.A_list[i].T @ self.A_list[i] / self._p[i]
                for i in range(self.N))
        g = sum(self.A_list[i].T @ self.b_list[i] / self._p[i]
                for i in range(self.N))
        self._x_opt = np.linalg.solve(H, g)
        return self._x_opt


def generate_least_squares_data(N: int, d: int, p: int, seed: int = 0):
    """生成 N 个智能体的本地 (A_i, b_i)，每个 A_i ∈ R^{p × d}。"""
    rng = np.random.RandomState(seed)
    A_list = [rng.randn(p, d) for _ in range(N)]
    b_list = [rng.randn(p) for _ in range(N)]
    return A_list, b_list


# ---------------------------------------------------------------------------
# MNIST non-IID logistic regression
# ---------------------------------------------------------------------------

class LogRegCost:
    """L2-regularized multinomial logistic regression。

    决策变量展平为 ``(d_feat * n_class,)`` 向量。
    """

    def __init__(self, X_list, y_list, n_class: int, reg: float = 1e-3):
        self.X_list = X_list
        self.y_list = y_list
        self.n_class = n_class
        self.reg = reg
        self.N = len(X_list)
        self.d_feat = X_list[0].shape[1]
        self._d = self.d_feat * n_class
        self._x_opt: Optional[np.ndarray] = None

    def problem_dim(self) -> int:
        return self._d

    def _local_grad(self, w_flat: np.ndarray, idx: int) -> np.ndarray:
        W = w_flat.reshape(self.d_feat, self.n_class)
        Xi = self.X_list[idx]
        yi = self.y_list[idx]
        if Xi.shape[0] == 0:
            return self.reg * w_flat
        logits = Xi @ W
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(yi)), yi] = 1.0
        grad_W = Xi.T @ (probs - onehot) / max(len(yi), 1) + self.reg * W
        return grad_W.ravel()

    def grad_fns(self):
        return [(lambda w, i=i: self._local_grad(w, i)) for i in range(self.N)]

    def global_optimum(self, max_iter: int = 2000, lr: float = 0.5) -> np.ndarray:
        """集中式 GD 求解 x*（已收敛时缓存）。

        如果 ``max_iter`` 步后梯度范数仍 > 1e-4：
          1. 发 ``UserWarning`` 让读者知道相对误差的分母可能不准；
          2. **不缓存**未收敛的 w，下次调用（可传更大 ``max_iter``）会重算。
        """
        if self._x_opt is not None:
            return self._x_opt
        X_all = np.vstack(self.X_list)
        y_all = np.concatenate(self.y_list)
        w = np.zeros(self._d)
        last_grad_norm = float('inf')
        converged = False
        for _ in range(max_iter):
            W = w.reshape(self.d_feat, self.n_class)
            logits = X_all @ W
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            onehot = np.zeros_like(probs)
            onehot[np.arange(len(y_all)), y_all] = 1.0
            grad_W = X_all.T @ (probs - onehot) / len(y_all) + self.reg * W
            w = w - lr * grad_W.ravel()
            last_grad_norm = float(np.linalg.norm(grad_W))
            if last_grad_norm < 1e-4:
                converged = True
                break

        if not converged:
            import warnings
            warnings.warn(
                f"LogRegCost.global_optimum did not converge: ||grad|| = "
                f"{last_grad_norm:.2e} after {max_iter} iters (threshold 1e-4). "
                f"Subsequent relative-error metrics will be biased by the "
                f"un-converged x_opt. Consider raising max_iter / tuning lr. "
                f"NOT caching this result so the next call may retry.",
                stacklevel=2,
            )
            return w   # 返回但不缓存，让下次调用重算

        self._x_opt = w
        return w


# ---------------------------------------------------------------------------
# IEEE 39-bus economic dispatch (quadratic)
# ---------------------------------------------------------------------------

class QuadraticDispatchCost:
    """``f_i(p) = 0.5 a_i p_i^2 + b_i p_i + c_i``。

    决策变量 p ∈ R^N，每个智能体 i 只依赖第 i 维 p_i，但分布式优化要所有
    智能体在 p 上达成一致。论文里以此作为分布式经济调度的简化模型。

    .. note::
       常数项 ``c_i`` **不影响梯度也不影响最优解**（``∂f/∂p_i`` 与 c
       无关）。它仅在评估目标值 ``f(p)`` 时使用。本类目前不暴露
       ``f(p)`` 接口，所以 ``c`` 字段实际只是占位以匹配论文记号。
    """

    def __init__(self, a, b, c):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.N = len(a)
        self._d = self.N

    def problem_dim(self) -> int:
        return self._d

    def grad_fns(self):
        return [self._make(i) for i in range(self.N)]

    def _make(self, i: int):
        a_i, b_i = self.a[i], self.b[i]

        def g(p: np.ndarray) -> np.ndarray:
            grad = np.zeros_like(p)
            grad[i] = a_i * p[i] + b_i
            return grad

        return g

    def global_optimum(self) -> np.ndarray:
        # ∇sum f = a_i p_i + b_i = 0 → p_i = -b_i / a_i
        return -self.b / self.a
