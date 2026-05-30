"""
Dataset loading and cost-model factories
=========================================

This module handles data loading (MNIST download, PCA dimensionality
reduction, non-IID partitioning) and instantiates the cost classes from
``costs.py``. The cost models themselves live in ``costs.py``.
"""

from __future__ import annotations

import gzip
import os
import struct
from typing import Any
from urllib.request import urlretrieve

import numpy as np

from costs import LogRegCost, QuadraticDispatchCost

# ---------------------------------------------------------------------------
# MNIST loader (no torchvision dependency)
# ---------------------------------------------------------------------------

_MNIST_TRAIN_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
}
"""Training-set URLs only; distributed optimization uses train, not test."""


def _download(url: str, dst: str) -> None:
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        urlretrieve(url, dst)


def _read_idx_images(path: str) -> np.ndarray:
    opener: Any = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(
                f"Bad MNIST images magic number {magic} (expected 2051) in {path}; "
                f"file may be corrupted or wrong format."
            )
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return data.astype(np.float32) / 255.0


def _read_idx_labels(path: str) -> np.ndarray:
    opener: Any = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(
                f"Bad MNIST labels magic number {magic} (expected 2049) in {path}; "
                f"file may be corrupted or wrong format."
            )
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_mnist(cache_dir: str = "./_data"):
    """Download (only on first call) and load the MNIST training set.
    Returns ``(X, y)``."""
    img_path = os.path.join(cache_dir, "train-images-idx3-ubyte.gz")
    lbl_path = os.path.join(cache_dir, "train-labels-idx1-ubyte.gz")
    try:
        _download(_MNIST_TRAIN_URLS["train_images"], img_path)
        _download(_MNIST_TRAIN_URLS["train_labels"], lbl_path)
    except Exception as e:
        raise RuntimeError(f"MNIST download failed: {e}")
    X = _read_idx_images(img_path)
    y = _read_idx_labels(lbl_path)
    return X, y


# ---------------------------------------------------------------------------
# Non-IID partition factory
# ---------------------------------------------------------------------------

def make_mnist_noniid(N: int = 100, classes_per_agent: int = 3, n_class: int = 10,
                       samples_per_agent: int = 200, feature_dim: int = 64,
                       seed: int = 0) -> LogRegCost:
    """Non-IID partition: each agent receives samples from
    ``classes_per_agent`` digit classes.

    For RAM/speed reasons the features are reduced to ``feature_dim`` via PCA.

    .. warning::
       The defaults ``samples_per_agent=200`` x ``feature_dim=64`` x
       ``n_class=10`` give an under-determined problem (640 parameters vs.
       200 samples) that is held together by L2 reg=1e-3. The convergence of
       ``LogRegCost.global_optimum`` is weaker than for synthetic LS, and
       trial-to-trial variance is larger. We recommend MC >= 10 for
       ``--dataset mnist``.
    """
    X, y = load_mnist()
    rng = np.random.RandomState(seed)

    # PCA: take 5000 samples, center, then SVD.
    Xc = X - X.mean(axis=0, keepdims=True)
    idx_pca = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    _, _, Vt = np.linalg.svd(Xc[idx_pca], full_matrices=False)
    proj = Vt[:feature_dim].T
    X_proj = (X - X.mean(axis=0, keepdims=True)) @ proj

    by_class = {c: np.where(y == c)[0].tolist() for c in range(n_class)}
    for c in by_class:
        rng.shuffle(by_class[c])

    X_list, y_list = [], []
    for _ in range(N):
        chosen = rng.choice(n_class, size=classes_per_agent, replace=False)
        Xi_parts, yi_parts = [], []
        per_class = max(1, samples_per_agent // classes_per_agent)
        for c in chosen:
            if len(by_class[c]) < per_class:
                idx_take = rng.choice(np.where(y == c)[0], size=per_class, replace=True)
            else:
                idx_take = by_class[c][:per_class]
                by_class[c] = by_class[c][per_class:]
            Xi_parts.append(X_proj[idx_take])
            yi_parts.append(np.full(len(idx_take), c, dtype=np.int64))
        X_list.append(np.vstack(Xi_parts))
        y_list.append(np.concatenate(yi_parts).astype(np.int64))

    return LogRegCost(X_list, y_list, n_class=n_class, reg=1e-3)


# ---------------------------------------------------------------------------
# IEEE 39-bus economic dispatch factory
# ---------------------------------------------------------------------------

def make_ieee39_dispatch(seed: int = 0) -> QuadraticDispatchCost:
    """Generate the 39 quadratic generator-cost coefficients for the IEEE
    39-bus system.

    .. warning::
       The paper does not specify exact coefficients; we sample reproducible
       random coefficients from a typical economic-dispatch range found in
       the literature. The resulting numbers **cannot be compared directly**
       with the absolute totals from standard 39-bus benchmarks such as
       MATPOWER or PYPOWER. The total-cost figures the reader will see are
       only meaningful in a relative sense, for trend comparison between
       methods.
    """
    rng = np.random.RandomState(seed)
    N = 39
    a = rng.uniform(0.005, 0.025, size=N)
    b = rng.uniform(8.0, 14.0, size=N)
    c = rng.uniform(100.0, 200.0, size=N)
    return QuadraticDispatchCost(a, b, c)
