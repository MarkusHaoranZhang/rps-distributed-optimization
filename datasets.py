"""
数据集加载与代价模型工厂
==========================

本模块负责数据加载（MNIST 下载、PCA 降维、非 IID 切分）和实例化
``costs.py`` 中的代价类。代价模型本身在 ``costs.py``。
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
"""仅训练集 URL；分布式优化只用 train，不需要 test。"""


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
    """下载（仅首次）并加载 MNIST 训练集。返回 ``(X, y)``。"""
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
    """非 IID 分配：每个 agent 拿 ``classes_per_agent`` 个数字类的样本。

    出于 RAM/速度考虑，特征做 PCA 降维到 ``feature_dim`` 维。

    .. warning::
       默认 ``samples_per_agent=200`` × ``feature_dim=64`` × ``n_class=10`` 是欠
       定问题（参数 640 > 样本 200），靠 L2 reg=1e-3 救稳。``LogRegCost.global_optimum``
       的收敛性比 synthetic LS 弱，trial-to-trial 方差较大。``--dataset mnist`` 的
       结果建议用 MC ≥ 10。
    """
    X, y = load_mnist()
    rng = np.random.RandomState(seed)

    # PCA：抽 5000 个样本中心化后 SVD
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
    """生成 IEEE 39 总线系统的 39 个发电机二次代价系数。

    .. warning::
       论文里没给具体系数；这里用文献中典型的经济调度系数范围生成可复现
       的随机系数。结果**不可与 MATPOWER / PYPOWER 等标准 39-bus benchmark
       的绝对数值直接比较**——读者跑出的总成本数字是相对意义上的，仅
       用于方法间趋势对比。
    """
    rng = np.random.RandomState(seed)
    N = 39
    a = rng.uniform(0.005, 0.025, size=N)
    b = rng.uniform(8.0, 14.0, size=N)
    c = rng.uniform(100.0, 200.0, size=N)
    return QuadraticDispatchCost(a, b, c)
