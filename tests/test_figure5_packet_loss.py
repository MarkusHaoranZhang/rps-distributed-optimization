"""``simulate_symmetric_packet_loss`` 的不变量回归测试。

这个函数被 ``main.figure_5`` 用来产生论文 4.5.3 通信退化子图的数据。
它必须保证 W_mod 仍是合法的双随机矩阵（对称、行和=1、列和=1、非负），
否则 GT 的物理意义崩坏：figure_5 中间子图反映的不再是"丢包鲁棒性"，
而是"W 失双随机后伪信号"。

历史 bug 防回归（按发现时间排）：
1. 第一版只把 W[i,j] 置 0 不修对角，破坏行和=1。
2. 第二版只修行随机，单边丢包破坏列随机性 → Y-tracking 失效。
3. 当前版用对称丢包 + 对角修补，保证 W_mod 仍是合法双随机。

本测试套**直接调用 main.figure_5 用的真实函数**，不复制实现——避免
v0.4.3 时 ``test_gradient_tracking_with_gamma_preserves_row_sum`` 那种
"假阳性测试"陷阱（重写一份逻辑做断言，真实函数改了测试不会发现）。
"""

import numpy as np
import pytest

from distributed_optimization import build_graph, simulate_symmetric_packet_loss


@pytest.mark.parametrize("loss_rate", [0.0, 0.05, 0.25, 0.5, 0.75])
def test_packet_loss_preserves_double_stochasticity(loss_rate):
    """丢包后 W_mod 必须保持双随机（行和=列和=1）+ 非负 + 对称。"""
    W, _, _ = build_graph(N=20, seed=3)
    rng = np.random.RandomState(int(loss_rate * 100) + 1)
    W_mod = simulate_symmetric_packet_loss(W, loss_rate, rng)

    np.testing.assert_allclose(W_mod.sum(axis=1), 1.0, atol=1e-10,
        err_msg=f"loss={loss_rate}: W_mod 行和 ≠ 1, 破坏行随机")
    np.testing.assert_allclose(W_mod.sum(axis=0), 1.0, atol=1e-10,
        err_msg=f"loss={loss_rate}: W_mod 列和 ≠ 1, 破坏列随机（Y-tracking 失效）")
    np.testing.assert_allclose(W_mod, W_mod.T, atol=1e-12,
        err_msg=f"loss={loss_rate}: W_mod 不对称")
    assert (W_mod >= -1e-12).all(), \
        f"loss={loss_rate}: W_mod 出现负元素"


def test_higher_loss_means_more_zeros():
    """更大丢包率应该让 W_mod 中更多元素为 0（单调性）。"""
    W, _, _ = build_graph(N=30, seed=7)
    W_5 = simulate_symmetric_packet_loss(W, 0.05, np.random.RandomState(42))
    W_50 = simulate_symmetric_packet_loss(W, 0.50, np.random.RandomState(42))
    # W_50 的非零元素数 ≤ W_5 的（更多丢包 → 更多 0）
    assert (W_50 == 0).sum() >= (W_5 == 0).sum()


def test_zero_loss_returns_unchanged_W():
    """loss_rate=0 时 W_mod 应与 W 完全相等。"""
    W, _, _ = build_graph(N=15, seed=11)
    W_mod = simulate_symmetric_packet_loss(W, 0.0, np.random.RandomState(0))
    np.testing.assert_array_equal(W_mod, W)


def test_invalid_loss_rate_raises():
    W, _, _ = build_graph(N=10, seed=0)
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError, match="loss_rate"):
        simulate_symmetric_packet_loss(W, -0.1, rng)
    with pytest.raises(ValueError, match="loss_rate"):
        simulate_symmetric_packet_loss(W, 1.5, rng)
