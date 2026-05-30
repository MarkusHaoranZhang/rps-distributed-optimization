"""Invariant regression tests for ``simulate_symmetric_packet_loss``.

This function is used by ``main.figure_5`` to produce the data for the
communication-degradation panel in paper Section 4.5.3. It must guarantee
that ``W_mod`` is still a valid doubly-stochastic matrix (symmetric, row
sum = column sum = 1, non-negative); otherwise the physical meaning of
gradient tracking breaks down and the middle panel of figure 5 no longer
reflects "packet-loss robustness" but a "fake signal from a non-doubly-
stochastic W".

Historical bugs guarded against (in order of discovery):
1. The first version zeroed ``W[i, j]`` without repairing the diagonal,
   breaking row sum = 1.
2. The second version repaired only row stochasticity; one-sided drops
   broke column stochasticity, which makes Y-tracking fail.
3. The current version uses symmetric drops + diagonal repair, keeping
   ``W_mod`` valid and doubly stochastic.

These tests **call the same function used by main.figure_5 directly**;
they do not duplicate the implementation, to avoid the "false-positive
test" trap from v0.4.3 (where
``test_gradient_tracking_with_gamma_preserves_row_sum`` reimplemented the
logic inside the assertion, so changes to the real function would not be
caught).
"""

import numpy as np
import pytest

from distributed_optimization import build_graph, simulate_symmetric_packet_loss


@pytest.mark.parametrize("loss_rate", [0.0, 0.05, 0.25, 0.5, 0.75])
def test_packet_loss_preserves_double_stochasticity(loss_rate):
    """``W_mod`` after losses must remain doubly stochastic
    (row sum = col sum = 1) + non-negative + symmetric."""
    W, _, _ = build_graph(N=20, seed=3)
    rng = np.random.RandomState(int(loss_rate * 100) + 1)
    W_mod = simulate_symmetric_packet_loss(W, loss_rate, rng)

    np.testing.assert_allclose(W_mod.sum(axis=1), 1.0, atol=1e-10,
        err_msg=f"loss={loss_rate}: W_mod row sum != 1, broke row stochasticity")
    np.testing.assert_allclose(W_mod.sum(axis=0), 1.0, atol=1e-10,
        err_msg=f"loss={loss_rate}: W_mod col sum != 1, broke col stochasticity (Y-tracking fails)")
    np.testing.assert_allclose(W_mod, W_mod.T, atol=1e-12,
        err_msg=f"loss={loss_rate}: W_mod is not symmetric")
    assert (W_mod >= -1e-12).all(), \
        f"loss={loss_rate}: W_mod has negative entries"


def test_higher_loss_means_more_zeros():
    """A higher loss rate should leave more zero entries in ``W_mod``
    (monotonicity)."""
    W, _, _ = build_graph(N=30, seed=7)
    W_5 = simulate_symmetric_packet_loss(W, 0.05, np.random.RandomState(42))
    W_50 = simulate_symmetric_packet_loss(W, 0.50, np.random.RandomState(42))
    # The non-zero count of W_50 should be <= that of W_5 (more drops ->
    # more zeros).
    assert (W_50 == 0).sum() >= (W_5 == 0).sum()


def test_zero_loss_returns_unchanged_W():
    """At ``loss_rate=0``, ``W_mod`` must equal ``W`` exactly."""
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
