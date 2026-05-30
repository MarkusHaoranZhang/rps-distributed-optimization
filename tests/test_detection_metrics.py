"""Unit tests for ``detection_and_false_alarm_rates`` (paper Section
4.4.3)."""

import numpy as np

from experiments import detection_and_false_alarm_rates


def _adj_chain(N: int) -> np.ndarray:
    """Chain adjacency: ``i`` connects to ``i +/- 1``."""
    adj = np.zeros((N, N))
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def test_perfect_detection_no_false_alarm():
    """``gamma_{neighbor, faulty} = 0`` and ``gamma_{neighbor, healthy} = 1``
    => ``detection = 1``, ``false_alarm = 0``."""
    N = 5
    adj = _adj_chain(N)
    faulty = [2]
    G = np.ones((N, N))
    G[:, 2] = 0.0   # zero out agent 2's column (except the diagonal)
    G[2, 2] = 1.0
    history = [G.copy() for _ in range(10)]
    det, fa = detection_and_false_alarm_rates(history, faulty, N, adj=adj)
    assert det == 1.0
    assert fa == 0.0


def test_no_detection_no_false_alarm():
    """All gammas are 1 (nobody is judged faulty) =>
    ``detection = 0``, ``false_alarm = 0``."""
    N = 5
    adj = _adj_chain(N)
    faulty = [2]
    G = np.ones((N, N))
    history = [G.copy() for _ in range(5)]
    det, fa = detection_and_false_alarm_rates(history, faulty, N, adj=adj)
    assert det == 0.0
    assert fa == 0.0


def test_full_false_alarm():
    """Every healthy agent is misjudged => ``false_alarm = 1``."""
    N = 4
    adj = _adj_chain(N)
    faulty: list = []   # no true fault
    G = np.zeros((N, N))   # every column is judged faulty
    history = [G.copy() for _ in range(3)]
    det, fa = detection_and_false_alarm_rates(history, [0], N, adj=adj)
    # detection depends on faulty=[0]; here we only check fa.
    assert fa == 1.0


def test_empty_history_returns_nan():
    det, fa = detection_and_false_alarm_rates([], [0], N=5)
    assert np.isnan(det)
    assert np.isnan(fa)


def test_empty_faulty_list_returns_nan():
    G = np.ones((4, 4))
    det, fa = detection_and_false_alarm_rates([G], [], N=4)
    assert np.isnan(det)
    assert np.isnan(fa)


def test_partial_detection():
    """2 of 3 faulty agents are identified => ``detection = 2/3``."""
    N = 6
    adj = np.ones((N, N)) - np.eye(N)
    faulty = [1, 3, 5]
    G = np.ones((N, N))
    # Agents 1 and 3 are identified (column mean 0); agent 5 is not
    # (column mean 1).
    G[:, 1] = 0.0; G[1, 1] = 1.0
    G[:, 3] = 0.0; G[3, 3] = 1.0
    history = [G.copy()]
    det, fa = detection_and_false_alarm_rates(history, faulty, N, adj=adj)
    assert abs(det - 2.0 / 3.0) < 1e-10
    assert fa == 0.0
