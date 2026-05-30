"""``detection_and_false_alarm_rates`` 的单元测试（论文 Section 4.4.3）。"""

import numpy as np

from experiments import detection_and_false_alarm_rates


def _adj_chain(N: int) -> np.ndarray:
    """链式邻接：i 连 i±1。"""
    adj = np.zeros((N, N))
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def test_perfect_detection_no_false_alarm():
    """γ_{邻居,faulty}=0，γ_{邻居,healthy}=1 → detection=1, false_alarm=0。"""
    N = 5
    adj = _adj_chain(N)
    faulty = [2]
    G = np.ones((N, N))
    G[:, 2] = 0.0   # 把 agent 2 的列全置 0（除自身）
    G[2, 2] = 1.0
    history = [G.copy() for _ in range(10)]
    det, fa = detection_and_false_alarm_rates(history, faulty, N, adj=adj)
    assert det == 1.0
    assert fa == 0.0


def test_no_detection_no_false_alarm():
    """γ 全为 1（没人被判为故障）→ detection=0, false_alarm=0。"""
    N = 5
    adj = _adj_chain(N)
    faulty = [2]
    G = np.ones((N, N))
    history = [G.copy() for _ in range(5)]
    det, fa = detection_and_false_alarm_rates(history, faulty, N, adj=adj)
    assert det == 0.0
    assert fa == 0.0


def test_full_false_alarm():
    """所有 healthy agent 被误判 → false_alarm=1。"""
    N = 4
    adj = _adj_chain(N)
    faulty: list = []   # 没有真实故障
    G = np.zeros((N, N))   # 所有列都被判为故障
    history = [G.copy() for _ in range(3)]
    det, fa = detection_and_false_alarm_rates(history, [0], N, adj=adj)
    # detection 与 faulty=[0] 相关；这里用 fa 检查
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
    """3 个故障 agent 中 2 个被识别 → detection=2/3。"""
    N = 6
    adj = np.ones((N, N)) - np.eye(N)
    faulty = [1, 3, 5]
    G = np.ones((N, N))
    # agent 1, 3 被识别（列均值 0），agent 5 未识别（列均值 1）
    G[:, 1] = 0.0; G[1, 1] = 1.0
    G[:, 3] = 0.0; G[3, 3] = 1.0
    history = [G.copy()]
    det, fa = detection_and_false_alarm_rates(history, faulty, N, adj=adj)
    assert abs(det - 2.0 / 3.0) < 1e-10
    assert fa == 0.0
