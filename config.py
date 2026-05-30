"""
集中配置 + 核心数据结构
========================

所有调优旋钮都收拢在 ``RPSConfig`` 里。``run_optimization`` 接收一个
``RPSConfig`` 实例而不是一堆散参数；任何想复现某次实验的人只需要存下这一个对象。

PMF 三元组从 ``Tuple[tuple, ndarray, ndarray]`` 升级为 dataclass，得到字段命
名、类型检查、未来可扩展（缓存 singleton 向量等）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# Permutation Mass Function — replaces the old (events, mass, masks) tuple
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PMF:
    """Random Permutation Set 上的截断质量函数。

    Attributes
    ----------
    events : tuple[tuple[int, ...], ...]
        按顺序排列的事件元组；每个事件是一个有序的智能体索引元组。
    mass : np.ndarray
        形状 ``(E,)``，对应每个事件的质量；总和 = 1（除非空 PMF）。
    masks : np.ndarray
        形状 ``(E,)``；每个事件成员的位掩码。当作用域大小 ≤ 63 时为 int64，
        否则降级为 object dtype 容纳 Python 大整数。
    """

    events: tuple
    mass: np.ndarray
    masks: np.ndarray

    def __len__(self) -> int:
        return len(self.events)

    @property
    def is_empty(self) -> bool:
        return len(self.events) == 0

    @classmethod
    def empty(cls) -> "PMF":
        return cls(events=(), mass=np.empty(0),
                   masks=np.empty(0, dtype=np.int64))


# ---------------------------------------------------------------------------
# Fault configuration schema
# ---------------------------------------------------------------------------

class FaultConfig(TypedDict, total=False):
    """``apply_fault_injection`` 接收的故障配置 schema。

    .. note::
       这是一个**仅作文档用途**的 ``TypedDict``。运行时所有调用点
       （``apply_fault_injection`` / ``run_optimization`` / ``validate_fault_config``）
       的形参类型仍是 ``dict``，因为各处构造 fault_config 时为了便利使用了字面
       量。``validate_fault_config`` 会在运行时执行 schema 检查（缺字段、类型错、
       取值范围），所以 schema 违例不会静默通过。

       想得到 mypy 级别的强制保证可以把所有调用点改成 ``FaultConfig``——但当前
       论文配套不需要，运行时校验已经覆盖了所有失败模式。

    Required
    --------
    onset : int
        故障注入起始的迭代步。
    agents : list[int]
        故障智能体的全局索引列表（可空，表示无故障）。
    type : str
        ``"constant"`` / ``"drift"`` / ``"intermittent"`` 之一。

    Optional / type-specific
    ------------------------
    delta : np.ndarray | None
        故障的偏差幅度 base，形状 ``(d,)``。``constant`` / ``drift`` /
        ``intermittent`` 都用它。
    drift_cap : float
        仅 ``type="drift"`` 时使用：drift 渐增到该倍数后饱和（默认 100.0）。
        论文 Section 4.4 的 small-fault regime 假设要求偏差有界；这是工程上
        把无界线性 drift 转成有界 ramp 的方法。
    prob : float
        仅 ``type="intermittent"`` 时使用：每步触发故障的概率。
    """

    onset: int
    agents: list
    type: str
    delta: Optional[np.ndarray]
    drift_cap: float
    prob: float


# ---------------------------------------------------------------------------
# RPS configuration — 所有调优旋钮集中在这里
# ---------------------------------------------------------------------------

@dataclass
class RPSConfig:
    """RPS 诊断 + 软折扣的所有调优参数。

    论文公式参考
    -----------
    - h_hop / k_trunc : 论文 Section 4.1 的可诊断作用域和截断阶数。
    - window_len      : 论文 Section 4.1 的滑动窗口长度 s。
    - eta             : 论文 Eq.(9) 的 softmax 温度。
    - top_m           : PMF 截断保留的 top 事件数；工程优化项。
    - top_agents_k    : compute_pmf 的两阶段事件枚举裁剪参数；工程优化项。
    - gain            : 连续软折扣 γ = exp(-gain · P_OPT) 的灵敏度。
                        论文 Eq.(12) 的连续化，gain 是新增工程参数。
    - tau_quantile    : 论文 Section 4.3 把 τ 校准为无故障期熵分布的指定分位。
    - diagnose_every  : 工程节流：每隔多少步触发一次诊断重计算。
                        论文公式里没有这一项；LOS 是 O(E²) 主瓶颈。
    - proxy_std_weight : magnitude_proxy = max(mean_inc, weight · std_inc)。
                        weight 控制 std 增量在故障源代理中的权重；工程参数。
    - proxy_global_weight : compute_pmf 内 contrib + weight · proxy 的混合系数；
                        把全局 proxy 加进每智能体局部信号。论文公式里没有。

    这些「工程参数」都不在论文的核心方法论里，但在我们的实验设置下是必要的。
    详见 ``IMPLEMENTATION_NOTES.md``。
    """

    # ----- core (paper) -----
    h_hop: int = 2
    k_trunc: int = 3
    window_len: int = 20
    eta: float = 1.0
    burn_in: int = 100
    tau: Optional[float] = None
    """显式 τ；None 表示按 ``tau_quantile`` 在 burn-in 末从熵分布校准。"""
    tau_quantile: float = 0.95
    """τ 校准：当 ``tau`` 为 None 时取无故障期熵的此分位（论文 Section 4.3）。"""

    # ----- engineering knobs -----
    top_m: int = 16
    top_agents_k: int = 5
    gain: float = 4.0
    diagnose_every: int = 5
    proxy_std_weight: float = 2.0
    proxy_global_weight: float = 0.5

    # ----- baseline-related -----
    uniform_factor: float = 0.9      # Uniform-Discount 的固定折扣
    chi2_confidence: float = 0.99    # Hard-Threshold 的卡方置信度

    # ----- ablation switches -----
    record_diagnosis: bool = True
    record_agent_idx: int = 0        # 诊断日志记录哪个智能体的视角

    def __post_init__(self) -> None:
        """构造时立即校验参数合法性，让无效配置在源头报错。"""
        problems: list[str] = []
        if self.h_hop < 1:
            problems.append(f"h_hop must be ≥ 1, got {self.h_hop}")
        if self.k_trunc < 1:
            problems.append(f"k_trunc must be ≥ 1, got {self.k_trunc}")
        if self.window_len < 1:
            problems.append(f"window_len must be ≥ 1, got {self.window_len}")
        if self.eta <= 0:
            problems.append(f"eta must be > 0, got {self.eta}")
        if self.burn_in < self.window_len:
            problems.append(
                f"burn_in ({self.burn_in}) must be ≥ window_len ({self.window_len}) "
                f"(否则故障期前没有完整滑窗可用于校准)"
            )
        if not (0.0 < self.tau_quantile < 1.0):
            problems.append(f"tau_quantile must be in (0, 1), got {self.tau_quantile}")
        if self.top_m < 1:
            problems.append(f"top_m must be ≥ 1, got {self.top_m}")
        if self.top_agents_k < 1:
            problems.append(f"top_agents_k must be ≥ 1, got {self.top_agents_k}")
        if self.gain < 0:
            problems.append(f"gain must be ≥ 0, got {self.gain}")
        if self.diagnose_every < 1:
            problems.append(f"diagnose_every must be ≥ 1, got {self.diagnose_every}")
        if self.proxy_std_weight < 0:
            problems.append(f"proxy_std_weight must be ≥ 0, got {self.proxy_std_weight}")
        if self.proxy_global_weight < 0:
            problems.append(f"proxy_global_weight must be ≥ 0, got {self.proxy_global_weight}")
        if not (0.0 <= self.uniform_factor <= 1.0):
            problems.append(f"uniform_factor must be in [0, 1], got {self.uniform_factor}")
        if not (0.0 < self.chi2_confidence < 1.0):
            problems.append(f"chi2_confidence must be in (0, 1), got {self.chi2_confidence}")
        if self.record_agent_idx < 0:
            problems.append(f"record_agent_idx must be ≥ 0, got {self.record_agent_idx}")
        if problems:
            raise ValueError("Invalid RPSConfig:\n  - " + "\n  - ".join(problems))

    def replace(self, **changes) -> "RPSConfig":
        """返回一个修改了若干字段的副本（用于参数敏感性扫描）。"""
        from dataclasses import replace
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

KNOWN_METHODS: tuple = (
    "Ideal",
    "Hard-Threshold",
    "Uniform-Discount",
    "Byzantine-Resilient",
    "RPS-Symmetric",
    "RPS-NoOrder",
    "RPS-Full",
)
"""所有支持的方法名（论文 Section 4.4.2）。

新增/重命名/调整顺序时需要同步：

- ``main._FIG2_METHODS``（Figure 2 / Table 2 列出的子集，目前少了 RPS-NoOrder 因
  RPS-NoOrder 在 Figure 6 单独出现）；
- ``experiments.run_optimization`` 第 3 步的方法分派（HT/Uniform/RPS/Byzantine）；
- ``tests/test_run_optimization.py::test_each_method_runs`` 通过
  ``parametrize(list(KNOWN_METHODS))`` 自动覆盖，新方法会自动加入冒烟测试。
"""

RPS_METHODS: tuple = ("RPS-Full", "RPS-Symmetric", "RPS-NoOrder")


def is_rps_method(method: str) -> bool:
    return method in RPS_METHODS


# ---------------------------------------------------------------------------
# Runtime validation for the fault_config dict
# ---------------------------------------------------------------------------

_VALID_FAULT_TYPES = ("constant", "drift", "intermittent")


def validate_fault_config(fault_config: dict, *, d: Optional[int] = None) -> None:
    """检查 fault_config 字典是否符合 ``FaultConfig`` schema。

    字段缺失或类型错误时抛出 ``ValueError`` 并附带问题列表。
    在 ``run_optimization`` 入口处调用，让格式错误在源头暴露。

    Parameters
    ----------
    fault_config : 待校验的字典
    d            : 若给出，会额外检查 ``delta.shape == (d,)``
    """
    problems: list[str] = []
    if 'onset' not in fault_config:
        problems.append("missing required key 'onset'")
    elif not isinstance(fault_config['onset'], int) or fault_config['onset'] < 0:
        problems.append(f"'onset' must be non-negative int, got {fault_config['onset']!r}")

    if 'agents' not in fault_config:
        problems.append("missing required key 'agents'")
    elif not isinstance(fault_config['agents'], (list, tuple)):
        problems.append(f"'agents' must be list/tuple, got {type(fault_config['agents']).__name__}")

    if 'type' not in fault_config:
        problems.append("missing required key 'type'")
    elif fault_config['type'] not in _VALID_FAULT_TYPES:
        problems.append(
            f"'type' must be one of {_VALID_FAULT_TYPES}, got {fault_config['type']!r}"
        )

    ftype = fault_config.get('type')
    if ftype in ('constant', 'drift', 'intermittent'):
        if fault_config.get('agents') and fault_config.get('delta') is None:
            problems.append(f"'delta' must be provided for type='{ftype}' with agents")
        elif d is not None and fault_config.get('delta') is not None:
            delta = np.asarray(fault_config['delta'])
            if delta.shape != (d,):
                problems.append(
                    f"'delta' shape {delta.shape} does not match decision dim ({d},)"
                )
    if ftype == 'intermittent':
        if 'prob' not in fault_config:
            problems.append("type='intermittent' requires 'prob' key")
        elif not (0.0 <= float(fault_config['prob']) <= 1.0):
            problems.append(f"'prob' must be in [0, 1], got {fault_config['prob']!r}")
    if ftype == 'drift' and 'drift_cap' in fault_config:
        if float(fault_config['drift_cap']) <= 0:
            problems.append(f"'drift_cap' must be > 0, got {fault_config['drift_cap']!r}")

    if problems:
        raise ValueError("Invalid fault_config:\n  - " + "\n  - ".join(problems))
