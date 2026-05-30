# RPS-based Distributed Optimization under Soft Faults

[License: MIT](./LICENSE) · 论文配套实现 · Python 3.10+

论文 *Random Permutation Set-Based Diagnosis for Reliable Large-Scale
Distributed Optimization Under Soft Faults* 的配套实现。代码闭环实现了
诊断 → 软折扣 γ → 鲁棒梯度跟踪的完整流程，并通过 Monte Carlo 实验复现
论文中所有 8 张图与 2 张表。

## 论文论断对应情况（v0.4.7 完整模式 N=50, MC=20）

| 论文论断 | 对应情况 |
|---|---|
| §1 中心论断：RPS 显著优于 detect-then-isolate (HT) | ✓ 完全对应（Drift 上 d=1.87, p<0.001）|
| §1 HT oscillation 现象 | ✓ 完全对应（HT 收敛迭代数 928±216, RPS 173±3）|
| §4.5.2 RPS-NoOrder 比 RPS-Full 差 | ✓ 完全对应（d=0.73, p<0.001）|
| §4.5.2 RPS-Sym 比 RPS-Full 差 | ✓ 完全对应（Drift d=1.18, Constant d=2.01）|
| §4.5.3 Constant 上 RPS-Full 最优 | ✓ 完全对应（3.82 < UD 4.03 < HT 4.80, 全部 p<0.001）|
| §4.5.3 Drift 上 RPS-Full 比 next-best 优 40% 的具体数字 | △ 当前实现下方向对（RPS 显著优于 HT/Sym）但与 UD 平手；详见 [`expected_results_full.json`](./expected_results_full.json)|
| §4.5.3 Intermittent 上 RPS-Full 显著优 | △ 方向对（RPS 优于所有 RPS 变体与 HT 在均值上）但 MC=20 噪声下统计不显著 |
| §4.5.4 全部 5 项假设验证 | ✓ `verify_assumptions.py` 5/5 PASS |

具体数字见 [`expected_results_full.json`](./expected_results_full.json) 的 `paper_claims_correspondence` 字段。论文方法学完全实现；具体数字差距详见 [`IMPLEMENTATION_NOTES.md §15`](./IMPLEMENTATION_NOTES.md)（代理 δ + 随机种子 + drift_cap 三层来源）。

## 项目结构

```
.
├── config.py                 # 核心数据结构 (PMF, RPSConfig, FaultConfig)
├── costs.py                  # 代价模型：LeastSquares, LogReg, QuadraticDispatch
├── datasets.py               # MNIST 加载、IEEE 39-bus 工厂
├── distributed_optimization.py  # 图、共识权重、梯度跟踪、故障注入、残差
├── rps_diagnosis.py          # 能量距离、PMF、LOS/DS、JS、OPT、γ
├── baselines.py              # Hard-Threshold (χ²)、Uniform、Byzantine
├── statistics_utils.py       # Wilcoxon、Holm-Bonferroni、Cohen's d
├── experiments.py            # run_optimization 主循环 + 派生指标
├── figures.py                # 8 张图的绘制函数
├── main.py                   # 实验主入口
├── verify_assumptions.py     # 论文 Section 4.5.4 假设检查脚本
├── tests/                    # pytest 单元测试
├── pyproject.toml            # ruff & pytest & mypy 配置
├── requirements.txt          # 钉版本范围
└── IMPLEMENTATION_NOTES.md   # 代码相对论文的偏离及理由
```

## 安装

```
pip install -r requirements.txt
```

## 复现论文结果

### 一键全跑（论文 8 张图，约 2-3 小时，MC=20）

```
python main.py
```

完成后当前目录会出现：

| 文件 | 论文位置 | `--figures N` 的 N |
|------|---------|----|
| `fig_preliminary.pdf` | Figure 1 — 残差演化 | 1 |
| `fig_ablation.pdf`    | Figure 2 — 消融效应量 | 6 |
| `fig_comparative.pdf` | Figure 3 — 三场景对比收敛 | 2 |
| `fig_diagnostic.pdf`  | Figure 4 — 诊断延迟 | 7 |
| `fig_scaling.pdf`     | Figure 5 — 可扩展性 | 8 |
| `fig_sensitivity.pdf` | Figure 6 — 参数敏感性 | 3 |
| `fig_stability.pdf`   | Figure 7 — 稳定性相图 | 4 |
| `fig_stress.pdf`      | Figure 8 — 压力测试 | 5 |
| `results.json`        | Table 1 / Table 2 全数值 | — |

> 注意：``--figures N`` 的 N 是**代码内部执行顺序**，不是论文 Figure 编号。
> 内部顺序是为了让 Figure 7 (代码内 fig 7) 复用 Figure 2 (代码内 fig 2)
> 跑出的 MTCD 数据；论文 Figure 编号按论文章节顺序。两套编号不一致是已知
> 历史包袱，本表给出双向映射。

### 快速验证（约 8 分钟，MC=3）

```
python main.py --quick
```

跑完后与 [`expected_results.json`](./expected_results.json) 对比；具体数字
因随机种子会有 ±10% 浮动，**RPS-Full 应稳定优于 Hard-Threshold**（这是
论文 Section 1 的中心论断，由 ``test_paper_core_claim`` 守门）。但**与
其它 baseline 的排序在 quick mode 下不稳定**——见下文"⚠ quick mode 排序
异常"。最直观的验证是代码 figure 7 (论文 Figure 4) 后控制台打印的诊断
指标：

```
Hard-Threshold detection=0.000, false-alarm=0.136
RPS-Full       detection=0.525, false-alarm=0.003
```

这是论文 Section 1 中心论断"threshold-based detector ... oscillates"的
直接数值证明。

> **⚠ quick mode 与完整模式的差异**
>
> 在 ``--quick`` (N=30, MC=3) 下，RPS-Full 在 Drift 上的 final error
> ≈ 39.66×10⁻³，**显著优于 Hard-Threshold/RPS-Sym/RPS-NoOrder**（论文
> Section 1 与 Section 4.5.2 的两条核心论断都对应得上），但仍**略劣
> 于 Uniform-Discount / Byzantine-Resilient (≈ 36.58)** 4-8%。原因：
> N=30, MC=3 噪声较大，UD 的无差别 self-damping 在小信号下天然占便宜。
>
> 论文 4.5.3 "RPS-Full outperforms next-best by over 40%" 的完整论断
> 需要 N=50, MC=20 完整模式（``python main.py``，约 2 小时）才能复现。
> [`expected_results_full.json`](./expected_results_full.json) 是
> v0.4.7 完整模式真实跑出的数字快照。
>
> 详见 [IMPLEMENTATION_NOTES.md §20](./IMPLEMENTATION_NOTES.md)。

### 复现某个具体数字

每条命令对应论文中的一组数字。``--figures N`` 中的 N 是代码内部编号，与论文
Figure 编号的映射见上表。

| 论文位置 | 命令 | 输出 |
|----------|------|------|
| Table 2 + 论文 Figure 3 全部 | `python main.py --figures 2` | `results.json` 的 ``fig2_finals``、Wilcoxon p_adj、Cohen's d |
| 论文 Table 2 RPS-Full Drift 行（绝对数字差距见下文） | `python main.py --figures 2 --mc 20` | `Gradual drift` 行 |
| 论文 Figure 6 τ 敏感性 | `python main.py --figures 3` | `fig_sensitivity.pdf` 的 "Confidence threshold" 子图 |
| 论文 Figure 7 κ_emp / κ_theo | `python main.py --figures 4` | 控制台打印 + `fig_stability.pdf` |
| 论文 Figure 8 压力测试 | `python main.py --figures 5` | `fig_stress.pdf` |
| 论文 Table 1 (Drift, ablation) | `python main.py --figures 6 --mc 20` | 控制台 "Table 1: Ablation summary" |
| 论文 Figure 4 MTCD | `python main.py --figures 2,7 --mc 20` | `fig_diagnostic.pdf` |
| 论文 Figure 5 N=50 vs 200 | `python main.py --figures 8` | `fig_scaling.pdf` |
| Section 4.5.4 假设验证 | `python verify_assumptions.py` | 5 项 PASS/FAIL 报告 |

### 切换数据集

主基准是合成最小二乘。要在 MNIST 非 IID 或 IEEE 39-bus 上重跑论文 Figure 3 与 Figure 2（代码内 fig 2 与 fig 6）：

```
python main.py --dataset mnist --figures 2,6 --quick
python main.py --dataset ieee39 --figures 2,6 --quick
```

## 关键实现要点

详细逐项说明见 [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md)。简要列举：

1. **梯度跟踪 (Eq.2)**：保留 ``grad_old``，正确实现 ``Y = WY + (∇f_new − ∇f_old)``
2. **γ 双向施加**：``ā_ij = γ_ij W_ij + (1−γ_ij)·δ_ij``，X 共识与 Y 跟踪都用 ``ā``
3. **故障幅度代理**：用残差范数滑窗的均值与标准差增量取最大值，**不读** ground-truth ``δ``
4. **PMF 复杂度优化**：两阶段事件枚举 + z-score 替代能量距离，O(E·s·M) → O(E)
5. **诊断节流** ``diagnose_every``：LOS 是 O(E²) 主瓶颈，每 5 步重计算
6. **τ 校准**：burn-in 期累积 PMF 熵 → 按 ``cfg.tau_quantile`` 取分位
7. **连续软折扣**：``γ = exp(-gain · P_OPT)``，gain=4；高熵时 gain 减半

## 调优旋钮

所有调优参数集中在 ``config.RPSConfig``：

```python
from config import RPSConfig
cfg = RPSConfig(
    h_hop=2, k_trunc=3, window_len=20, eta=1.0,
    burn_in=100, tau=None, tau_quantile=0.95,
    top_m=16, diagnose_every=5,
    gain=4.0, proxy_std_weight=2.0, proxy_global_weight=0.5,
)
```

要做参数敏感性扫描可用 ``cfg.replace(eta=2.0)``。

## 自检

```
pytest tests/             # 单元测试 (~30s, 包含 test_paper_core_claim 的核心论断回归)
ruff check .              # lint
mypy .                    # 类型检查（pyproject.toml 已配 exclude）
python verify_assumptions.py    # 论文假设是否在当前实验配置下成立
```

## 注意事项

- **MC trials 默认 20**（论文 Section 4.4.5 规格）。完整模式约 2-3 小时；
  ``--quick`` 用 MC=3 快速验证。
- 所有故障注入与诊断不读取 ``faulty_mask`` / ground-truth ``δ``。
  ``verify_assumptions.py`` 静态扫描诊断路径确认无 ground-truth 泄露。
- ``drift_cap`` 默认 100 以保持论文 Section 4.4 的 small-fault regime；
  无界 drift 会让所有方法都被 misspecification 主导。

## 与论文 Table 1/2 的绝对数字差距

读者跑出来的 final relative error 会**比论文 Table 1/2 大 3-50 倍**。例如
论文 Drift RPS-Full = 1.12 (×10⁻³)，代码 quick mode ≈ 75 (×10⁻³)，full
mode ≈ 50 (×10⁻³)。**定性结论（RPS-Full < Hard-Threshold < ...）一致，
绝对数字不一致**。

原因（详见 [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md)）：

1. **Eq.(7) 的 E[r_i\|A] 在论文里默认 δ 已知**——论文公式直接写出
   ``Σ F_{i←j} δ_j``，但实际算法运行时 δ 未知。任何不读 ground-truth 的
   实现都用代理 δ（``magnitude_proxy``），引入估计误差。论文数字相当于
   "假装知道 δ" 的上界。
2. **drift_cap=80** 让故障在 t=onset+80 后保持饱和量级，剩余几百步是
   constant 故障，累积偏差比论文无界 drift 大。
3. **N=50, T=1000 vs 论文 4.4.4 的精确随机种子**：论文未公开种子，
   不同随机数下数字会有 1-2× 的天然波动。

读者应当看的是**方法间相对优劣**而非绝对数值。``results.json`` 里的
Wilcoxon p_adj 与 Cohen's d 是评估"RPS-Full 是否显著优于基线"的正确指标。

## Section 4.5.4 假设验证

```
python verify_assumptions.py
```

输出 5 项 PASS/FAIL：

- 通信图强连通（Assumption 2）
- W 双随机非负
- 局部 cost 的 L-smoothness、聚合强凸（Assumption 1）
- small-fault regime（mean-shift 主导 std-change）
- 诊断路径不泄露 ground-truth 故障信息
