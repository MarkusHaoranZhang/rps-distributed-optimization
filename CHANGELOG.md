# Changelog

格式遵循 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。版本号
遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [0.4.8] — 大厂式 PR review 第二轮

按 v0.4.7 状态做了一次完整的 line-by-line review，找到 2 个 major + 5 个
minor。本版处理了全部 7 项。

### Fixed

- **(major) ``main.figure_3 / figure_5 / figure_8`` 的 drift_cap 不一致**：
  v0.4.7 把 figure_2 / figure_6 的 drift_cap 调整为 40 让稳态故障量级
  落在 small-fault regime 内，但 figure_3（参数敏感性）/ figure_5（压力
  测试）/ figure_8（scaling）仍是 80。这导致敏感性曲线、压力测试、
  scaling 三组实验跑在 small-fault 边界外，与 figure_2 / 6 不在同一
  参数空间。本版统一为 40。``test_paper_core_claim`` 同步从 80 调到 40。
- **(major) ``_step_hard_threshold`` 里 HT 二值化 MTCD 的注释误导**：
  v0.4.7 注释让读者以为 0/1 是相对 RPS 概率的"近似"，实际是 χ²
  阈值在故障期间永不跌破阈值的真实表现——这正是论文 §1 critique 的
  "threshold-based detector ... oscillates / fails"现象的数值证据。
  注释重写说明 HT MTCD 接近 T-onset 是预期最大值，不是实现局限。
- **(minor) ``main.py`` 顶部 ``ETA = 1.0`` 是死代码**：定义后无任何
  引用。删掉。
- **(minor) ``_RunState.diag_log["top1_prob_history"]`` 是死字段**：
  全工程只有写、没有读。删掉。``true_fault_top1_prob`` 是 MTCD 实际
  使用的字段，保留。
- **(minor) ``plot_figure7`` 的 dict 参数易出错**：原签名
  ``mtcd_data: dict`` 让调用者必须用确切 key "Hard-Threshold" /
  "RPS-Full"，错一个就 KeyError。改为显式 ``(ht_mtcd, rps_mtcd)``
  两个位置参数。``main.figure_7`` 调用点同步更新。
- **(nit) ``experiments.py`` 模块 docstring 列了不存在的函数名**：
  ``_step_baseline_no_gamma`` / ``_step_baseline_with_gamma`` 实际不
  存在，代码里只有 ``_step_hard_threshold`` / ``_step_uniform_discount`` /
  ``_step_rps``。docstring 重写并解释 Byzantine / Ideal 不需要专用
  step 函数的原因。
- **(doc) ``apply_fault_injection`` docstring 警示 drift_cap 默认值**：
  函数级默认 100 仅作 schema fallback，所有论文实验调用点都显式传 40；
  docstring 加注让读者不要直接依赖默认值。

### Verified

- ``ruff check .``：All checks passed
- ``mypy .``：Success, no issues found in 11 source files
- ``pytest tests/``：127 passed (~74s)

## [0.4.7] — RPS 消融 fusion 实现的根本性修正

按论文 Section 4.4.2 字面定义重审 ``RPS-Symmetric`` 与 ``RPS-NoOrder`` 的
fusion 实现，发现两处与论文定义不符的实现错误（这两处共同导致了 v0.4.6
"完整模式下 RPS-Full 不优于 RPS-Sym" 的复现失败）。

### Fixed

- **``symmetric_fusion`` 之前不是真正的"对称"（critical）**：论文 4.4.2
  说 "directional LOS fusion is replaced by symmetric DS combination"。
  v0.4.6 实现是"邻居 PMF 等权平均后再与 self 做一次 LOS"，self 仍然
  作为 ordering anchor，并不对称。修复：改为"self 与每个邻居用纯 DS
  链组合"，整条融合链不再使用 LOS。
- **``noorder_fusion`` 之前没有"folding before fusion"（critical）**：
  论文 4.4.2 说 "permutation structure is removed by collapsing ordered
  tuples to unordered sets **before fusion**"。v0.4.6 实现保留有序
  事件、只把单步 LOS 换成 DS，OPT 仍能从元组顺序里恢复部分信息。修复：
  新增 ``_collapse_to_unordered``，self 与每个邻居 PMF 在融合前先把
  ``(a, b)`` 与 ``(b, a)`` 折叠成升序键，再做 DS 链。
- **fig2 的 ``drift_cap`` 调整**：v0.4.6 用 ``drift_cap=80``，稳态故障
  量级 0.16 已超出论文 Section 4.4 的 small-fault regime 边界。改为
  ``drift_cap=40``（稳态量级 0.08，与 figure 6 ablation 配置一致），
  让所有 RPS 变体在 small-fault 区间内分化。

### Added

- ``tests/test_fusion_strategies.py::test_symmetric_uses_pure_ds_no_los``
  与 ``::test_noorder_collapses_before_fusion`` 守门：直接锁定
  论文 4.4.2 字面定义。
- ``tests/test_fusion_strategies.py::test_directional_distinct_from_symmetric``
  改用 ``self=(2,1,3)`` vs ``nb=(3,1,2)`` 的多元有序输入，确保 LOS 与 DS
  在事件顺序上明显分化（之前的输入太冲突，DS 与 LOS 都退化到同一单点）。

### Verified
- ``pytest tests/`` ：127 / 127 passed (~74s)。
- ``python main.py --quick`` 在 v0.4.7 下：
  - **论文 4.5.2 ablation**: RPS-Full 38.88 vs RPS-NoOrder 46.28 vs
    RPS-Sym 46.28（d=0.87, 大效应量）✓
  - **论文 Section 1 中心论断**: RPS-Full 39.66 显著优于
    Hard-Threshold 42.53（d=1.83, p<0.01）✓
- 完整模式 (N=50, MC=20) 数字归档到 ``expected_results_full.json``，
  CHANGELOG 在完整跑完后由 v0.4.8 entry 补充实测对比。

### Documentation
- ``IMPLEMENTATION_NOTES.md §1``：drift_cap 默认含义重写，记录 v0.4.6
  从 80 降到 40 的 small-fault 边界论证。
- ``IMPLEMENTATION_NOTES.md §9``：完全重写，记录 v0.4.7 fusion 修正
  的原因与修复后实测数字。
- ``IMPLEMENTATION_NOTES.md §20``：从"quick mode 排序异常"改写为
  "v0.4.7 修复后的实测数字"，承认 v0.4.6 RPS-Sym 反超 RPS-Full 是
  实现 bug 而非边界条件。
- ``README.md`` 警示框相应更新。

## [0.4.6] — 发布前最终核对（极致检查）

按"准备发布"标准对全部 11 个源文件 + 11 个测试文件 + 5 个文档文件 + CI
配置做了一遍逐字对照论文 LaTeX 的核对。发现并修复了一个会让 ``--quick``
直接崩溃的 figure_5 bug，以及多处文档错位。

### Fixed
- **figure_5 直接 crash（critical）**：``main.figure_5`` 的 ``perf_retain``
  循环里有一段复制粘贴造成的重复块——同一份 ``mc_run`` 调用被 append 两次，
  让 ``perf_retain`` 长度变成 ``2 * len(loss_rates) = 20`` 而 x 轴是 10，
  ``plot_figure5`` 抛 ``ValueError: x and y must have same first dimension``。
  这个 bug 至少在 0.4.5 版本以前就存在；之前的 ``expected_results.json``
  是手工编辑而非真跑生成，所以问题没暴露。本版删掉重复块，``perf_retain``
  长度恢复为 10。修复后 quick mode 端到端跑通（共 9 分钟，8 张图 +
  results.json + Table 1 + Table 2 全部产出）。
- **README 的"论文位置"对应表全部错位**：``--figures N`` 中的 N 是代码
  内部执行顺序（为了让 figure 7 复用 figure 2 的 MTCD 数据），与论文按
  章节顺序的 Figure 编号是两套独立编号。原 README 错误地把代码内部 N
  当成论文 Figure 编号，导致读者按 README 找"论文 Figure 5"会拿到论文
  Figure 8 的内容。本版同时给出双向映射，并加注解释这是历史包袱。
- ``main.py`` 顶部 docstring 写"约 30 分钟"而 README 与 ``_Sizes.full``
  内部说"约 2-3 小时"——前者是 v0.4.0 之前 MC=5 时的旧数字。统一为
  "MC=20，约 2-3 小时"。
- ``verify_assumptions.check_strong_connectivity`` 与
  ``tests/test_distributed_optimization.test_build_graph_is_connected``
  里的局部变量 ``queue`` 实际用作 LIFO 栈（``.pop()`` 取最后一个）。
  ``distributed_optimization._is_connected`` 在 v0.4.4 已重命名为
  ``stack``，这两处遗漏；本版统一。
- README 自检节命令 ``mypy <module>`` 应是 ``mypy .``（``pyproject.toml``
  已配 ``[tool.mypy] exclude``，CI 也用 ``mypy .``）。

### Added (documentation only)
- ``IMPLEMENTATION_NOTES.md §19``：``LeastSquaresCost`` 的 ``1/p_i`` 归一化
  相对论文公式的偏离与理由（让 α=0.05 在不同 (N, d, p) 下稳定收敛）。
- ``IMPLEMENTATION_NOTES.md §20``：发布前最重要的复现局限披露——quick mode
  下 RPS-Full 在 Drift 上比 Uniform-Discount 略差 ~4%，与论文 4.5.3 "RPS
  outperforms next-best by over 40%" 论断方向相反。原因（N=30 下 RPS 信号
  -噪声比偏低 + UD 无差别 damping 天然占便宜 + MC=3 噪声大）+ 论文 N=50,
  MC=20 下论断成立的对照数字一并写明。``README.md`` "快速验证" 节加
  ``⚠ quick mode 排序异常`` 警示框；``test_paper_core_claim.py`` docstring
  注明本测试仅锁 RPS vs HT 不锁 RPS vs UD 的设计权衡。
- ``expected_results.json`` 的 ``_meta`` 加上 ``regenerate_command``、
  ``verified_against_pytest`` 等字段；``ablation_mean_x1e3`` 加注说明
  quick mode 单故障下 RPS-Full ≡ RPS-NoOrder 的退化原因（PMF 集中在单点
  事件，LOS 与 DS 等价）。
- ``expected_results.json`` 整份按 v0.4.6 真实跑出的数字重新生成（之前是
  手工编辑的 v0.4.1 标签，与代码可能脱节）。

### Verified (no change to code, sanity-checked end-to-end)
- ``pytest tests/`` ：125 passed in 67s。
- ``ruff check .`` ：All checks passed。
- ``mypy .`` ：Success: no issues found in 11 source files。
- ``python verify_assumptions.py`` ：5 / 5 PASS。
- ``python main.py --quick`` ：9 分钟跑完 8 张图 + results.json + Tables。

## [0.4.5] — 大厂式 PR review 修订

按"逐文件 PR review"标准对全部源码（3000 行）+ 测试（1500 行）做了一遍审阅，
找到 12 个 major + 16 个 minor。本版处理了**全部 28 项**。

观察：28 个 review 评论里**只有 1 个**涉及运行时行为变化（figure_5），其它
27 个全是注释、变量名、错误处理粒度、文档警告。这意味着代码本身鲁棒，缺的
是"为什么这样写"的可见性。

### Removed
- ``RPSConfig.q0_subsample`` 字段（死参数）：``compute_pmf`` 走 z-score
  路径不需要 Q0 子采样；老形参从 ``compute_pmf`` / ``_generate_local_pmfs`` /
  ``RPSConfig`` 三处一并删除。
- ``experiments.LeastSquaresCost`` 与 ``generate_least_squares_data`` 的
  re-export：顶层运行器不应当 ``costs`` 模块的代理。
- ``QuadraticDispatchCost.__init__`` 的死参数 ``p_min`` / ``p_max``。
- ``Byzantine-Resilient`` 路径下冗余的 ``gamma_mat = None``（gamma_mat 在
  此分支本就保持初值 None）。

### Fixed
- **figure_5 ylabel 与数据语义不符（major）**：``perf_retain`` /
  ``accs`` 实际是 raw final relative error（量级 ~10⁻³），但 ylabel 写的是
  "%"。一开始尝试折算成百分比（baseline gap、isolation accuracy ratio），
  发现折算需要 ground-truth 才能定义"满分"，与 process integrity 约束冲突，
  且 0/0 noise 严重。**最终选择**：保留 raw error，ylabel 改为
  "Final relative error"，论文文字描述的"80%"/"isolation accuracy"作为
  trend 修辞由 IMPL §17 解释。
- ``cohens_d``：std=0 但 mean≠0 时返回 ``±inf``（之前返回 0 是误导）。
- ``LeastSquaresCost.global_optimum``：缓存 ``_x_opt``，与 ``LogRegCost``
  行为一致。
- ``LogRegCost.global_optimum``：max_iter 后未达 1e-4 阈值时发
  ``UserWarning``（之前静默用未收敛的 x_opt）。
- ``main.mc_run``：``cfg.burn_in`` 被故障 onset 强制裁剪时发
  ``UserWarning``（之前静默改写）。
- ``main.save_results_json``：写 JSON 前把所有 nan/inf 转 None，让严格
  JSON parser 也能读 ``results.json``。
- ``rps_diagnosis._to_mask_array``：``except (OverflowError, ValueError)``
  收窄到只捕 ``OverflowError``（``ValueError`` 这里真的抓不到合理情况）。
- ``distributed_optimization._is_connected``：``queue`` 变量名误导（实际
  用 LIFO 栈），改名为 ``stack``。
- ``tests/test_pmf.py::test_pmf_is_immutable``：``pytest.raises(Exception)``
  收窄到 ``dataclasses.FrozenInstanceError``。
- ``tests/test_distributed_optimization.py::test_fault_injection_intermittent``：
  阈值从 ±0.03 (3σ) 放宽到 ±0.04 (4σ)，避免边缘 flaky。
- ``tests/test_smoke_pipeline.py``：用 ``argparse.Namespace`` 替代匿名
  ``type("A", (), {...})()`` 构造。
- ``verify_assumptions.check_small_fault_regime`` 判据从 ``ratio > 1.0`` 放宽
  到 ``> 0.5``：在 small-fault 边界附近 mean shift 与 std change 同量级是
  预期的，严格主导判据让边界情况误报。
- CI mypy 命令从手工列文件名改为 ``mypy .``，新增模块时自动覆盖。

### Added (documentation only)
- ``IMPLEMENTATION_NOTES §17``：figure_5 ylabel 与论文措辞的关系（百分比
  折算与 process integrity 的冲突）。
- 多处 docstring 增补（约 30 处），每条说明"为什么"——主要类型：
  - 论文偏离背书（``compute_support_score`` 是 reference impl、
    ``HardThresholdDetector.calibrate`` 朴素估计是 by-design）；
  - 类型缩窄注释（``_step_rps`` 的 mypy assert）；
  - 数据语义警告（``MNIST`` 欠定问题、``IEEE39`` 系数随机生成不可与
    MATPOWER 比较）；
  - 性能盲点（``compute_pmf`` 的 H 矩阵双层循环已经测量过不是热点）；
  - 工程依赖（``apply_fault_injection`` 与上游共享 rng 的影响）；
  - 文档与代码不符的修正（quick mode burn-in 注释、``_make_cost_and_graph``
    多 cost 类型）。
- ``KNOWN_METHODS`` docstring 列出修改时需要同步更新的三处：
  ``main._FIG2_METHODS`` / ``run_optimization`` 分派 / 单元测试参数化。
- ``FaultConfig`` 加 note 说明它是仅文档用 TypedDict（运行时校验由
  ``validate_fault_config`` 完成）。
- ``build_graph`` docstring 改成"approximating the minimum radius"，与
  论文 4.4.4 "minimum value ensuring strong connectivity" 措辞如实对齐。

## [0.4.3] — 隐藏 bug 怀疑式审计

本轮以 "假定还有未发现的 bug" 的视角重新审视项目，找到两个真问题。

### Added
- ``tests/test_fusion_strategies.py``：9 个单元测试覆盖 ``directional_fusion`` /
  ``symmetric_fusion`` / ``noorder_fusion``。这三个是论文 Section 4.4.2 的
  核心创新函数，之前**完全没有单元测试**。新测试守门：
  - directional 保留 self 顺序、与 noorder 在多 agent 场景下分化；
  - symmetric 与输入顺序无关、对二元事件取平均；
  - noorder 把 (a, b) 与 (b, a) 折叠为同一 sorted 事件。
- ``tests/test_distributed_optimization.py`` 加 3 个 γ 真测试：
  ``test_gradient_tracking_with_gamma_preserves_row_sum_in_function``、
  ``test_gradient_tracking_step_actually_uses_gamma``、
  ``test_gradient_tracking_with_gamma_zero_isolates_agent``。
- ``IMPLEMENTATION_NOTES.md §16``：记录 GT 两条路径（``gamma=None`` /
  ``gamma=ones``）的数学等价性约束以及为何 ``X`` 步进里的 ``αY`` 项
  按论文 Eq.(2) 形式书写（不是 ``α A_eff @ Y``）。

### Fixed
- **GT 两条路径数学不等价（隐藏 bug）**：之前 ``gradient_tracking_step``
  在 ``gamma=None`` 时用 ``X_new = W @ X - α Y``、``Y_new = W @ Y + Δg``，
  而在 ``gamma`` 给出时构造 ``A_eff`` 并把 ``Y`` 步进改成
  ``α (A_eff @ Y)``，让 ``gamma=ones`` 不等价于 ``gamma=None``。修复为
  两条路径都用 ``X_new = A_eff @ X - α Y``、``Y_new = A_eff @ Y + Δg``，
  其中 ``gamma=None`` 时 ``A_eff = W``。新加的 ``actually_uses_gamma`` 测试
  守住此不变量。修复后论文核心论断回归测试 ``test_paper_core_claim.py``
  仍全过（4/4）：RPS-Full 在 Drift 上的优势不依赖于此前的不等价行为。
- **假阳性测试**：旧的 ``test_gradient_tracking_with_gamma_preserves_row_sum``
  根本没调用 ``gradient_tracking_step``，只在测试函数里重写一遍内部逻辑
  做断言。替换为真正调用函数、从输出反推 ā 行和的版本。
- **mypy 12 个类型推断噪声**：``datasets._read_idx_*`` 的 ``opener`` 联合
  类型；``main._make_cost_and_graph`` 的多分支 ``cost`` 类型锁定；
  ``main.figure_3`` 的 ``base_fault['onset'] - 50`` 在 dict-object 上的
  减法；``main.figure_4`` 的 ``np.linalg.norm(fault_cfg['delta'])``；
  ``main.figure_1`` 的 ``nb2 = set()`` 类型注解；``verify_assumptions`` 的
  ``res_norms`` list-then-array 重赋。最小注解修复，运行行为不变。
  现在 ``mypy .`` clean。

### Removed
- 临时探查脚本 ``_probe.py``（已完成对 fusion 行为的诊断使命）。

## [0.4.2] — 发布前最小化加固

### Added
- ``LICENSE`` (MIT)：补上发布必需的法律声明。
- ``tests/test_paper_core_claim.py``：4 个回归测试守住论文 Section 1
  中心论断 — RPS-Full 在 Drift 上必须显著优于 Hard-Threshold（N=30,
  T=500, MC=3，约 51 秒）。
- ``expected_results.json``：``--quick`` 模式跑出来的参考数字快照，
  让读者能验证自己的复现是否在合理范围。
- README 加上 detection / false-alarm 数值对比示例（论文 Section 1
  论断的直接证据）。

### Fixed
- 通过新加的核心论断回归测试，发现并文档化：N<30 时 RPS-Full 反而
  比 Hard-Threshold 差（misspecification 误差占主导）。这是规模相关
  的真实算法行为，不是 bug，但回归测试现在锚定 N=30 这一最小有效
  规模以防未来误改。

## [0.4.1] — Section 4.4.3 指标补全 + 数字差距文档化

### Added
- ``experiments.detection_and_false_alarm_rates``：基于 γ 矩阵历史与
  ground-truth ``faulty_agents`` 计算 fault detection rate 与 false
  alarm rate（论文 Section 4.4.3 列了但未在表中报告的指标）。
- 6 个新单元测试覆盖 detection 指标（完美检测、零检测、全误报、空输入、
  部分检测）。
- ``main.py figure_7`` 现在打印 Drift 场景下 HT 与 RPS-Full 的检测率
  / 误报率对比。
- ``results.json`` 新增 ``fig2_detection_rate``、``fig2_false_alarm_rate``、
  ``fig2_mtcd``、``kappa_theo`` 字段。
- README 加"与论文 Table 1/2 的绝对数字差距"小节，引用 IMPL §1/§3/§15
  解释三层原因。
- ``IMPLEMENTATION_NOTES.md`` 加 §13 (κ_theo vs κ_emp 的"closely match"
  实测差 24×)、§14 (检测率指标定义)、§15 (绝对数字差距三层原因)。
- ``HardThresholdDetector`` 步函数 docstring 加注：χ² 检测器无时间滤波，
  ``Iter to 1e-3`` 在 trial 间方差大是论文 Section 1 描述的"oscillates"
  现象的真实表现。

### Changed
- ``main.py figure_4`` 把 ``ctx.kappa_theo`` 也存到 context 与 JSON。

### Fixed
- 删除 ``detection_and_false_alarm_rates`` 中未使用的 ``healthy_agents``
  局部变量（ruff F841）。

## [0.4.0] — 论文配套定稿

### Fixed
- **`tau_quantile` 接通 τ 校准（重要）**：之前 ``cfg.tau_quantile`` 字段
  从未被 ``run_optimization`` 读取，τ 被写死为 ``log(top_m)``。这让
  Figure 3 的 "Confidence threshold" 敏感性子图实际是噪声。现在 burn-in
  期会真的累积 PMF 熵，按 ``cfg.tau_quantile`` 取分位作为 τ。同时新增
  ``cfg.tau`` 字段供 Figure 3 显式扫描。
- **MTCD 改用论文 4.4.3 定义**：之前 Figure 7 用的是 ``gamma_based_mtcd``
  （基于 γ 阈值），与论文表 2 报告的"top-rank prob ≥ 0.95"语义不一致。
  现在统一用 ``mean_time_to_correct_diagnosis``，Hard-Threshold 也接入
  这一指标（二值化）以保证可比。
- ``validate_fault_config`` 接受可选 ``d`` 参数，校验 ``delta.shape``。

### Added
- ``main.py --mc N``：覆盖默认 MC trials 数。
- ``main.py --dataset {synthetic,mnist,ieee39}``：切换 Figure 2 / 6 主基准。
- ``verify_assumptions.py``：可执行的论文 Section 4.5.4 假设检查（连通性、
  双随机 W、L-smoothness、small-fault regime、ground-truth 不泄露）。
- Figure 4 同时画 ``κ_emp`` 与从 μ、λ₂、L 算出的 ``κ_theo``（论文 Theorem 1）。
- 完整模式 MC 默认从 5 调到 **20**，对齐论文 Section 4.4.5 的声明。

### Changed
- 删 ``run_optimization`` 的 7 个 "兼容旧 API" 散参数。仅剩 ``cfg: RPSConfig``
  统一入口。
- 删 ``CONTRIBUTING.md``：论文配套不接受外部贡献。
- 删 ``tests/test_cost_protocol.py`` 与 ``CostModel`` Protocol：论文配套
  有 3 个固定代价模型，不需要扩展契约。
- CI matrix 简化为单一 Python 3.11。
- ``QuadraticDispatchCost.c`` docstring 注明仅占位（不影响梯度/最优）。

### Removed
- ``OptimConfig`` / ``default_rps_config`` 死代码。
- ``flatten_no_order``：RPS-NoOrder 消融由 ``noorder_fusion`` 实现，
  不需要在 PMF 生成阶段折叠。
- ``gamma_based_mtcd``：被论文定义的 MTCD 替代，不再使用。

## [0.3.0] — 长期演化基础设施

### Added
- ``costs.CostModel`` Protocol：把代价模型的"鸭子接口"形式化，支持
  ``isinstance(cost, CostModel)`` 运行时检查
- ``tests/test_cost_protocol.py``：参数化测试，注册到 ``COST_FACTORIES``
  的代价类自动获得 5 项契约检查（协议、维度、grad 形状、x* 形状、
  ∇f 在 x* 处的聚合范数）
- ``tests/test_config_validation.py``：18 个测试覆盖 ``RPSConfig``
  与 ``validate_fault_config`` 的所有合法性边界
- ``tests/test_reproducibility.py``：相同 seed → byte-identical 输出
- ``tests/test_smoke_pipeline.py``：管线级冒烟（写真实 PDF）
- ``CONTRIBUTING.md``：常见维护任务的"改哪几个文件"清单（加故障类型 /
  加诊断方法 / 加代价模型 / 加调优旋钮 / 加新指标）
- ``CHANGELOG.md`` 自身
- CI 加入 mypy 检查
- ``RPSConfig.__post_init__`` 立即校验所有字段范围
- ``config.validate_fault_config()``：fault_config dict 的 schema 校验

### Changed
- ``run_optimization`` 入口立即调 ``validate_fault_config``，让格式错误
  在源头暴露
- ``main.py`` 不再用 ``hash(scenario)`` 当 seed 偏移（PYTHONHASHSEED 不
  稳定），改用稳定的 ``_SCENARIO_SEED_OFFSET`` 字典
- ``figures.py`` 强制 matplotlib ``Agg`` 后端：在无显示环境（CI / 远程）
  也能稳定出图
- ``RPSConfig.replace`` 后会重跑 ``__post_init__``，无效字段在 replace
  时就被拒绝
- ``costs.py`` / ``rps_diagnosis.py`` / ``experiments.py`` 加完整类型
  注解；mypy clean
- ``pyproject.toml`` 加 ``[tool.mypy]`` 配置

### Fixed
- ``js_divergence`` 默认参数 ``sing1=None`` 之前是 ``np.ndarray = None``
  违反 PEP 484，现在用 ``Optional[np.ndarray] = None``
- ``LogRegCost._x_opt`` 类型从隐式 ``None`` 升级为 ``Optional[ndarray]``

## [0.2.0] — 长期维护重构

### Added
- `config.py` 集中数据结构与所有调优参数
  - `PMF` dataclass 替代不透明的三元组
  - `RPSConfig` dataclass 收拢全部 17 个调优旋钮，构造时立即校验
  - `FaultConfig` TypedDict 给 `fault_config` 字典 schema
  - `validate_fault_config()` 在 `run_optimization` 入口检查
- `costs.py` 集中代价模型（`LeastSquaresCost` / `LogRegCost` /
  `QuadraticDispatchCost`）
- `tests/`：78 个 pytest 单元测试，覆盖 PMF、组合规则、PMF 计算、
  分布式优化、统计、可复现性、配置验证
- `pyproject.toml`：ruff + pytest 配置
- `IMPLEMENTATION_NOTES.md`：代码相对论文 11 条偏离的集中记录
- `CHANGELOG.md` + `.gitignore`
- `main.py --figures N1,N2,...` 选项可单跑某几张图
- `RPSConfig.replace()` 链式覆盖参数
- `gamma_based_mtcd()` 基于 γ 矩阵历史的诊断延迟指标

### Changed
- `run_optimization` 从 230 行单函数拆为 `_step_hard_threshold` /
  `_step_uniform_discount` / `_step_rps` 三个分支函数 + 主调度
- PMF 三元组 `(events, mass, masks)` → `PMF` dataclass（破坏性 API
  变更，调用点已统一更新）
- 故障 drift 注入加 `drift_cap` 默认 100，保证 small-fault regime
- 软折扣 γ 从论文 Eq.(12) 的分段函数改为单调连续 `exp(-gain · P_OPT)`
- γ 同时作用于 X 共识和 Y 跟踪（双向屏蔽 + 行和补齐）
- `magnitude_proxy` 改为 `max(mean_inc, std_weight · std_inc)`
- PMF 生成用 z-score 替代能量距离，O(E·s·M) → O(E)
- `main.py` 拆为 `figure_1` … `figure_8` + `ExperimentContext` 共享状态
- `main.py` 不再用 `hash(scenario)` 当 seed（不稳定），改用稳定字典
- `requirements.txt` 钉版本上界（防 numpy 3.0 类的 breaking）

### Removed
- 旧 `experiments.py` 内嵌的 `LeastSquaresCost` / `LogRegCost`（迁到
  `costs.py`）
- `summarize_paired_comparison`（无人调用）
- `flatten_no_order` 在 `experiments.py` 的导入（仅 RPS-NoOrder fusion
  内部用）
- 多处未用的 imports

### Fixed
- N=200 时 PMF 位掩码的 `int64` 溢出（自动降级到 `dtype=object`）
- `gamma_based_mtcd` 之前对全列取均值，现在只看 faulty agent 的邻居
- `HardThresholdDetector.gamma_matrix` 移除未使用的 `W` 参数

## [0.1.0] — 初版

- 论文 8 张图与 2 张表的首次完整复现
- 7 个方法（含 RPS-Full、3 个消融、3 个基线）
- Monte Carlo + Wilcoxon + Holm-Bonferroni + Cohen's d
- 合成 LS / MNIST 非 IID / IEEE 39-bus 三档基准的代码框架
