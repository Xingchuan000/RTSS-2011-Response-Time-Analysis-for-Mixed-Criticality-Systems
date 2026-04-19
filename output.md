# AMC 复现项目阶段输出汇总

## 阶段 A 输出

### 1. 项目目录树

```text
/Users/x1ngchuan/Documents/AMC
├── README.md
├── environment.yml
├── pyproject.toml
├── amc_py
│   ├── __init__.py
│   ├── models.py
│   ├── priorities.py
│   ├── rta.py
│   ├── bounds.py
│   ├── smc.py
│   ├── amc.py
│   ├── generator.py
│   ├── experiments.py
│   └── utils.py
├── scripts
│   └── run_experiments.py
└── tests
    ├── test_models.py
    └── test_smoke.py
```

### 2. 新增文件清单

- `README.md`
- `environment.yml`
- `pyproject.toml`
- `amc_py/__init__.py`
- `amc_py/models.py`
- `amc_py/priorities.py`
- `amc_py/rta.py`
- `amc_py/bounds.py`
- `amc_py/smc.py`
- `amc_py/amc.py`
- `amc_py/generator.py`
- `amc_py/experiments.py`
- `amc_py/utils.py`
- `scripts/run_experiments.py`
- `tests/test_models.py`
- `tests/test_smoke.py`

### 3. 核心数据结构说明

- `Criticality`：关键级枚举，取值为 `LO / HI`。
- `Task`：任务模型，包含 `period / deadline / c_lo / c_hi / criticality`，并做参数合法性校验。
- `TaskSet`：任务集合封装，支持 `add`、迭代、长度查询、`from_iterable`。
- `SchedulabilityResult`：可调度性分析结果容器。
- `PriorityAssignmentResult`：优先级分配结果容器。

### 4. 环境安装命令

```bash
cd /Users/x1ngchuan/Documents/AMC
conda env create -f environment.yml
conda activate amc-repro
```

### 5. 测试运行命令

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q
```

### 6. 当前项目是否已可 import

- 是，`import amc_py` 可成功执行，且核心符号可导入。

### 7. 当前阶段尚未完成的部分

- `priorities.py / rta.py / bounds.py / smc.py / amc.py / generator.py / experiments.py / utils.py` 在阶段A时仅为骨架或占位。
- 算法细节与实验框架需在后续阶段继续实现。

---

## 阶段 B 输出

### 已完成的小任务

1. B1 静态优先级排序：实现 `DM / CM / CrMPO / reindex`。
2. B2 固定点求解器：实现 `solve_fixed_point`，支持收敛、超截止期、最大迭代处理。
3. B3 LO 模式分析：实现 `compute_r_lo` 与 `analyze_lo_mode`。
4. B4 HI 模式分析：实现 `compute_r_hi` 与 `analyze_hi_mode`。
5. B5 UB-H&L：实现 `ub_l_test / ub_h_test / ub_hl_test`。
6. B6 SMC：实现 `compute_smc_response_time / smc_sched_test`。
7. B7 SMC-no：实现 `compute_smc_no_response_time / smc_no_sched_test`。
8. B8 测试：新增 `test_priorities / test_rta_basic / test_bounds / test_smc`，并通过。

### 新增/修改文件

- `amc_py/priorities.py`
- `amc_py/rta.py`
- `amc_py/bounds.py`
- `amc_py/smc.py`
- `tests/test_priorities.py`
- `tests/test_rta_basic.py`
- `tests/test_bounds.py`
- `tests/test_smc.py`

### 已实现函数（接口清单）

- `sort_by_deadline_monotonic(tasks) -> list[Task]`
- `sort_by_criticality_monotonic(tasks) -> list[Task]`
- `sort_by_crmpo(tasks) -> list[Task]`
- `reindex_priorities(tasks) -> dict[str, int]`
- `solve_fixed_point(recurrence_fn, start_value, deadline, max_iter=1000) -> int | None`
- `compute_r_lo(task, higher_priority_tasks) -> int | None`
- `analyze_lo_mode(tasks, ordered_tasks) -> SchedulabilityResult`
- `compute_r_hi(task, higher_priority_hi_tasks) -> int | None`
- `analyze_hi_mode(tasks, ordered_tasks) -> SchedulabilityResult`
- `ub_l_test(tasks) -> bool`
- `ub_h_test(tasks) -> bool`
- `ub_hl_test(tasks) -> bool`
- `compute_smc_response_time(task, higher_priority_tasks) -> int | None`
- `smc_sched_test(ordered_tasks) -> SchedulabilityResult`
- `compute_smc_no_response_time(task, higher_priority_tasks) -> int | None`
- `smc_no_sched_test(ordered_tasks) -> SchedulabilityResult`

### LO 模式 / HI 模式 / UB-H&L / SMC / SMC-no 实现说明

- LO 模式：全部任务按 `C(LO)` 进入标准固定点 RTA。
- HI 模式：仅 HI 任务参与分析，干扰仅来自更高优先级 HI 任务，按 `C(HI)` 计算。
- UB-H&L：`ub_hl_test = ub_l_test && ub_h_test`，排序采用 DM。
- SMC：干扰项按 `C_j(min(L_i, L_j))`。
- SMC-no：干扰项按 `C_j(L_i)`，相较 SMC 更保守。

### 最小使用示例（展示 SMC 与 SMC-no 差异）

- 任务集：
  - `tau_lo = (T=5, D=5, C_lo=1, C_hi=4, LO)`
  - `tau_hi = (T=10, D=9, C_lo=1, C_hi=2, HI)`
- 在 DM 顺序下：
  - `SMC` 对 `tau_hi` 响应时间 `R=3`，可调度。
  - `SMC-no` 对 `tau_hi` 超过截止期，不可调度。

### 测试命令

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q
```

### 测试结果

- `13 passed in 0.02s`

### 当前哪些分析器已可用

- 优先级排序：`DM / CM / CrMPO`
- 固定点求解器：`solve_fixed_point`
- 模式分析：`LO mode / HI mode`
- 可行性分析：`UB-L / UB-H / UB-H&L`
- 混合关键级分析：`SMC / SMC-no`

### 当前仍未覆盖的边界情况

- 未系统覆盖大量 `D < T` 场景的随机回归测试。
- `max_iter` 达上限但未超截止期时，当前统一返回 `None`，未细分诊断码。
- `reindex_priorities` 对同名任务未做冲突检测。
- 阶段C的 `AMC-rtb / AMC-max / OPA` 尚未实现。

---

## 阶段 C 输出

### 已完成的小任务

1. C1 AMC-rtb（Method 1）实现完成。
2. C2 AMC-max（Method 2）实现完成。
3. C3 OPA（Audsley）实现完成，并可与 `smc/smc_no/amc_rtb/amc_max` 组合。
4. C4 统一评估入口 `evaluate_taskset(tasks, method, priority_policy)` 实现完成。
5. C5 阶段C测试完成，新增 `test_amc_rtb.py / test_amc_max.py / test_opa.py`。

### AMC-rtb 实现说明

- 核心函数：
  - `compute_amc_rtb_response_time(task, higher_priority_tasks, r_lo_map)`
  - `amc_rtb_sched_test(ordered_tasks)`
- 实现流程：
  1. 先求每个任务 `R_LO`。
  2. 对 HI 任务使用切换方程：
     `R = C_i(HI) + Σ_hpLO ceil(R_LO_i/T_j)*C_j(LO) + Σ_hpHI ceil(R/T_j)*C_j(HI)`
  3. 返回 `max(R_LO_i, R_HI_i, R_MC_i)` 作为最终响应时间。
- 对 LO 任务，AMC-rtb 直接使用 `R_LO`。

### AMC-max 实现说明

- 核心函数：
  - `candidate_switch_points(task_i, ordered_tasks, r_lo_i)`
  - `compute_M(task_k, s, t)`
  - `compute_amc_max_response_time_for_switch(task_i, ordered_tasks, s)`
  - `compute_amc_max_response_time(task_i, ordered_tasks)`
  - `amc_max_sched_test(ordered_tasks)`
- 候选切换点处理：
  - 在 `[0, R_LO_i)` 区间内枚举高优先级 LO 任务释放点 `k*T_j`，去重排序得到 `S`。
- 对每个 `s∈S`：
  - 计算一个模式切换响应时间上界，最后取最大值。
- 最终同样与 `R_LO / R_HI` 取最大得到响应时间。

### OPA 实现说明

- 核心函数：
  - `lowest_priority_candidates(tasks)`
  - `audsley_opa(tasks, sched_test_fn)`
- 输入输出约定：
  - 输入：任务列表 + 一个“给定完整顺序后可判定可调度性”的函数。
  - 输出：`PriorityAssignmentResult`
    - `success=True` 时，`priorities` 给出 `{task_name: priority_index}`（`0` 为最高）。
    - `success=False` 时，表示不存在满足该分析器的 OPA 可行顺序。

### evaluate_taskset 接口说明

- 函数：`evaluate_taskset(tasks, method, priority_policy)`
- `method` 支持：
  - `ub_hl`
  - `smc`
  - `smc_no`
  - `amc_rtb`
  - `amc_max`
- `priority_policy` 支持：
  - `dm`
  - `crmpo`
  - `opa`
- 行为：
  - 先按策略得到有序任务（`opa` 时先运行 Audsley）。
  - 再调用对应分析器并返回统一 `SchedulabilityResult`。

### 示例结果

1. AMC-rtb 与 SMC 对比（同一任务集）
   - `SMC`: 不可调度，响应时间 `{'tau_lo': 2}`
   - `AMC-rtb`: 可调度，响应时间 `{'tau_lo': 2, 'tau_hi1': 6, 'tau_hi2': 14}`

2. AMC-max 与 AMC-rtb 对比（同一任务集）
   - `AMC-rtb`: 可调度
   - `AMC-max`: 可调度
   - 本示例下两者响应时间一致：`{'tau_lo': 2, 'tau_hi1': 6, 'tau_hi2': 14}`

3. OPA 运行示例
   - 任务集 `{tau3, tau1, tau2}` 在 `amc_rtb` 分析器下的 OPA 结果：
   - `success=True`
   - `priorities={'tau2': 0, 'tau1': 1, 'tau3': 2}`

### 本阶段测试结果

- 命令：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q
```

- 结果：`20 passed in 0.02s`

### 当前限制

- AMC-max 当前实现采用论文/参考实现风格的离散候选点枚举，尚未做性能优化（大规模任务集时耗时会增加）。
- 目前仅覆盖双关键级模型（`LO/HI`），未扩展到多关键级。

---

## 阶段 D 输出

### 已完成的小任务

1. D1 实现 `uunifast(num_tasks, total_util)`。
2. D2 实现周期与任务参数生成：
   - `sample_period_log_uniform(min_period, max_period)`
   - `generate_task(...)`
   - `generate_taskset(...)`
3. D3 实现实验批处理：
   - `run_utilization_sweep(...)`
   - `run_cf_sweep(...)`
   - `run_cp_sweep(...)`
   - `run_taskset_size_sweep(...)`
4. D4 实现 weighted schedulability：
   - `compute_weighted_schedulability(...)`
5. D5 实现绘图模块：
   - `plot_schedulable_percentage(...)`
   - `plot_weighted_schedulability(...)`
6. D6 实现脚本入口：
   - `scripts/run_single_example.py`
   - `scripts/run_small_experiment.py`
   - 并保留 `scripts/run_experiments.py` 作为兼容入口。

### 任务集生成器参数说明

- `num_tasks`：任务数。
- `total_util`：目标总利用率。
- `min_period/max_period`：周期采样范围（log-uniform）。
- `cf`：HI 任务放大系数，按 `C(HI)=cf*C(LO)` 构造。
- `cp`：HI 任务比例（0~1）。
- `seed`：随机种子，保证实验可复现。
- `deadline_equals_period`：默认 `True`，即 `D=T`（论文风格隐式截止期）。

### 支持的实验 sweep 类型

- `utilization sweep`：固定其余参数，仅改变总利用率。
- `cf sweep`：固定其余参数，仅改变 `CF`。
- `cp sweep`：固定其余参数，仅改变 `CP`。
- `taskset size sweep`：固定其余参数，仅改变任务数 `N`。

### CSV 输出字段说明

原始结果 CSV（例如 `small_util_sweep.csv`）包含：
- `sweep_type`
- `sweep_value`
- `taskset_id`
- `seed`
- `method`
- `priority_policy`
- `num_tasks`
- `target_total_util`
- `actual_total_util_lo`
- `actual_total_util_hi`
- `cf`
- `cp`
- `min_period`
- `max_period`
- `schedulable`
- `details`

加权统计 CSV（例如 `small_util_sweep_weighted.csv`）包含：
- `sweep_value`
- `util_sum`
- `weighted_sum`
- `taskset_count`
- `schedulable_ratio`
- `weighted_schedulability`

### 绘图函数输入输出说明

- `plot_schedulable_percentage(results, x_col, output_path, title)`
  - 输入：原始结果 DataFrame（含 `schedulable`）
  - 输出：可调度率折线图（PNG）
- `plot_weighted_schedulability(weighted_results, x_col, output_path, title)`
  - 输入：加权统计 DataFrame（含 `weighted_schedulability`）
  - 输出：weighted schedulability 折线图（PNG）

### 如何运行单示例脚本

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/run_single_example.py
```

### 如何运行小规模实验脚本

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/run_small_experiment.py
```

### 生成的示例 CSV 文件路径

- `/Users/x1ngchuan/Documents/AMC/outputs/small_util_sweep.csv`
- `/Users/x1ngchuan/Documents/AMC/outputs/small_util_sweep_weighted.csv`

### 生成的示例图片路径

- `/Users/x1ngchuan/Documents/AMC/outputs/small_util_schedulable_percentage.png`
- `/Users/x1ngchuan/Documents/AMC/outputs/small_util_weighted_schedulability.png`

### 当前实验框架与论文设定的差异说明

1. 当前任务集生成使用整数离散化（`C` 与 `T` 取整），与论文连续参数模型存在数值差异。
2. 当前默认 `D=T`，但未扩展到论文中更复杂的截止期分布实验。
3. 当前 sweep 是轻量可复现实验框架，尚未完全复刻 `mceval` 中全部评测规模与历史参数组合。
4. 当前统计已支持 `schedulable ratio` 与 `weighted schedulability`，但尚未加入更多论文图表中的多维对比（例如多方法同图批量汇总脚本）。

### 新增/修改文件（阶段 D）

- `amc_py/generator.py`
- `amc_py/experiments.py`
- `scripts/run_single_example.py`
- `scripts/run_small_experiment.py`
- `scripts/run_experiments.py`
- `tests/test_generator.py`
- `tests/test_experiments_d.py`

### 测试与运行结果

- 全量测试命令：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q
```

- 结果：`26 passed in 0.93s`

### 当前限制

- 目前绘图风格是基础科研图模板，尚未提供论文排版级统一主题模板。
- 大规模 sweep（例如每点上千 taskset）尚未做并行化优化。
- 当前脚本默认单分析器单策略执行，批量多方法对比仍建议在阶段 E 增强。

---

## 阶段 E 输出

### 已完成的小任务

1. E1 与论文趋势做对照：新增趋势回归测试，并做 50 组随机任务集统计。
2. E2 与 `mceval` 做行为对照：定位任务生成器、AMC/SMC/OPA、实验驱动代码位置并形成差异报告。
3. E3 补充回归测试：新增 `tests/test_trends_e.py`，验证关键趋势与 UB-H&L 示例。
4. E4 补充最终 README：完善项目背景、环境安装、测试、示例、实验、目录、限制与差异说明。
5. E5 输出最终交付说明：明确当前能力、已对齐部分与后续建议。

### 论文趋势对照结论

- 检查目标趋势：`AMC-max >= AMC-rtb >= SMC >= SMC-no`（可调度性意义下的不弱于关系）。
- 在 50 组可复现随机任务集（`num_tasks=6, U=0.75, CF=2.0, CP=0.5`）上的统计结果：
  - `SMC-no` 可调度数量：12
  - `SMC` 可调度数量：12
  - `AMC-rtb` 可调度数量：14
  - `AMC-max` 可调度数量：15
- 趋势违规计数：
  - `SMC-no -> SMC` 违规：0
  - `SMC -> AMC-rtb` 违规：0
  - `AMC-rtb -> AMC-max` 违规：0
- 结论：当前实现与论文预期趋势一致（在上述回归样本上未出现反例）。

另外，`UB-H&L` 在同样 50 组样本中可调度数量为 37，显著高于上述响应时间分析器；并已构造示例验证 `UB-H&L=True` 且 `SMC-no=False`，符合“通常更宽松”的经验结论。

### mceval 差异报告

#### 参考位置（Java 仓库）

1. 任务生成器：
   - `src/com/github/dumpram/mceval/taskgen/UUniFastDiscard.java`
2. AMC 实现：
   - `src/com/github/dumpram/mceval/rtimes/ResponseTimeAMCrtb.java`
   - `src/com/github/dumpram/mceval/rtimes/ResponseTimeAMCmax.java`
3. SMC/SMC-no：
   - `src/com/github/dumpram/mceval/rtimes/ResponseTimeSMC.java`
   - `src/com/github/dumpram/mceval/rtimes/ResponseTimeSMCno.java`
4. OPA：
   - `src/com/github/dumpram/mceval/assignments/PriorityAssignmentOPA.java`
5. 实验驱动：
   - `src/com/github/dumpram/mceval/evaluation/SchedulabilityTest.java`
   - `src/com/github/dumpram/mceval/results/SchedulabilityU.java`
   - `src/com/github/dumpram/mceval/results/SchedulabilityTestN38.java`

#### 主要差异

1. 任务生成过滤规则：
   - `mceval` 有 `UUniFast-Discard`、`delta` 容差、`hyperperiodlimit`、可选可行性过滤与黑名单。
   - 当前 Python 版实现核心 UUniFast 与参数生成，但未完整实现上述全部过滤策略。

2. 周期采样策略：
   - `mceval` 默认均匀整数采样并避免重复周期。
   - 当前 Python 版使用 log-uniform，便于模拟更广尺度周期分布。

3. 实验组织方式：
   - `mceval` 倾向在多个 Java `main` 入口中固化实验配置。
   - 当前 Python 版采用模块化 sweep API + 脚本入口，更便于二次开发。

4. 绘图生态：
   - `mceval` 使用 `matplotlib4j`。
   - 当前 Python 版直接使用 `matplotlib`，更直观地输出 PNG 与 CSV。

### 新增或修复的测试列表（阶段 E）

- `tests/test_trends_e.py`
  - `test_trend_chain_on_generated_tasksets`
  - `test_ub_hl_can_be_looser_than_smc_no_example`

### README 内容概述

最终 README 已补齐以下内容：
1. 项目背景与当前完成度（A-E）。
2. 环境安装步骤（conda）。
3. 测试运行方法。
4. 单示例运行方法。
5. 小规模实验运行方法。
6. 目录结构说明。
7. 当前能力总结。
8. 已知限制。
9. 与论文及 `mceval` 的差异说明。
10. 交接建议与下一步方向。

### 项目当前能力总结

1. 分析器：`UB-H&L / SMC / SMC-no / AMC-rtb / AMC-max`
2. 优先级策略：`DM / CrMPO / OPA`
3. 统一入口：`evaluate_taskset(tasks, method, priority_policy)`
4. 实验能力：
   - `util/cf/cp/n` 四类 sweep
   - CSV 输出
   - weighted schedulability 统计
   - 基础折线图输出
5. 工程能力：
   - conda 环境
   - pytest 回归
   - 示例与小实验脚本

### 已知限制与后续建议

1. 仍未完整复刻 `mceval` 的全部任务筛选逻辑（如 hyperperiod 限制与黑名单机制）。
2. 大规模实验尚未并行化，运行性能还有提升空间。
3. 当前图表模板偏基础，后续可增加论文风格统一绘图主题。
4. 若目标是严格逐项复现实验曲线，建议下一步固定随机种子策略并对齐 `mceval` 参数与过滤流程。

### 最终交付文件清单（阶段 E 相关）

- `README.md`（重写）
- `tests/test_trends_e.py`（新增）
- `output.md`（追加阶段 E 输出）

### 测试结果

- 命令：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q
```

- 结果：`28 passed in 0.98s`

---

## 阶段0~3 测试说明（最新）

以下命令用于验证当前已完成阶段（0,1,2,3）的交付。

### 1. 阶段0：环境与可复现安装

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pip install -e .
conda run -n amc-repro python -m pytest -q
```

验收点：
- `pip install -e .` 成功。
- 测试可直接执行。

### 2. 阶段1：OPA 与评估入口正确性

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q tests/test_opa.py tests/test_evaluation_api.py
```

验收点：
- OPA 语义相关用例全部通过。
- 非法 method/policy 组合能报明确错误。

### 3. 阶段2：CrMPO baseline 与方法矩阵

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q tests/test_evaluation_api.py
```

验收点：
- `method="crmpo_baseline"` 可独立调用。
- 仅允许 `priority_policy="crmpo"`。

### 4. 阶段3：生成器修复与统计校验

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q tests/test_generator.py
conda run -n amc-repro python scripts/validate_generator.py --mode fast --num-tasksets 200 --seed 2026
```

验收点：
- `deadline_equals_period=False` 时可生成 `D<T` 任务。
- 支持 `criticality_assignment=fixed_count/bernoulli` 切换。
- 支持 `fast/paper` 模式与 YAML 配置加载。
- 输出统计文件：
  - `reports/generator_validation/taskset_stats.csv`
  - `reports/generator_validation/task_stats.csv`
  - `reports/generator_validation/generator_validation_plots.png`
  - `reports/generator_validation/generator_validation_report.md`

### 5. 全量回归

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q
```

当前结果：`43 passed`。

### 阶段3补充完成说明（本轮）

1. 修正 `paper` 模式默认参数，确保默认支持 `D<T`：
   - `deadline_equals_period=False`
   - `criticality_assignment=bernoulli`
2. 补充生成器详细中文注释：
   - deadline 采样逻辑说明
   - HI 任务采样策略说明
   - YAML 配置解析约束说明
3. 增补阶段3单元测试：
   - `test_make_generation_config_modes`：验证 `fast/paper` 参数差异
   - `test_paper_mode_generates_arbitrary_deadlines`：验证 `paper` 模式可生成 `D<T`
4. 更新 `docs/task_generation.md`：
   - 明确两种模式默认差异
   - 新增模式调用示例

---

## 阶段4~5 测试说明（最新）

### 1. 阶段4：Fig.1~Fig.5 复现实验脚本

统一入口：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/reproduce_rtss11.py --figure fig1 --mode fast --num-tasksets 80 --seed 2026
conda run -n amc-repro python scripts/reproduce_rtss11.py --figure fig2 --mode fast --num-tasksets 80 --seed 2026
conda run -n amc-repro python scripts/reproduce_rtss11.py --figure fig3 --mode fast --num-tasksets 80 --seed 2026
conda run -n amc-repro python scripts/reproduce_rtss11.py --figure fig4 --mode fast --num-tasksets 80 --seed 2026
conda run -n amc-repro python scripts/reproduce_rtss11.py --figure fig5 --mode fast --num-tasksets 80 --seed 2026
```

验收点：
- 每张图都生成：
  - `outputs/figX/raw_results.csv`
  - `outputs/figX/aggregated_results.csv`
  - `outputs/figX/figX.png`
- Fig.2~Fig.4 额外包含 util 层聚合：
  - `outputs/figX/util_layer_aggregated.csv`
- Fig.5 的 raw 数据中应包含并可观察 `D<T` 任务（`has_deadline_less_than_period=True`）。
- 每次运行自动更新：`reports/figX_reproduction_notes.md`。

### 2. 阶段5：统计聚合与加权可调度率

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q tests/test_statistics.py
```

验收点：
- 覆盖 util 层聚合手算一致性。
- 覆盖跨 util 加权聚合手算一致性。
- 覆盖空样本边界。
- 覆盖全成功 / 全失败边界。

### 3. 当前全量回归

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q
```

当前结果：`43 passed`。
