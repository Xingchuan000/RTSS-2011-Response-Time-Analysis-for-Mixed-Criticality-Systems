# Runtime Simulation 说明（阶段运行时模拟）

本文档说明当前仓库中的运行时仿真模型、模式切换规则、scenario 用法，以及静态分析与运行时仿真的关系。

## 1. 目标与定位

`amc_py/runtime.py` 提供的是一个离散时间（tick-based）固定优先级仿真器，用于回答下面的问题：

1. 在给定优先级顺序下，任务在某个“实际执行时间场景”里会如何运行？
2. 是否触发 `LO -> HI` 模式切换？
3. 是否出现 deadline miss、是否丢弃了 LO jobs？

这个模块不替代静态分析，而是作为“静态结论的运行时观察补充”。

## 2. 运行时模型假设

当前版本采用以下关键假设：

1. 单处理器、固定优先级抢占式调度。
2. 时间离散化为整数 tick，每个 tick 是半开区间 `[t, t+1)`。
3. 任务参数来自 `Task(name, period, deadline, c_lo, c_hi, criticality)`。
4. job 在 `release_index * period` 时释放，绝对截止期是 `release_time + deadline`。
5. job 的实际执行时间由 `ExecutionScenario` 决定，而不是直接等于 `c_lo` 或 `c_hi`。

这些假设与当前 AMC 实验框架的目标保持一致：先保证语义可复现、可测试，再考虑进一步性能与模型扩展。

## 3. Mode Switch 规则（LO -> HI）

运行时默认从 `SystemMode.LO` 开始。切换规则如下：

1. 当系统还在 LO 模式时，某个 HI job 的累计执行时间满足 `executed_time > c_lo`，触发 `LO -> HI`。
2. 切换生效时刻定义为触发 tick 的结束边界，即 `switch_time = t + 1`。
3. `SimulationResult.mode_switch` 会记录触发任务、release 索引和触发时执行量。
4. 系统最多记录一次切换事件，进入 HI 后不再重复切换。

进入 HI 模式后的行为：

1. 可选丢弃活动中的 LO jobs（`RuntimeConfig.drop_lo_jobs_on_hi_switch=True` 为默认）。
2. 抑制未来 LO jobs 的释放（future LO release suppress）。
3. HI jobs 继续运行，直到完成或 miss。

## 4. Scenario 用法

`amc_py/runtime_scenarios.py` 提供统一场景接口 `ExecutionScenario` 和常用工厂函数：

1. `make_nominal_scenario()`
说明：所有任务都按 `c_lo` 执行，通常不会触发模式切换。

2. `make_single_hi_overrun_scenario(task_name, release_index=0, overrun_to="c_hi")`
说明：指定 HI 任务的某次 release 超限，常用于验证切换语义。

3. `make_all_hi_jobs_hi_budget_scenario(task_names=None)`
说明：所有或指定 HI 任务都按 `c_hi` 执行，常用于高压力路径。

4. `make_table_scenario(actual_costs, default_hi="c_lo", default_lo="c_lo")`
说明：通过 `(task_name, release_index) -> actual_cost` 显式给定某些 job 的执行时间，适合精细单测。

约束规则：

1. HI 任务实际执行时间必须满足 `1 <= actual_cost <= c_hi`。
2. LO 任务实际执行时间必须满足 `1 <= actual_cost <= c_lo`。
3. 非法值会在 scenario 层被立即拒绝，避免错误进入调度主循环。

## 5. API 分层

当前建议按下面两层 API 使用：

1. 底层 API：`simulate_ordered_taskset(ordered_tasks, scenario, config)`
适用：你已经有明确优先级顺序，想直接看运行时行为。

2. 集成 API（AMC runtime bridge）：`simulate_taskset_with_policy(tasks, method, priority_policy, scenario, config)`
适用：你希望复用 `method + priority_policy` 输入形式，由系统自动解析优先级并桥接到当前 AMC 运行时语义。
限制：该接口当前只支持 AMC family method（`amc_rtb`、`amc_max`）。

第 5 轮新增统一对照接口（同样属于 AMC runtime bridge）：

1. `compare_static_and_runtime(tasks, method, priority_policy, scenario, config)`
返回 `RuntimeComparisonResult`，同时包含：
静态分析结果、运行时仿真结果、最终顺序和方法元信息。
限制：该接口当前只支持 AMC family method（`amc_rtb`、`amc_max`）。

为什么要做这个限制（fail-fast）：

1. 当前 runtime 主循环实现的是 AMC 风格模式切换语义；
2. 若允许 `smc` / `smc_no` 等方法通过 bridge，会让调用方误以为“这些方法已有对应 runtime 语义实现”；
3. 因此在 bridge 入口直接拒绝非 AMC 方法，比“看起来跑通但语义不对”更安全。

## 6. 静态分析与 Runtime 的关系

静态分析与运行时仿真不是“二选一”，而是互补：

1. 静态分析（`evaluate_taskset`）给出理论可调度性判定和响应时间边界。
2. 运行时仿真在具体 scenario 下展示真实执行轨迹（切换、丢弃、miss）。
3. 两者通过 `resolve_ordering()` 共享同一优先级解析逻辑，避免排序口径分叉。
4. 当前 runtime bridge 仅对 AMC family method 开放；其它静态方法若要做 runtime 研究，需要单独定义其运行时解释，不应直接套用本 bridge。

常见理解方式：

1. 静态可调度 + nominal 场景通常应运行稳定；
2. 静态可调度并不等于“所有 overrun 场景都无 miss”；
3. 通过对比不同 scenario，可观察系统在压力下的行为弹性与降级策略。

## 7. 最小示例

可直接运行仓库示例脚本：

```bash
PYTHONPATH=. python scripts/run_runtime_example.py
```

脚本会打印两组场景（nominal / single HI overrun）的：

1. method / priority_policy
2. ordered task names
3. static result
4. runtime mode switch / misses / dropped jobs
5. 每个已释放 job 的执行状态摘要
