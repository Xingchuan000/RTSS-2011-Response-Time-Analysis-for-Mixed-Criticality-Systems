# Commit 1 接口梳理记录（RTSS2011 taskset factory）

## 1. generator.py 参数支持情况

`amc_py/generator.py` 中的 `generate_taskset(...)` 已支持以下参数：

- `num_tasks`
- `total_util`
- `min_period`
- `max_period`
- `time_scale`
- `cf`
- `cp`
- `deadline_mode`
- `criticality_assignment`

其中 `deadline_mode` 支持 `implicit / ratio_uniform / arbitrary_paper`，`criticality_assignment` 支持 `fixed_count / bernoulli`。

## 2. AMC-rtb 单任务集分析入口

单任务集分析可通过两条现有入口完成：

1. 低层函数：`amc_py/amc.py::amc_rtb_sched_test(ordered_tasks)`
- 返回 `SchedulabilityResult`，包含 `schedulable` 与 `response_times`。

2. 统一评估入口：`amc_py/experiments.py::evaluate_taskset(tasks, method="amc_rtb", priority_policy=...)`
- 会先进行优先级解析（可选 `dm/crmpo/opa`），再执行 AMC-rtb 分析。

补充说明：当前 `SchedulabilityResult.response_times` 仅返回每任务最终响应时间上界，不单独拆分 `R_LO` 与 `R*` 字段。

## 3. 事件驱动运行时入口与统计

事件驱动仿真主要入口为：

- `amc_py/event_runtime.py::simulate_ordered_taskset_event_driven(...)`
- `amc_py/event_runtime.py::EventRuntimeEngine`

结果对象为 `SimulationResult`，支持以下统计：

- `mode_changes`（通过 `mode_change_count()` 统计）
- `lo_cancellations`（通过 `lo_job_cancellation_count()` 统计）
- `deadline_misses`（`deadline_misses` 列表长度）
- `budget_overruns`（`budget_overrun_count()` 统计）

## 4. DQN 训练脚本当前 workload 支持

当前正式训练脚本 `scripts/train_dqn_amc.py` 仅支持 small 场景：

- `--scenario nominal`
- `--scenario stress`

尚未提供 `--workload rtss11` 参数（按任务书要求在后续 commit 扩展）。
