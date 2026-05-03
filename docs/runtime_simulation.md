# Runtime Simulation 说明（AMC / AMC+）

本文档描述当前运行时仿真器的语义、配置与动态预算机制。

## 1. 仿真器定位

`amc_py/runtime.py` 是离散时间（tick-based）单处理器固定优先级抢占式仿真器，用于在给定任务顺序和执行场景下观测：

1. 模式切换（`LO -> HI`）与模式恢复（`HI -> LO`）。
2. LO job 局部取消事件（AMC+）。
3. deadline miss 与调度轨迹。

## 2. AMC vs AMC+ runtime semantics

通过 `RuntimeConfig(semantics=...)` 选择语义：

1. `RuntimeSemantics.AMC_PLUS`
- LO job 超过 LO-mode runtime budget：只取消该 LO job。
- HI job 超过 LO-mode runtime budget：切换到 HI mode。
- HI mode 期间抑制 LO release；当系统空闲时恢复到 LO mode。

2. `RuntimeSemantics.AMC`
- 任意 job（含 LO/HI）超过 LO-mode runtime budget：切换到 HI mode。
- 切换时可按 `drop_lo_jobs_on_hi_switch` 配置丢弃活动 LO jobs。

## 3. 动态预算机制

运行时预算与任务设计参数分离：

1. 设计时参数：`Task.c_lo` / `Task.c_hi`。
2. 运行时预算：`BudgetState`（`amc_py/budget_runtime.py`）。

关键点：

1. 不修改 `Task.c_lo` 表示动态预算。
2. 不传 `budget_state` 时，仿真器默认用 `Task.c_lo` 初始化预算向量。
3. 可通过 `BudgetUpdate(time, updates)` 在指定时刻更新预算。
4. 每个 tick 在 release 前应用该 tick 的 budget update。

## 4. 事件与统计

`SimulationResult` 中提供：

1. `mode_switches` / `mode_change_count()`
2. `mode_recoveries` / `mode_recovery_count()`
3. `job_cancellations` / `lo_job_cancellation_count()`
4. `budget_update_events`
5. `deadline_misses`

兼容接口 `mode_switch` 保留为“第一条 mode switch 事件”。

## 5. release 抑制与恢复规则

HI mode 下，LO release 会被抑制；同时其 `release_index` 仍推进。这保证：

1. 不会在恢复 LO mode 后补发 HI 期间错过的历史 LO jobs。
2. 恢复后只从新的周期点继续释放。

## 6. 场景层（ExecutionScenario）

`amc_py/runtime_scenarios.py` 提供常用场景工厂：

1. `make_nominal_scenario()`
2. `make_single_hi_overrun_scenario(...)`
3. `make_single_lo_overrun_scenario(...)`
4. `make_all_hi_jobs_hi_budget_scenario(...)`
5. `make_table_scenario(...)`

约束：

1. HI job: `actual_cost <= c_hi`
2. LO job: 允许 `actual_cost > c_lo`，是否继续执行由 runtime budget 机制决定。

## 7. 最小示例

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/run_amc_plus_runtime_example.py
```

脚本输出字段：

`case, semantics, mode_changes, lo_cancellations, recoveries, deadline_misses, final_mode`

