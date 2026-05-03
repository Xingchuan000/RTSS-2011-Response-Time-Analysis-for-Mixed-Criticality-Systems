# Runtime Simulation 变更摘要（阶段 1-10）

本文档汇总 AMC+/动态预算运行时实现的关键增量。

## 阶段 1：多事件结果模型

1. `SimulationResult` 从单一 `mode_switch` 扩展为：
- `mode_switches`
- `job_cancellations`
2. 新增 `JobCancellationEvent`。
3. 保留兼容属性 `mode_switch`（返回首个切换事件）。

## 阶段 2：BudgetState

1. 新增 `amc_py/budget_runtime.py`。
2. 引入 `BudgetState`，预算与 `Task.c_lo` 解耦。
3. 不再通过修改 `Task.c_lo` 表示运行时预算。

## 阶段 3：LO overrun 场景放开

1. scenario 层允许 LO `actual_cost > c_lo`。
2. 新增 `make_single_lo_overrun_scenario(...)`。

## 阶段 4：runtime 接入 BudgetState

1. `simulate_ordered_taskset(..., budget_state=None)`。
2. HI/LO overrun 判断改为基于 runtime budget。
3. 新增 LO 局部取消路径与 `JobCancellationEvent` 记录。

## 阶段 5：AMC 与 AMC+ 语义开关

1. 新增 `RuntimeSemantics`：
- `AMC`
- `AMC_PLUS`
2. `RuntimeConfig.semantics` 默认 `AMC_PLUS`。
3. AMC 与 AMC+ 在 LO overrun 行为上可配置切换。

## 阶段 6：HI 空闲恢复 LO

1. 新增 `ModeRecoveryEvent`。
2. HI mode 且无活动 job 时恢复 LO mode。
3. HI 期间抑制 LO release 时推进 release index，避免恢复后补发历史 LO jobs。

## 阶段 7：多次事件累计

1. 支持多次 mode switch / mode recovery / LO cancellation。
2. 增加累计统计接口：
- `mode_change_count()`
- `mode_recovery_count()`
- `lo_job_cancellation_count()`

## 阶段 8：预算更新时间表（非 DQN）

1. 新增 `BudgetUpdate`。
2. 新增 `BudgetState.apply_updates(...)`。
3. `simulate_ordered_taskset(..., budget_updates=None)` 支持按时刻应用更新。
4. 新增 `BudgetUpdateEvent` 与 `SimulationResult.budget_update_events`。

## 阶段 9：AMC+ baseline 示例脚本

1. 新增 `scripts/run_amc_plus_runtime_example.py`。
2. 脚本展示：
- AMC_PLUS + LO overrun
- AMC + LO overrun
- AMC_PLUS + dynamic budget（更新前/后）

## 阶段 10：文档与回归

1. 更新 `docs/runtime_simulation.md`（语义与预算机制说明）。
2. 更新 `README.md`（新增脚本运行方式）。
3. 增加/维护 runtime 相关测试，确保回归通过。

