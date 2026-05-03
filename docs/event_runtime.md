# 事件驱动 Runtime（event_runtime）

## 1. tick 与 event 两套 runtime 的定位

- `amc_py/runtime.py`：tick-based runtime，按整数 tick 推进，作为历史基线。
- `amc_py/event_runtime.py`：event-driven runtime，按事件队列推进，用于 DQN-AMC 复现前置环境。

两者并存，`runtime.py` 不被事件驱动实现替换。

## 2. 共享数据模型

tick/runtime 与 event/runtime 共享以下结构：

- `RuntimeConfig`
- `RuntimeSemantics`
- `SimulationResult`
- `BudgetState`
- `BudgetUpdate`

因此两种 runtime 的结果统计（`mode_change_count`、`lo_job_cancellation_count` 等）可直接对比。

## 3. 事件驱动 runtime 的事件类型

当前事件类型定义在 `amc_py/event_models.py`：

- `BUDGET_UPDATE`
- `JOB_COMPLETION`
- `BUDGET_OVERRUN`
- `DEADLINE_CHECK`
- `JOB_ARRIVAL`

## 4. AMC+ 与 AMC 的 overrun 语义差异

- `AMC_PLUS`：
  - LO job overrun：只取消该 LO job，系统保持 LO 模式。
  - HI job overrun：触发 `LO -> HI`，并按配置丢弃活动 LO jobs。
- `AMC`：
  - 任意 job overrun：触发 `LO -> HI`。

## 5. 动态预算更新的生效时机

- `BudgetUpdate` 会被转换为 `BUDGET_UPDATE` 事件。
- 在同一时刻中，`BUDGET_UPDATE` 先于 arrival/overrun/deadline 检查处理。
- 更新后会重排当前运行 job 的 completion/overrun 事件，并使旧 token 失效。

## 6. 事件优先级规则

同一时间点的事件处理顺序为：

1. `BUDGET_UPDATE`
2. `JOB_COMPLETION`
3. `BUDGET_OVERRUN`
4. `DEADLINE_CHECK`
5. `JOB_ARRIVAL`

该顺序保证：

- completion 与 deadline 同时发生时，先完成后检查，避免误判 miss；
- budget update 在同刻内最先生效；
- arrival 最后处理，减少同刻重排噪声。

## 7. 最小使用示例

```python
from amc_py.event_runtime import simulate_taskset_with_policy_event_driven
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario

result = simulate_taskset_with_policy_event_driven(
    tasks=tasks,
    method="amc_rtb",
    priority_policy="dm",
    scenario=make_nominal_scenario(),
    config=RuntimeConfig(end_time=100, semantics=RuntimeSemantics.AMC_PLUS),
)
```

## 8. 当前未实现内容（明确边界）

目前仍未实现：

1. DQN agent
2. AMC-rtb budget safety checker
3. agent 作为最低优先级 HI task 的运行时建模
4. automotive workload generator
