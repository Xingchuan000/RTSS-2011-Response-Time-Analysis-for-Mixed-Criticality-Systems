# Pre-DQN Runtime Interface 说明

## 1. 为什么不直接把 DQN 塞进 `simulate_ordered_taskset_event_driven`

当前 `simulate_ordered_taskset_event_driven(...)` 的职责是稳定复现 AMC+/AMC 事件语义。为了避免在接入学习器时破坏已有运行时语义，本项目将 DQN 交互需求拆成独立层：

1. 运行时核心层：`amc_py/event_runtime.py`（含 `EventRuntimeEngine`）；
2. RL 支撑层：`monitor / observation / actions / safety`；
3. 交互封装层：`runtime_wrapper` 与 `AmcBudgetEnv`。

这样做的目标是：在不侵入主循环调度语义的情况下，先把可观测、可动作、可验证的接口打稳，再接入 DQN。

## 2. `RuntimeMonitor` 统计项

`amc_py/rl/monitor.py` 的 `RuntimeMonitor` 负责在运行过程中记录：

1. `recent_execution`：每个任务最近一次 job 的执行量（完成、LO overrun、HI overrun 时更新）；
2. `job_start_count`：job 首次启动计数（同一 job 被抢占恢复不重复计数）；
3. `lo_overrun_count` / `hi_overrun_count`：预算超限统计；
4. `reward_since_last_agent`：两次 agent 激活之间累计奖励。

奖励定义固定为：

1. Job start：`+0.1`；
2. LO-job budget overrun：`-1.0`；
3. HI-job budget overrun：`-2.0`；
4. 其他事件：`0.0`。

通过 `consume_reward()` 读取并清零累计奖励。

## 3. observation 字段顺序与归一化

观测由 `amc_py/rl/observation.py` 的 `build_observation(...)` 构建，返回 `AgentObservation`：

1. `time`：当前时间；
2. `state_vector`：按 `ordered_tasks` 顺序展开的 `[(B_i, c_i), ...]`；
3. `raw_budgets`：任务名到当前预算映射；
4. `raw_recent_costs`：任务名到最近执行量映射。

关键规则：

1. 顺序必须严格跟随 `ordered_tasks`；
2. 每任务输出两个分量：`normalized_budget`、`normalized_recent_cost`；
3. 若任务尚无 recent execution，固定用 `0`；
4. 归一化函数：

```python
def _normalize(value: int, lo: int, hi: int) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))
```

## 4. action space 生成规则

离散动作由 `amc_py/rl/actions.py` 中 `build_budget_action_space(...)` 生成：

1. 每个动作选 1 个任务做 `+10%`；
2. 再从其余任务中选 2 个任务各做 `-5%`；
3. 总动作数：`n(n-1)(n-2)/2`；
4. 动作顺序是确定性的；
5. `action_id` 从 `0` 连续编号。

## 5. budget action 整数化规则

`apply_budget_action_candidate(...)` 将动作转为候选更新（不原地改预算）：

1. increase 使用 `ceil(old * 1.10)`；
2. decrease 使用 `floor(old * 0.95)`；
3. 任何预算下限为 `1`；
4. HI 任务预算不超过 `c_hi`；
5. LO 任务预算不超过 `deadline`。

## 6. safety checker 公式与保守性

安全检查实现于 `amc_py/rl/safety.py` 的 `RuntimeBudgetSafetyChecker`。

输入是完整候选预算向量；若缺失任务预算或预算小于等于 0，直接拒绝。

检查项：

1. HI task 的 LO-mode 保守检查：

```text
B_i + sum_{j in hp(i)} ceil(R_LO_i / T_j) * B_j <= R_LO_i
```

2. HI task 的 mode-switch 保守检查：

```text
C_i(HI)
+ sum_{j in hpL(i)} ceil(R_LO_i / T_j) * B_j
+ sum_{j in hpH(i)} ceil(D_i / T_j) * C_j(HI)
<= D_i
```

3. LO task 可选检查（`check_lo_tasks=True`）：

```text
B_i + sum_{j in hp(i)} ceil(D_i / T_j) * B_j <= D_i
```

这是保守过滤器，目标是先保证“不会放行明显不安全预算”，不追求最大放行率。

## 7. `AmcBudgetEnv.reset/step` 语义

`amc_py/rl/env.py` 提供不依赖 gymnasium 的环境接口。

1. `reset(seed)`：
- 初始化 `EventRuntimeEngine`、`RuntimeMonitor`、`BudgetState`；
- 返回初始 `AgentObservation`。

2. `step(action_id)`：
- `action_id=None` 表示 NoOp；
- 非空动作时先生成 candidate，再过 safety checker；
- 通过才应用预算更新；
- 运行时推进到下一次 agent 激活时刻；
- 返回 `AgentStepResult(observation, reward, done, info)`。

3. `done=True`：当前时间达到 `runtime_config.end_time`。

`info` 至少包含：

1. `time`；
2. `accepted`；
3. `reject_reason`；
4. `mode_changes`；
5. `lo_cancellations`；
6. `deadline_misses`。

## 8. NoOp / Random baseline 的用途

1. `NoOpBudgetAgent`：验证“引入 agent 接口但不动作”时，不改变 AMC+ baseline 关键统计；
2. `RandomBudgetAgent`：验证 action/safety/reward/日志管线可闭环运行，并可用固定 seed 复现轨迹。

## 9. 后续接入 DQN 的入口

后续 DQN 不应直接改 `event_runtime.py` 主循环，而是通过环境接口：

```python
env = AmcBudgetEnv(...)
obs = env.reset(seed=...)
step_result = env.step(action_id)
```

推荐训练路径：仅依赖 `obs.state_vector`、`reward`、`done`、`info`，并保留 safety checker 作为硬约束。
