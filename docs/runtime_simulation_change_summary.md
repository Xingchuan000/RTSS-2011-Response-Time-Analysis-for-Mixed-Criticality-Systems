# Runtime Simulation 变更总览（第 1~6 轮）

本文档汇总本次“新增运行时模拟”工作的全部代码与测试改动，便于评审、交接与后续维护。

## 1. 目标与范围

本次改造目标是把 AMC 项目的静态分析能力扩展为“可观察运行时行为”的仿真能力，并与现有 `method + priority_policy` 入口打通。

交付内容覆盖：

1. 运行时数据模型与场景层（scenario）。
2. tick-based 固定优先级仿真器（含 LO->HI 切换语义）。
3. 与 `resolve_ordering()`、`evaluate_taskset()` 的集成桥接接口。
4. 对外 API 导出、集成测试、示例脚本、文档说明。

## 2. 分阶段变更摘要

### 第 1 轮：暴露公共排序接口（为 runtime 复用铺路）

核心变化：

1. 在 `experiments` 中暴露 `resolve_ordering(tasks, priority_policy, method)` 公共函数。
2. `evaluate_taskset()` 改为通过 `resolve_ordering()` 获取顺序，统一静态入口与后续 runtime 的排序口径。
3. 对 `dm / crmpo / opa` 路径补充直接测试。

影响文件：

1. `amc_py/experiments.py`
2. `tests/test_evaluation_api.py`

### 第 2 轮：运行时数据模型与 scenario 层

核心变化：

1. 新增 `amc_py/runtime_models.py`：
`SystemMode`、`RuntimeConfig`、`Job`、`ModeSwitchEvent`、`DeadlineMiss`、`ScheduleTick`、`SimulationResult`、`RuntimeComparisonResult`。
2. 新增 `amc_py/runtime_scenarios.py`：
`ExecutionScenario`、`_validate_actual_cost()` 以及 4 类场景工厂（`nominal`、`single_hi_overrun`、`all_hi_jobs_hi_budget`、`table`）。
3. 增加约束校验：HI job `actual_cost <= c_hi`，LO job `actual_cost <= c_lo`。

影响文件：

1. `amc_py/runtime_models.py`（新增）
2. `amc_py/runtime_scenarios.py`（新增）
3. `tests/test_runtime_scenarios.py`（新增）

### 第 3 轮：基础仿真器（nominal/LO-only 路径）

核心变化：

1. 新增 `amc_py/runtime.py` 的基础调度循环：
job 释放、优先级选择、执行推进、completion 记录、deadline miss 检查、trace 输出。
2. 提供辅助函数：
`compute_hyperperiod()`、`compute_default_end_time()`、`_build_job()`、`_release_jobs_at_time()`、`_select_highest_priority_ready_job()`、`_check_deadline_misses()`、`_should_switch_to_hi()`。
3. 完成 nominal 路径与 stop-at-first-miss 语义测试。

影响文件：

1. `amc_py/runtime.py`（新增）
2. `tests/test_runtime_simulator.py`（新增）

### 第 4 轮：补全 HI 模式切换语义

核心变化：

1. 在主循环接入 LO->HI 切换判定与事件记录（`ModeSwitchEvent`）。
2. 新增 `_drop_active_lo_jobs()` 并在切换后按配置丢弃活动 LO jobs。
3. 在 HI 模式下抑制未来 LO job 释放（future LO release suppress）。
4. 追加切换语义测试：single overrun、drop 行为、drop 开关、future suppress、单次切换约束。

影响文件：

1. `amc_py/runtime.py`
2. `tests/test_runtime_simulator.py`

### 第 5 轮：打通 method/policy 入口与静态-运行时对照

核心变化：

1. `runtime.py` 新增：
`simulate_taskset_with_policy()` 与 `compare_static_and_runtime()`。
2. 复用第 1 轮公共接口 `resolve_ordering()`，未在 runtime 中重复实现排序逻辑。
3. `compare_static_and_runtime()` 聚合输出静态结果、运行时结果与最终任务顺序。
4. `amc_py/__init__.py` 导出 runtime 公共 API。
5. 新增集成测试覆盖 `dm / crmpo / opa` 三条路径与 compare 接口。

影响文件：

1. `amc_py/runtime.py`
2. `amc_py/__init__.py`
3. `tests/test_runtime_integration.py`（新增）

### 第 6 轮：补示例与文档，完成回归

核心变化：

1. 新增示例脚本 `scripts/run_runtime_example.py`，演示：
`nominal scenario` 与 `single HI overrun scenario`。
2. 新增文档 `docs/runtime_simulation.md`，说明：
runtime 假设、mode switch 规则、scenario 用法、静态分析与 runtime 关系。
3. 完成 runtime 相关测试、全量测试与示例脚本实跑。

影响文件：

1. `scripts/run_runtime_example.py`（新增）
2. `docs/runtime_simulation.md`（新增）

## 3. 本次新增/修改文件清单（runtime 模拟相关）

### 核心代码

1. `amc_py/runtime_models.py`（新增）
2. `amc_py/runtime_scenarios.py`（新增）
3. `amc_py/runtime.py`（新增并持续迭代）
4. `amc_py/experiments.py`（新增 `resolve_ordering` 公共接口）
5. `amc_py/__init__.py`（导出 runtime 公共 API）

### 测试

1. `tests/test_runtime_scenarios.py`（新增）
2. `tests/test_runtime_simulator.py`（新增）
3. `tests/test_runtime_integration.py`（新增）
4. `tests/test_evaluation_api.py`（补 `resolve_ordering` 测试）

### 脚本与文档

1. `scripts/run_runtime_example.py`（新增）
2. `docs/runtime_simulation.md`（新增）

## 4. 对外可用 API（当前）

通过 `amc_py` 包可直接使用以下 runtime 能力：

1. 数据模型：
`RuntimeConfig`、`SystemMode`、`Job`、`SimulationResult`
2. 场景：
`ExecutionScenario`、`make_nominal_scenario()`、`make_single_hi_overrun_scenario()`、`make_all_hi_jobs_hi_budget_scenario()`、`make_table_scenario()`
3. 仿真与集成接口：
`simulate_ordered_taskset()`、`simulate_taskset_with_policy()`、`compare_static_and_runtime()`

## 5. 行为语义变化（重点）

1. 新增运行时执行语义，不仅输出“可调度/不可调度”，还能观察：
mode switch、drop、deadline miss、trace。
2. 模式切换规则固定为：HI job 执行量越过 `c_lo` 后在 `t+1` 生效。
3. 切换后可配置是否丢弃活动 LO jobs，默认丢弃。
4. 切换后抑制 future LO releases。
5. 静态分析与 runtime 通过 `resolve_ordering()` 共享统一排序口径。

## 6. 回归与验证结果（本次完成时）

在 `amc-repro` 环境中，以下验证通过：

1. `tests/test_runtime_integration.py`：`7 passed`
2. `tests/test_runtime_scenarios.py tests/test_runtime_simulator.py tests/test_runtime_integration.py`：`73 passed`
3. `pytest -q` 全量：`133 passed`
4. `scripts/run_runtime_example.py`：可正常运行并输出 nominal/overrun 两组结果摘要

## 7. 后续维护建议

1. 若继续扩展 runtime 功能（例如多核、更多降级策略），优先保持 `runtime_models.py` 的数据契约稳定。
2. 新增策略或排序规则时，优先接入 `resolve_ordering()`，避免静态/运行时口径分叉。
3. 对语义变化（尤其 mode switch、drop、release suppress）必须先补测试再改逻辑，避免回归。

