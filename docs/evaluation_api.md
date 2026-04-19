# 统一评估入口说明（阶段 1.3）

## 1. 入口函数

- `amc_py.experiments.evaluate_taskset(tasks, method, priority_policy)`

统一返回：`SchedulabilityResult`

## 2. 输入校验

为避免实验脚本静默跑偏，入口新增了统一参数校验：

- `tasks` 不能为空；
- 任务名必须唯一（用于稳定优先级映射）；
- `method` 必须是已注册方法；
- `method` 与 `priority_policy` 必须满足兼容矩阵。

非法组合会抛出明确 `ValueError`，并附带可选策略列表。

## 3. 一致性约定

- 返回对象的 `method` 字段始终标识分析方法；
- `details` 中包含 `method` 与 `priority_policy` 元信息；
- OPA 分配失败时不会崩溃，而是返回 `schedulable=False` 的统一结果。

## 4. 典型调用

```python
from amc_py.experiments import evaluate_taskset

result = evaluate_taskset(tasks, method="amc_rtb", priority_policy="opa")
print(result.schedulable, result.details)
```
