# 统计与加权可调度率说明（阶段5）

## 1. 两层聚合目标

阶段5将统计流程拆成两层：

1. util 层内聚合（`aggregate_by_util`）
2. 跨 util 聚合（`aggregate_weighted_schedulability`）

这样可以清晰表达“先算每个 util，再按 util 权重做总聚合”。

## 2. 第一层：util 层内聚合

对固定分组（例如某个 method + sweep_value）内的某个 util 层：

- `util_sum = Σ U_i`
- `weighted_success_sum = Σ(U_i * I_i)`
- `schedulable_ratio = Σ I_i / N`
- `weighted_schedulability_at_util = weighted_success_sum / util_sum`

其中：

- `U_i`：第 i 个样本的实际 LO 利用率
- `I_i`：可调度指示变量（可调度=1，不可调度=0）

## 3. 第二层：跨 util 加权聚合

对同一外层 sweep 点（例如某个 CF）跨 util 汇总：

- `total_util_sum = Σ util_sum_u`
- `total_weighted_success = Σ weighted_success_sum_u`
- `weighted_schedulability = total_weighted_success / total_util_sum`
- `schedulable_ratio = total_success / total_tasksets`

## 4. 代码入口

- `amc_py/aggregation.py`
  - `aggregate_by_util(...)`
  - `aggregate_weighted_schedulability(...)`

兼容接口：

- `amc_py.experiments.compute_weighted_schedulability(...)`

## 5. 手算示例

样本（同一组内）：

- util=0.5：`(U=0.5,I=1)`, `(U=0.4,I=0)`
- util=0.7：`(U=0.7,I=1)`, `(U=0.8,I=1)`

则：

- util=0.5：`weighted_sched=0.5/(0.5+0.4)=0.555...`
- util=0.7：`weighted_sched=1.0`

跨 util 后：

- `weighted_sched=(0.5+1.5)/(0.9+1.5)=2.0/2.4=0.8333...`

## 6. 测试覆盖

`tests/test_statistics.py` 覆盖：

- util 层聚合手算一致性
- 跨 util 聚合手算一致性
- 空样本
- 全成功/全失败边界
