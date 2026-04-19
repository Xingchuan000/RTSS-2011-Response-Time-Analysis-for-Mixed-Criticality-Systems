# 方法与优先级策略兼容矩阵（阶段 2.2）

## 1. 矩阵

| Method | DM | CrMPO | OPA |
|---|---:|---:|---:|
| ub_hl | yes | yes | no |
| smc | yes | yes | yes |
| smc_no | yes | yes | yes |
| amc_rtb | yes | yes | yes |
| amc_max | yes | yes | yes |
| crmpo_baseline | no | yes | no |

## 2. 设计说明

- `crmpo_baseline` 在论文中就是 CrMPO 排序下的基线，因此不允许 DM/OPA。
- `ub_hl` 在本项目默认用于上界判定，不提供 OPA 组合。
- 其余响应时间分析方法支持 DM/CrMPO/OPA 三类策略。

## 3. 非法组合行为

对非法组合，`evaluate_taskset(...)` 将抛出 `ValueError`，
错误信息明确指出：

- 当前 method；
- 被拒绝的 priority_policy；
- 该 method 允许的 policy 列表。
