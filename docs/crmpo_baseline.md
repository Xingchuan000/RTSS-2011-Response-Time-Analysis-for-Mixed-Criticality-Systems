# CrMPO Baseline 说明（阶段 2.1）

## 1. 论文定义对齐

依据 Baruah 等（RTSS 2011）实验方法描述：

- CrMPO（Criticality Monotonic Priority Ordering）优先级：
  先按关键级（高在前），再按截止期（短在前）。
- 响应时间分析使用“每任务一个执行时间参数”：
  - HI 任务使用 `C(HI)`；
  - LO 任务使用 `C(LO)`。

这与 “所有任务一律按 `C(HI)`” 不同。

## 2. 本项目实现

新增独立方法：`method="crmpo_baseline"`

- 只能与 `priority_policy="crmpo"` 组合；
- 通过标准固定优先级响应时间方程计算：

`R_i = C_i + Σ ceil(R_i / T_j) * C_j`

其中 `C_i/C_j` 按上述“单一执行时间参数”规则选取。

## 3. 与 `sort_by_crmpo()` 的区别

- `sort_by_crmpo()` 仅负责排序；
- `crmpo_baseline` 是完整分析方法：排序 + 响应时间判定。

## 4. 与其他方法的边界

- 与 `smc / smc_no / amc_rtb / amc_max` 不同，
  `crmpo_baseline` 不做模式切换建模，仅作为论文对比基线。
