# AMC Python 复现项目（阶段 A-E）

## 1. 项目背景

本项目用于复现混合关键级调度分析（Adaptive Mixed-Criticality, AMC）相关方法，
参考论文与 `mceval` Java 实现，当前已完成到阶段 E：

- 基础模型与工程骨架
- LO/HI 模式响应时间分析
- SMC / SMC-no
- AMC-rtb / AMC-max
- OPA(Audsley) 优先级分配
- 统一评估接口
- 随机任务集生成、批量实验、CSV 导出、绘图
- 趋势回归测试与对照文档

## 2. 环境安装

推荐 Python 版本：`3.11`

### 2.1 使用 conda 创建环境

```bash
cd /Users/x1ngchuan/Documents/AMC
conda env create -f environment.yml
conda activate amc-repro
```

### 2.2 可选：检查 Python 包是否可导入

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -c "import amc_py; print('ok')"
```

### 2.3 使用 pip 安装（可编辑模式）

```bash
cd /Users/x1ngchuan/Documents/AMC
python -m pip install -e .[dev]
python -m pytest -q
```

若你只想安装运行依赖，可使用：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 3. 如何运行测试

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q
```

说明：
- 测试覆盖基础模型、响应时间分析、优先级分配、AMC 方法、实验框架与趋势回归。
- 当前版本测试应全部通过。

## 3.1 快速开始

只看本节即可完成一次完整验证：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q
conda run -n amc-repro python scripts/run_small_experiment.py
```

运行完成后可在 `outputs/` 查看 CSV 与图表文件。

## 4. 如何运行单示例

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/run_single_example.py
```

输出内容包括：
- 随机生成的任务参数
- `ub_hl / smc / smc_no / amc_rtb / amc_max` 在同一任务集上的判定与响应时间

## 4.1 如何运行 AMC+ Runtime 示例

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/run_amc_plus_runtime_example.py
```

输出会包含：
- `AMC_PLUS` 与 `AMC` 在 LO overrun 场景下的行为差异
- `mode_changes / lo_cancellations / recoveries / deadline_misses / final_mode`
- dynamic budget 更新前后的结果差异

## 4.2 如何运行事件驱动 Runtime 示例

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/run_event_runtime_example.py
```

输出会包含事件驱动 runtime 的四个对照案例：
- `case1_lo_overrun_event`（AMC+）
- `case2_lo_overrun_event`（AMC）
- `case3_dynamic_budget_no_update_event`（AMC+）
- `case3_dynamic_budget_updated_event`（AMC+）

更多语义说明见 `docs/event_runtime.md`。

## 5. 如何运行小规模实验

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/run_small_experiment.py
```

默认会在 `outputs/` 目录生成：
- `small_util_sweep.csv`
- `small_util_sweep_weighted.csv`
- `small_util_schedulable_percentage.png`
- `small_util_weighted_schedulability.png`

## 6. 目录说明

```text
/Users/x1ngchuan/Documents/AMC
├── amc_py/
│   ├── models.py               # 任务模型与结果数据结构
│   ├── priorities.py           # DM/CrMPO/OPA
│   ├── rta.py                  # LO/HI 固定点响应时间分析
│   ├── bounds.py               # UB-H&L
│   ├── smc.py                  # SMC/SMC-no
│   ├── amc.py                  # AMC-rtb/AMC-max
│   ├── generator.py            # UUniFast 与任务集生成
│   └── experiments.py          # 统一评估、sweep、统计、绘图
├── scripts/
│   ├── run_single_example.py   # 单任务集示例
│   ├── run_amc_plus_runtime_example.py # AMC+/AMC runtime 对比示例
│   ├── run_event_runtime_example.py # 事件驱动 runtime 对比示例
│   ├── run_small_experiment.py # 小规模 sweep + CSV + 图
│   └── run_experiments.py      # 兼容入口（调用小实验）
├── docs/
│   ├── event_runtime.md        # 事件驱动 runtime 使用与语义说明
│   └── ...
├── tests/                      # pytest 测试集合
├── outputs/                    # 实验产物目录
├── environment.yml
├── pyproject.toml
├── output.md                   # 分阶段交付总文档
└── README.md
```

## 7. 当前能力总结

当前版本已经具备以下可交付能力：

1. 可调度性分析方法：
   - `UB-H&L`
   - `SMC`
   - `SMC-no`
   - `AMC-rtb`
   - `AMC-max`

2. 优先级策略：
   - `dm`
   - `crmpo`
   - `opa`

3. 统一评估接口：
   - `evaluate_taskset(tasks, method, priority_policy)`

4. 实验能力：
   - `utilization/cf/cp/n` 四类 sweep
   - CSV 输出
   - `weighted schedulability` 统计
   - 基础科研曲线绘制

## 8. 当前限制

1. 当前默认双关键级模型（`LO/HI`），未扩展到多关键级。
2. 任务参数采用整数离散化，和论文中连续参数设定存在数值误差。
3. 大规模实验尚未做并行化或性能优化。
4. 当前图表样式偏基础，尚未做论文排版级统一主题。

## 9. 与论文及 mceval 的差异说明

### 9.1 与论文设定的差异

1. 目前默认 `D=T`（隐式截止期），未完整覆盖论文中的全部参数组合。
2. 实验规模以本地快速复现为主，尚未完全复制原文全部批量规模。

### 9.2 与 mceval(Java) 的差异

1. 任务生成器：
   - `mceval` 使用 `UUniFastDiscard`，并加入 `delta`、`hyperperiodlimit`、黑名单等过滤条件。
   - 当前 Python 版实现了 UUniFast 与核心参数，但尚未完全复刻这些额外过滤逻辑。

2. 实验驱动：
   - `mceval` 在 `results/` 与 `evaluation/` 中组织大量预设实验入口。
   - 当前 Python 版提供通用 sweep API 与小规模脚本，更偏模块化与可扩展。

3. 绘图生态：
   - `mceval` 基于 `matplotlib4j` 与 Java 流程。
   - 当前 Python 版直接使用 `matplotlib`，更便于后续数据分析二次开发。

## 10. 交接建议

1. 若要做论文级复现图，建议优先扩展：
   - 更多参数 sweep 组合
   - 多方法同图自动汇总
   - 大规模并行实验

2. 若要做行为逐行对齐，建议补齐：
   - `UUniFastDiscard` 的完整过滤规则
   - 与 `mceval` 相同的随机种子/任务筛选流程

## 11. Pre-DQN runtime interface

## 12. Constraint-Guided Pair 诊断使用说明

`scripts/generate_learnable_tasksets.py` 已新增 constraint-guided pair diagnostic：

- `--enable-constraint-guided-pair-diagnostic`：开启 constraint-guided pair 诊断。
- `--constraint-guided-pair-min-valid-count`：`fast_valid_constraint_guided_pair_count_mean` 的最小通过阈值，默认 `1.0`。
- `--constraint-guided-pair-top-k-risk`：每步用于构造 increase 候选的 top-k risk 数量，默认 `3`。
- `--constraint-guided-pair-top-k-decrease`：每个 increase 目标选取的约束贡献型 decrease 数量，默认 `4`。
- `--constraint-guided-pair-prefer-lo`：选择 decrease 目标时对 LO 任务加权优先。
- `--learnable-selection-target`：筛选目标，支持 `single|ranked_pair|constraint_guided_pair`。

当使用 `--learnable-selection-target constraint_guided_pair` 时，候选 taskset 的通过与拒绝主要由以下字段决定：

- `recommended_for_constraint_guided_pair_dqn`
- `constraint_guided_pair_not_recommended_reason`
- `fast_valid_constraint_guided_pair_count_mean`
- `fast_constraint_guided_pair_reject_hi_lo_mode_violation_mean`

示例命令（与计划文档一致）：

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/generate_learnable_tasksets.py \
  --automotive-num-runnables 150 \
  --num-tasksets 5 \
  --candidate-seed-start 0 \
  --learnable-max-attempts 50 \
  --learnable-generation-strategy two_stage_from_paper_exact \
  --learnable-selection-target constraint_guided_pair \
  --learnable-target-budget-util-min 0.35 \
  --learnable-target-budget-util-max 0.55 \
  --learnable-hi-budget-rho-min 0.35 \
  --learnable-hi-budget-rho-max 0.55 \
  --learnable-lo-budget-rho-min 0.25 \
  --learnable-lo-budget-rho-max 0.50 \
  --learnable-min-static-increase-reserve 4 \
  --learnable-min-static-hi-increase-reserve 2 \
  --learnable-min-static-decrease-reserve 4 \
  --learnable-fast-end-time 500000 \
  --learnable-fast-eval-seeds 3 \
  --learnable-fast-event-min 0 \
  --learnable-fast-event-max 120 \
  --learnable-fast-min-valid-increase 1 \
  --learnable-fast-min-valid-decrease 3 \
  --learnable-fast-min-balance 0.05 \
  --enable-constraint-guided-pair-diagnostic \
  --constraint-guided-pair-min-valid-count 1 \
  --constraint-guided-pair-top-k-risk 3 \
  --constraint-guided-pair-top-k-decrease 4 \
  --constraint-guided-pair-prefer-lo \
  --reward-mode interval_v1 \
  --action-space single \
  --budget-increase-ratio 0.01 \
  --budget-decrease-ratio 0.0125 \
  --include-explicit-noop \
  --budget-floor-ratio 0.9 \
  --observation-mode v11_full_10d \
  --ema-alpha 0.2 \
  --overrun-ema-alpha 0.1 \
  --history-k 8 \
  --event-window 10 \
  --max-cost-weight 0.7 \
  --risk-max-scale 3.0 \
  --include-safety-margin \
  --output-manifest outputs/tasksets/paper_learnable_constraint_guided_pair_r150_manifest.csv \
  --output-rejections outputs/tasksets/paper_learnable_constraint_guided_pair_r150_rejections.csv
```

在接入 DQN 之前，当前仓库已经提供可直接用于训练循环的运行时环境封装：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q tests/test_rl_env.py
conda run -n amc-repro python scripts/run_pre_dqn_runtime_baselines.py --end-time 100 --seed 0
```

相关文档：`docs/pre_dqn_runtime_interface.md`。

### 12.1 constraint_guided_transfer 动作空间（Step1~Step7）使用说明

本次实现把训练/评估/scan/generator 的 constraint-guided 语义统一为 `constraint_guided_transfer`（`constraint_guided_pair` 仅保留 alias）。

- 新增模块：`amc_py/rl/constraint_guided_pair.py`
  - `extract_task_rank_features_from_v11(...)`
  - `build_constraint_guided_increase_candidates(...)`
  - `select_constraint_guided_decrease_targets(...)`
  - `apply_single_increase_candidate(...)`
  - `apply_pair_candidate(...)`
- 动作空间扩展：`amc_py/rl/actions.py::build_budget_action_space(...)`
  - 新增正式 `action_space="constraint_guided_transfer"`，`constraint_guided_pair` 自动路由到该语义
  - 固定槽位数 = `top_k_risk`（不是 `top_k_risk * top_k_decrease`）
  - 动作数 = `1(noop) + top_k_risk`，默认是 `4`
- 环境配置扩展：`amc_py/rl/env.py::AmcBudgetEnv`
  - `action_space` 新增 `constraint_guided_pair`
  - 新增字段：
    - `constraint_guided_pair_top_k_risk`
    - `constraint_guided_pair_top_k_decrease`
    - `constraint_guided_pair_prefer_lo`
    - `constraint_guided_pair_include_hi_risk_boost`
    - `constraint_guided_pair_allow_increase_only_when_safe`
- 训练/评估 CLI 扩展：
  - `scripts/train_dqn_amc.py --action-space` 支持 `constraint_guided_transfer` 与 `constraint_guided_pair`
  - `scripts/evaluate_dqn_amc.py --action-space` 支持 `constraint_guided_transfer` 与 `constraint_guided_pair`
  - `--constraint-guided-pair-allow-increase-only-when-safe` 默认改为 `False`

注意事项：

- `constraint_guided_pair` 槽位动作是“动态解析动作”，不能直接调用 `apply_budget_action_candidate(...)` 执行；
- 该函数在收到 `is_constraint_guided_pair=True` 动作时会抛出 `ValueError`，这是计划要求的保护逻辑。

最小训练命令示例（仅示例动作空间参数）：

```bash
conda run -n amc-repro python scripts/train_dqn_amc.py \
  --workload small_stress \
  --episodes 2 \
  --validation-seeds 0 \
  --action-space constraint_guided_transfer \
  --include-explicit-noop
```

最小评估命令示例（需与训练动作空间保持一致）：

```bash
conda run -n amc-repro python scripts/evaluate_dqn_amc.py \
  --workload small_stress \
  --dqn-model outputs/dqn_amc/model_best.pt \
  --seeds 0 \
  --action-space constraint_guided_transfer \
  --include-explicit-noop
```

Step6~Step10 额外参数说明（训练/评估同名参数）：

- `--constraint-guided-pair-top-k-risk`：固定 risk 槽位数，默认 `3`
- `--constraint-guided-pair-top-k-decrease`：每个 risk 槽位下的 decrease 槽位数，默认 `5`
- `--constraint-guided-pair-prefer-lo / --no-constraint-guided-pair-prefer-lo`：decrease 目标是否偏向 LO，默认关闭
- `--constraint-guided-pair-include-hi-risk-boost`：是否启用“全任务 top-k + HI top-k”并集候选，默认关闭
- `--constraint-guided-pair-allow-increase-only-when-safe / --no-constraint-guided-pair-allow-increase-only-when-safe`：single increase 安全时是否允许直接执行 increase-only，默认开启

动作维度计算公式：

- `action_space_size = top_k_risk * top_k_decrease + (include_explicit_noop ? 1 : 0)`
- 例如默认参数 + 显式 noop：`3 * 5 + 1 = 16`

### 11.x Residual Correction（两阶段实现）

本次实现新增了文档 `AMCRTB_residual_correction_两阶段实现计划.md` 对应的两阶段能力：

- Step 1：保留原 action space，在执行前增加 residual safety fallback。
- Step 2：新增 `residual_ranked` action space，让 DQN 学习 AMCRTB 基线上的小幅修正。

#### 1) Step 1：Residual Safety Fallback

训练 CLI 新增参数：

- `--enable-residual-safety-fallback / --no-enable-residual-safety-fallback`
- `--residual-guard-hi-pressure-delta-limit`（默认 `0.03`）
- `--residual-guard-hi-pressure-abs-limit`（默认 `0.30`）
- `--residual-guard-reject-decrease-pressure-threshold`（默认 `0.05`）
- `--residual-guard-use-hi-pressure-max / --no-residual-guard-use-hi-pressure-max`（默认关闭，使用 mean）

开启后，动作执行会先做 residual guard：

- 若 HI 风险增量超阈值，拒绝动作；
- 若 HI 风险绝对值超阈值，拒绝动作；
- 若 decrease 命中高 pressure 任务，拒绝动作；
- 被拒绝动作按 fallback=NoOp 处理，并在 `train_log.csv` 中记录 `residual_guard_*` 字段。

Step 1 示例（`single`）：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/train_dqn_amc.py \
  --workload mc_fairgen \
  --action-space single \
  --budget-increase-ratio 0.025 \
  --budget-decrease-ratio 0.015 \
  --include-explicit-noop \
  --budget-floor-ratio 0.9 \
  --agent-period 50000 \
  --episodes 120 \
  --end-time 1000000 \
  --validation-end-time 1000000 \
  --reward-mode interval_v3_dense_B \
  --save-best-by pareto_relative_score \
  --enable-residual-safety-fallback \
  --residual-guard-hi-pressure-delta-limit 0.03 \
  --residual-guard-hi-pressure-abs-limit 0.30 \
  --residual-guard-reject-decrease-pressure-threshold 0.05
```

#### 2) Step 2：Residual Ranked Action Space

`--action-space` 新增：

- `residual_ranked`
- `residual_safe_ranked`
- `residual_safe_adjust_15a`

该动作空间固定 15 个槽位（`action_id=0..14`）：

- `0`: `noop`
- `1..4`: `increase_lo_risk(rank=0..3)`
- `5..6`: `decrease_lowest_risk(rank=0..1)`
- `7..8`: `decrease_lo_lowest_risk(rank=0..1)`
- `9..11`: `transfer_to_lo_risk_from_global_low(rank=0..2, decrease_pool=global_low_risk, decrease_count=1)`
- `12..13`: `transfer_to_lo_risk_from_lo_low(rank=0..1, decrease_pool=lo_low_risk, decrease_count=1)`
- `14`: `transfer_to_lo_risk_from_global_low2(rank=0, decrease_pool=global_low_risk, decrease_count=2)`

说明：

- `residual_ranked` 内部已经包含 noop，不需要再传 `--include-explicit-noop`。
- 训练日志会输出 residual 槽位的 concrete 解析结果，关键字段包括：
  - `residual_action_type`
  - `residual_rank`
  - `residual_resolved_increase_task`
  - `residual_resolved_decrease_tasks`
  - `residual_resolved_increase_idx`
  - `residual_resolved_decrease_indices`
- residual safety guard 已改为基于 `concrete action` 执行，因此 transfer / decrease 动作会按真实 decrease 目标检查风险。

Residual 动作槽位诊断脚本：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/diagnose_residual_ranked_actions.py
```

输出包含以下摘要字段，可快速核验动作空间是否按计划生效：

- `action_space=residual_ranked`
- `action_count=15`
- `transfer_action_count=6`
- `noop_count=1`
- `all_action_ids_contiguous=True`

Step 2 示例（`residual_ranked`）：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/train_dqn_amc.py \
  --workload mc_fairgen \
  --action-space residual_ranked \
  --budget-increase-ratio 0.025 \
  --budget-decrease-ratio 0.015 \
  --budget-floor-ratio 0.9 \
  --agent-period 50000 \
  --episodes 120 \
  --end-time 1000000 \
  --validation-end-time 1000000 \
  --reward-mode interval_v3_dense_B \
  --save-best-by pareto_relative_score \
  --enable-residual-safety-fallback \
  --residual-guard-hi-pressure-delta-limit 0.03 \
  --residual-guard-hi-pressure-abs-limit 0.30 \
  --residual-guard-reject-decrease-pressure-threshold 0.05 \
  --log-validation-policy-actions
```

#### 3) Step 3：Residual Safe Ranked Action Space

`residual_safe_ranked` 是在 `residual_ranked` 基础上的安全版 15 槽位动作空间，目标是避免策略退化到 pure decrease：

- `0`: `noop`
- `1..4`: `safe_increase_lo_risk(rank=0..3)`
- `5..8`: `safe_transfer_global_low_to_lo_risk(rank=0..3, decrease_pool=global_low_risk, decrease_count=1)`
- `9..12`: `safe_transfer_lo_low_to_lo_risk(rank=0..3, decrease_pool=lo_low_risk, decrease_count=1)`
- `13..14`: `safe_transfer_global_low2_to_lo_risk(rank=0..1, decrease_pool=global_low_risk, decrease_count=2)`

实现语义（严格按计划）：

- 不包含任何 pure decrease 动作；
- `safe_*` 动作在环境中先枚举 concrete candidates，再执行 residual guard + checker 过滤；
- `residual_rank=k` 表示第 `k` 个“安全可执行候选”；
- `valid_action_mask()` 与 `step()` 复用同一 safe 解析逻辑，mask 详情会记录：
  - `safe_candidate`
  - `safe_reject_reason`
  - `resolved_increase_task`
  - `resolved_decrease_tasks`
  - `increase_idx`
  - `decrease_indices`

`residual_safe_ranked` 训练示例：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/train_dqn_amc.py \
  --workload mc_fairgen \
  --action-space residual_safe_ranked \
  --budget-increase-ratio 0.025 \
  --budget-decrease-ratio 0.015 \
  --budget-floor-ratio 0.9 \
  --forbid-decreasing-hi-budgets \
  --enable-residual-safety-fallback \
  --log-validation-policy-actions
```

Residual 动作诊断脚本（支持 safe space 与 HI 降预算约束）：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/diagnose_residual_ranked_actions.py \
  --action-space residual_safe_ranked \
  --forbid-decreasing-hi-budgets
```

脚本会输出以下关键摘要字段用于快速验收：

- `action_count=15`
- `safe_adjust_increase_action_count=4`
- `guarded_decrease_action_count=0`
- `transfer_action_count>=6`
- `all_action_ids_contiguous=True`

#### 4) Step 4：Residual Safe Adjust 15A Action Space

`residual_safe_adjust_15a` 是不使用 transfer、同时保留 safe increase/safe decrease 的 15 槽位动作空间：

- `0`: `noop`
- `1..8`: `safe_increase_lo_utility(rank=0..7)`
- `9..14`: `safe_decrease_lo_redundant(rank=0..5)`

实现语义：

- 不使用固定 task-name anchor；
- 不使用 transfer；
- decrease 仅作用于 LO 任务；
- decrease 候选先按 redundant 评分排序，再统一走 `_budget_candidate_reject_reason(...)` 安全检查；
- `valid_action_mask()` 和 `step()` 使用同一解析器，避免 mask/执行语义不一致。

训练示例：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/train_dqn_amc.py \
  --workload mc_fairgen \
  --action-space residual_safe_adjust_15a \
  --budget-increase-ratio 0.025 \
  --budget-decrease-ratio 0.015 \
  --budget-floor-ratio 0.9 \
  --forbid-decreasing-hi-budgets \
  --enable-residual-safety-fallback \
  --log-validation-policy-actions
```

诊断示例：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/diagnose_residual_ranked_actions.py \
  --action-space residual_safe_adjust_15a \
  --forbid-decreasing-hi-budgets
```

诊断输出重点：

- `action_count=15`
- `safe_adjust_increase_action_count=8`
- `guarded_decrease_action_count=6`
- `transfer_action_count=0`

`--log-validation-policy-actions` 使用说明：

- 该开关默认关闭；开启后会在每个 validation checkpoint 记录并聚合策略动作分布。
- 会在 `validation_metrics.csv` 追加以下 JSON 聚合列：
  - `policy_action_definitions_json`
  - `policy_action_hist_json`
  - `policy_action_accepted_hist_json`
  - `policy_action_rejected_hist_json`
  - `policy_action_type_hist_json`
  - `policy_resolved_increase_task_hist_json`
  - `policy_resolved_decrease_task_hist_json`
  - `policy_action_reward_sum_json`
  - `policy_action_lo_delta_sum_json`
  - `policy_action_mode_delta_sum_json`
- 会额外输出长表文件 `validation_policy_actions.csv`，字段如下：
  - `episode`
  - `action_id`
  - `action_name`
  - `action_type`
  - `count`
  - `accepted_count`
  - `rejected_count`
  - `accepted_rate`
  - `reward_sum`
  - `reward_mean`
  - `lo_delta_sum`
  - `lo_delta_mean`
  - `mode_delta_sum`
  - `mode_delta_mean`
  - `resolved_increase_task`
  - `resolved_decrease_task`
  - `resolved_increase_tasks_json`
  - `resolved_decrease_tasks_json`

## 12. DQN 训练、评估与绘图

当前仓库已经包含最小 DQN 接入、正式 DQN CLI、训练诊断绘图脚本，以及可接入的 automotive workload 生成器。

## 13. 最小 taskset slack 验证（阶段 A/B）

本节对应 `minimal_taskset_slack_validation_plan.md` 的阶段 A、阶段 B。

### 13.1 阶段 A：扫描 baseline 与可调余量

脚本：`scripts/scan_taskset_headroom.py`

用途：
- 扫描多个 `fixed_taskset_seed × budget_scale` 组合；
- 对每个组合汇总 baseline 事件指标与动作空间/安全余量诊断指标；
- 输出二维扫描 CSV，用于分析预算缩放对 headroom 与事件强度的影响。

第一阶段（并行化改造前置重构）使用说明：
- 命令行参数与输出 CSV 字段保持不变，现有调用命令无需修改；
- 脚本内部已重构为 `ScanConfig` + `scan_one_taskset_seed(...)` 的结构化串行执行路径；
- 当前版本等价于按 `fixed_taskset_seed` 逐个串行扫描，便于后续在不改统计口径的前提下接入 `--workers` 并行化；
- 如需二次开发，建议直接复用 `scan_one_taskset_seed(taskset_seed, config)`，并通过 `ScanConfig.from_args(...)` 构造参数。

第二/三阶段（并行化 + 安全写出）使用说明：
- 新增参数：`--workers N`，用于控制并行进程数；`--workers 1` 为完全串行模式；
- 并行粒度是 `fixed_taskset_seed × budget_scale` 级别；
- 运行时会打印进度日志：`[scan] completed x/y fixed_taskset_seed=... budget_scale=...`；
- 任意 worker 失败会直接中断并报出出错 seed；
- CSV 由主进程统一写出，先写到 `*.tmp`，成功后原子替换到目标 `--output` 文件，避免中途失败留下半写结果。

本次 budget scale 扫描改动使用说明：
- 新增参数：`--budget-scales`，支持逗号分隔浮点列表（如 `0.85,0.90,0.95,1.00,1.05`）；
- 扫描种子参数支持 `--fixed-taskset-seeds` 的两种写法：`0,1,2` 或 `0:3`（半开区间，不含右端）；
- 评估种子支持 `--seeds`（推荐）或 `--eval-seeds`，两者均支持 `a,b,c` 与 `a:b`（半开区间）；
- `budget_scale` 会先作用到任务 `c_lo`（含 floor/upper bound 裁剪），再同时用于 baseline 与 diagnostic，保证两者预算口径一致；
- 输出按 `fixed_taskset_seed` 升序、`budget_scale` 升序排序。
- 新增 manifest 扫描模式：可通过 `--taskset-manifest` 直接读取
  `generate_learnable_tasksets.py` 产生的 accepted 行，不再依赖
  `paper_learnable_headroom + require_schedulable` 的内部搜索。

### 从 manifest 扫描 accepted taskset

当提供 `--taskset-manifest` 时：
- 扫描种子来自 manifest 的 `candidate_seed`（或 `--manifest-seed-column` 指定列）；
- `--fixed-taskset-seeds` 可省略，若同时提供则以 manifest 为准；
- 任务构造路径强制复用 two-stage 逻辑：`paper_exact(require_schedulable=True)` +
  `learnable headroom budget rewrite`，保证与生成脚本一致；
- 同一 `candidate_seed` 下 taskset 固定，仅随 `--seeds` 变化 execution scenario。

新增参数：
- `--taskset-manifest`
- `--manifest-seed-column`
- `--manifest-seed-limit`
- `--manifest-filter-recommended`
- `--manifest-output-selected`
- `--manifest-strict-parameter-check`

示例：

```bash
conda run -n amc-repro env PYTHONPATH=. python -u scripts/scan_taskset_headroom.py \
  --workload automotive \
  --taskset-manifest outputs/tasksets/paper_learnable_constraint_guided_r150_manifest_500_inc002.csv \
  --manifest-seed-column candidate_seed \
  --manifest-seed-limit 30 \
  --budget-scales 1.00 \
  --seeds 200:229 \
  --end-time 10000000 \
  --agent-period 100000 \
  --enable-constraint-guided-pair-diagnostic \
  --constraint-guided-pair-min-valid-count 2 \
  --constraint-guided-pair-top-k-risk 3 \
  --constraint-guided-pair-top-k-decrease 4 \
  --constraint-guided-pair-prefer-lo \
  --workers 4 \
  --manifest-output-selected outputs/tasksets/paper_learnable_constraint_guided_r150_manifest_selected.csv \
  --output outputs/taskset_slack_scan/paper_learnable_constraint_guided_r150_formal_scan.csv
```

示例命令（与计划文档对齐）：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro env PYTHONPATH=. python scripts/scan_taskset_headroom.py \
  --workload automotive \
  --automotive-mode paper_exact \
  --automotive-num-runnables 150 \
  --require-schedulable \
  --fixed-taskset-seeds 0,1,2 \
  --budget-scales 0.85,0.90,0.95,1.00,1.05 \
  --seeds 200:229 \
  --end-time 10000000 \
  --agent-period 100000 \
  --reward-mode interval_v1 \
  --action-space single \
  --budget-increase-ratio 0.025 \
  --budget-decrease-ratio 0.0125 \
  --include-explicit-noop \
  --budget-floor-ratio 0.9 \
  --observation-mode v11_full_10d \
  --ema-alpha 0.2 \
  --overrun-ema-alpha 0.1 \
  --history-k 8 \
  --event-window 10 \
  --max-cost-weight 0.7 \
  --risk-max-scale 3.0 \
  --include-safety-margin \
  --workers 4 \
  --output outputs/taskset_slack_scan/paper_exact_r150_seed012_budget_scale_scan.csv
```

主要输出字段：
- 组合键与缩放诊断：`fixed_taskset_seed`、`budget_scale`、`budget_scaled_task_count`、`budget_scale_effective_mean`、`budget_scale_effective_min`、`budget_scale_effective_max`；
- baseline 指标：`baseline_mode_changes_mean`、`baseline_lo_cancellations_mean`、`baseline_deadline_misses_sum` 等；
- headroom 指标：`valid_action_count_mean`、`valid_increase_count_mean`、`valid_decrease_count_mean`、`increase_decrease_balance`；
- ranked-pair 诊断指标（启用 `--enable-ranked-pair-diagnostic` 后输出）：
  `ranked_pair_candidate_count_mean`、`valid_ranked_pair_count_mean`、`valid_ranked_pair_count_no_safety_mean`、
  `valid_ranked_pair_to_single_increase_ratio`、`ranked_pair_reject_incremental_constraint_violation_mean`；
- headroom 分组：`headroom_group`（legacy）、`total_headroom_group`、`increase_headroom_group`、`decrease_headroom_group`、`balanced_headroom_group`；
- HI/LO 动作细分：`valid_increase_hi_count_mean`、`valid_increase_lo_count_mean`、`valid_decrease_hi_count_mean`、`valid_decrease_lo_count_mean`；
- 安全余量指标：`safety_margin_min_mean`、`safety_margin_min_p05`、`safety_margin_min_fraction_zero`；
- 预算利用率与风险指标：`total_budget_util_sum`、`hi_budget_util_sum`、`lo_budget_util_sum`、`risk_mean`、`surplus_mean`；
- 自动分组与推荐：`event_group`、`slack_group`、`recommended_for_dqn`、`not_recommended_reason`。

脚本结束时会打印 summary：
- 扫描 taskset 数；
- schedulable 数量；
- 推荐给 DQN 的数量；
- slack/event 分布统计。

### 13.2 阶段 B：选择代表性 taskset

脚本：`scripts/select_representative_tasksets.py`

用途：
- 读取阶段 A 扫描 CSV；
- 强制包含 `--must-include` 指定 seed（默认 1）；
- 自动补齐 medium/high event 候选，最终输出 `selected_tasksets.csv`。

示例命令：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro env PYTHONPATH=. python scripts/select_representative_tasksets.py \
  --input outputs/taskset_slack_scan/paper_exact_r150_taskset_scan.csv \
  --output outputs/taskset_slack_scan/selected_tasksets.csv \
  --current-seed 1 \
  --max-tasksets 5 \
  --min-tasksets 3
```

输出字段：
- `rank`、`fixed_taskset_seed`、`selection_roles`、`selection_reasons`；
- baseline 事件指标：`baseline_mode_changes_mean`、`baseline_lo_cancellations_mean`、`baseline_total_events_mean`；
- headroom 指标：`valid_increase_count_mean`、`valid_decrease_count_mean`、`increase_headroom_group`、`balanced_headroom_group`；
- 安全与推荐字段：`safety_margin_min_p05`、`recommended_for_dqn`、`not_recommended_reason`。

脚本结束时会打印：
- `Selected tasksets:`
- 每个候选的 seed / roles / 事件强度 / increase 余量 / 平衡分组。

### 13.3 `paper_learnable_headroom` 生成与使用（新增）

本次新增了 automotive 模式：`paper_learnable_headroom`。  
该模式保留 `paper_exact` 的任务结构生成语义，只替换初始预算生成逻辑：

- 按 `min/max budget` 与 HI/LO 分别的 `rho` 区间采样预算；
- 再按目标 `total budget utilization` 做整体缩放并回夹到 `min/max`；
- 生成过程可复现（固定 seed + 参数会得到相同 taskset）。

新增脚本：`scripts/generate_learnable_tasksets.py`，用于批量生成候选并筛选：
- 先做静态 reserve 检查（increase/HI increase/decrease）；
- 在 fast diagnostic 前增加 AMCRTB 设计时可调度性预检查（`amc_rtb + opa`）；
- 再做 fast 诊断（events/headroom/deadline_miss）；
- 输出：
  - `--output-manifest`：通过筛选的 accepted taskset；
  - `--output-rejections`：被拒绝候选及原因。

修复后的关键行为：
- 新增 `--require-schedulable` 参数，可要求 workload 在生成阶段先满足可调度；
- 新增 `--learnable-generation-strategy`：
  - `two_stage_from_paper_exact`（默认，推荐）；
  - `direct`（保留为可选对照策略）；
- 新增 safety-mask-aware 放松参数（用于在 fast diagnostic 前释放 safety-valid increase）：
  - `--learnable-enable-safety-relaxation`
  - `--learnable-relax-target-valid-increase`
  - `--learnable-relax-max-rounds`
  - `--learnable-relax-step-ratio`
  - `--learnable-relax-min-budget-floor-ratio`
- `R_LO 不可解`、安全检查器构造失败、可调度性相关错误会被细化写入 `reject_reason`，不会中断整个生成流程；
- `manifest/rejections` 都会包含预算利用率与静态 reserve 等关键元数据字段，便于复现与失败归因。
- fast diagnostic 新增 mask 诊断字段（用于定位“动作空间不足”还是“安全约束过强”）：
  - `fast_valid_increase_count_no_safety_mean`
  - `fast_valid_decrease_count_no_safety_mean`
  - `fast_mask_reject_incremental_constraint_violation_mean`
  - `fast_mask_reject_no_effective_budget_change_mean`
  - `fast_mask_reject_budget_floor_violation_mean`
  - `fast_mask_reject_budget_upper_bound_violation_mean`
  - `fast_mask_reject_decrease_hi_forbidden_mean`
- ranked-pair diagnostic（仅诊断，不改训练动作空间）新增参数：
  - `--enable-ranked-pair-diagnostic`
  - `--ranked-pair-min-valid-count`
  - `--ranked-pair-top-k-risk`
  - `--ranked-pair-top-k-surplus`
  - `--ranked-pair-decrease-mode`（`top1_surplus`/`top2_surplus`/`topk_surplus`）
  - `--ranked-pair-include-single-controls`
- 启用 ranked-pair 后新增输出字段：
  - `fast_ranked_pair_candidate_count_mean`
  - `fast_valid_ranked_pair_count_mean`
  - `fast_valid_ranked_pair_count_no_safety_mean`
  - `fast_valid_ranked_pair_to_single_increase_ratio`
  - `fast_ranked_pair_reject_incremental_constraint_violation_mean`
  - `fast_ranked_pair_reject_budget_floor_violation_mean`
  - `fast_ranked_pair_reject_budget_upper_bound_violation_mean`
  - `fast_ranked_pair_reject_no_effective_budget_change_mean`
  - `fast_ranked_pair_reject_decrease_hi_forbidden_mean`
  - `fast_ranked_pair_reject_unknown_mean`
  - `fast_ranked_pair_reject_unknown_no_safety_mean`
  - `fast_ranked_pair_reject_hi_lo_mode_violation_mean`
  - `fast_ranked_pair_reject_hi_mode_switch_violation_mean`
  - `fast_ranked_pair_reject_lo_mode_violation_mean`
  - `recommended_for_ranked_pair_dqn`
  - `ranked_pair_not_recommended_reason`

示例命令：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro env PYTHONPATH=. python scripts/generate_learnable_tasksets.py \
  --automotive-num-runnables 150 \
  --num-tasksets 3 \
  --candidate-seed-start 0 \
  --learnable-max-attempts 50 \
  --learnable-target-budget-util-min 0.35 \
  --learnable-target-budget-util-max 0.55 \
  --learnable-hi-budget-rho-min 0.35 \
  --learnable-hi-budget-rho-max 0.55 \
  --learnable-lo-budget-rho-min 0.25 \
  --learnable-lo-budget-rho-max 0.50 \
  --learnable-min-static-increase-reserve 4 \
  --learnable-min-static-hi-increase-reserve 2 \
  --learnable-min-static-decrease-reserve 4 \
  --learnable-fast-end-time 500000 \
  --learnable-fast-eval-seeds 3 \
  --learnable-fast-event-min 3 \
  --learnable-fast-event-max 80 \
  --learnable-fast-min-balance 0.12 \
  --require-schedulable \
  --reward-mode interval_v1 \
  --action-space single \
  --budget-increase-ratio 0.025 \
  --budget-decrease-ratio 0.0125 \
  --include-explicit-noop \
  --budget-floor-ratio 0.9 \
  --observation-mode v11_full_10d \
  --ema-alpha 0.2 \
  --overrun-ema-alpha 0.1 \
  --history-k 8 \
  --event-window 10 \
  --max-cost-weight 0.7 \
  --risk-max-scale 3.0 \
  --include-safety-margin \
  --output-manifest /tmp/learnable_manifest.csv \
  --output-rejections /tmp/learnable_rejections.csv
```

训练/评估/扫描脚本也已支持：
- `--automotive-mode paper_learnable_headroom`
- `--learnable-target-budget-util-min/max`
- `--learnable-hi-budget-rho-min/max`
- `--learnable-lo-budget-rho-min/max`

### 12.0 v11/v12 observation 配置与使用说明

当前训练/评估 CLI 已支持通过参数启用：

- `--observation-mode v10_basic`（默认，旧模式，`state_dim = 2 * n_tasks`）
- `--observation-mode v11_full_10d`（新模式，`state_dim = 10 * n_tasks + 8`）
- `--observation-mode v11_no_risk_9d`（v11 消融模式，`state_dim = 9 * n_tasks + 8`）
- `--observation-mode v11_no_util_9d`（v11 消融模式，`state_dim = 9 * n_tasks + 8`）
- `--observation-mode v11_no_max_9d`（v11 消融模式，`state_dim = 9 * n_tasks + 8`）
- `--observation-mode v11_no_priority_9d`（v11 消融模式，`state_dim = 9 * n_tasks + 8`）
- `--observation-mode v11_no_risk_no_util_8d`（v11 消融模式，`state_dim = 8 * n_tasks + 8`）
- `--observation-mode v11_lite_6d`（v11 紧凑模式，`state_dim = 6 * n_tasks + 8`）
- `--observation-mode v12_full_14d`（新模式，`state_dim = 14 * n_tasks + 8`）

所有新增 v11 消融模式都复用了 `v11_full_10d` 的同一套底层特征计算逻辑，
区别只在于“保留哪些 per-task 特征，以及按什么固定顺序拼接”，因此可直接做同口径对比。

`v11_full_10d` 的每任务 10 维顺序为：

1. `budget_norm`
2. `recent_cost_norm`
3. `ema_cost_norm`
4. `max_cost_k_norm`
5. `overrun_ema`
6. `risk`
7. `surplus`
8. `criticality`
9. `priority_norm`
10. `util_budget`

新增模式的每任务特征顺序如下：

- `v11_no_risk_9d`
  - `budget_norm`
  - `recent_cost_norm`
  - `ema_cost_norm`
  - `max_cost_k_norm`
  - `overrun_ema`
  - `surplus`
  - `criticality`
  - `priority_norm`
  - `util_budget`
- `v11_no_util_9d`
  - `budget_norm`
  - `recent_cost_norm`
  - `ema_cost_norm`
  - `max_cost_k_norm`
  - `overrun_ema`
  - `risk`
  - `surplus`
  - `criticality`
  - `priority_norm`
- `v11_no_max_9d`
  - `budget_norm`
  - `recent_cost_norm`
  - `ema_cost_norm`
  - `overrun_ema`
  - `risk`
  - `surplus`
  - `criticality`
  - `priority_norm`
  - `util_budget`
- `v11_no_priority_9d`
  - `budget_norm`
  - `recent_cost_norm`
  - `ema_cost_norm`
  - `max_cost_k_norm`
  - `overrun_ema`
  - `risk`
  - `surplus`
  - `criticality`
  - `util_budget`
- `v11_no_risk_no_util_8d`
  - `budget_norm`
  - `recent_cost_norm`
  - `ema_cost_norm`
  - `max_cost_k_norm`
  - `overrun_ema`
  - `surplus`
  - `criticality`
  - `priority_norm`
- `v11_lite_6d`
  - `budget_norm`
  - `recent_cost_norm`
  - `ema_cost_norm`
  - `overrun_ema`
  - `surplus`
  - `criticality`

以上这些 v11-family 模式都保留相同的全局 8 维：

1. `total_budget_util`
2. `hi_budget_util`
3. `lo_budget_util`
4. `recent_mode_change_rate`
5. `recent_lo_cancel_rate`
6. `recent_hi_overrun_rate`
7. `recent_lo_overrun_rate`
8. `safety_margin_min`

`v12_full_14d` 在 `v11_full_10d` 的每任务 10 维特征后，追加以下 4 个特征（顺序固定）：

- `positive_budget_drift`
- `negative_budget_drift`
- `task_cancel_ema`
- `safe_inc_possible`

重要说明：

- `task_cancel_ema` 第一版使用 task-level overrun event 作为 cancellation pressure proxy；
- 该值不是严格的 per-task LO cancellation 计数；
- `safe_inc_possible` 仅作为 observation hint，不会改变 action mask 与动作执行语义。

对应 v11 特征参数：

- `--ema-alpha`
- `--overrun-ema-alpha`
- `--history-k`
- `--event-window`
- `--max-cost-weight`
- `--risk-max-scale`
- `--include-safety-margin / --no-include-safety-margin`

训练示例（仅展示 observation 相关参数）：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/train_dqn_amc.py \
  --episodes 2 \
  --workload small \
  --scenario stress \
  --action-space single \
  --observation-mode v11_full_10d \
  --ema-alpha 0.2 \
  --overrun-ema-alpha 0.1 \
  --history-k 8 \
  --event-window 10 \
  --max-cost-weight 0.7 \
  --risk-max-scale 3.0 \
  --include-safety-margin
```

v11 消融训练示例：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/train_dqn_amc.py \
  --episodes 2 \
  --workload small \
  --scenario stress \
  --action-space single \
  --observation-mode v11_no_risk_9d \
  --ema-alpha 0.2 \
  --overrun-ema-alpha 0.1 \
  --history-k 8 \
  --event-window 10 \
  --max-cost-weight 0.7 \
  --risk-max-scale 3.0 \
  --include-safety-margin
```

如果要切换到其它消融模式，只需要替换 `--observation-mode`：

- `v11_no_util_9d`
- `v11_no_max_9d`
- `v11_no_priority_9d`
- `v11_no_risk_no_util_8d`
- `v11_lite_6d`

评估示例（需与训练期 observation 配置保持一致）：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/evaluate_dqn_amc.py \
  --model outputs/dqn_amc/model_final.pt \
  --workload small \
  --scenario stress \
  --action-space single \
  --observation-mode v11_full_10d \
  --ema-alpha 0.2 \
  --overrun-ema-alpha 0.1 \
  --history-k 8 \
  --event-window 10 \
  --max-cost-weight 0.7 \
  --risk-max-scale 3.0 \
  --include-safety-margin
```

v12 训练示例（仅展示 observation 相关参数）：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/train_dqn_amc.py \
  --episodes 2 \
  --workload small \
  --scenario stress \
  --action-space single \
  --observation-mode v12_full_14d \
  --ema-alpha 0.2 \
  --overrun-ema-alpha 0.1 \
  --history-k 8 \
  --event-window 10 \
  --max-cost-weight 0.7 \
  --risk-max-scale 3.0 \
  --include-safety-margin
```

日志与产物中已记录：

- step info / 训练日志：`observation_mode`、`state_dim`
- validation 指标：`observation_mode`、`state_dim_mean`
- 评估 CSV：`observation_mode`、`state_dim`
- 训练配置快照 `config.json`：`observation_mode` 与 `feature_config`

### 12.0.1 observation 测试与冒烟验证

新增测试文件：

- `tests/test_v11_observation.py`

覆盖项：

1. `v10_basic` 长度保持 `2 * n_tasks`
2. `v11_full_10d` 长度为 `10 * n_tasks + 8`
3. `v11_full_10d` 所有特征值在 `[0, 1]`
4. `step` 后 `v11_full_10d` 维度保持正确
5. 所有新增 v11 消融模式维度正确
6. 所有新增 v11 消融模式的特征值均位于 `[0, 1]`
7. `feature_state` 在 step 后存在且任务键集合保持一致
8. event window 长度不超过 `event_window`

运行命令：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q tests/test_v11_observation.py
```

新增 observation 全模式冒烟脚本：

- `scripts/smoke_test_v12_observation.py`

运行方式：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/smoke_test_v12_observation.py
```

预期输出：

```text
v10_basic: PASS
v11_full_10d: PASS
v11_no_risk_9d: PASS
v11_no_util_9d: PASS
v11_no_max_9d: PASS
v11_no_priority_9d: PASS
v11_no_risk_no_util_8d: PASS
v11_lite_6d: PASS
v12_full_14d: PASS
```

新增 v11 消融顺序校验脚本：

- `scripts/smoke_test_v11_lite_feature_order.py`

运行方式：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/smoke_test_v11_lite_feature_order.py
```

该脚本会验证所有新增模式是否严格等于 `v11_full_10d` 删除指定列后的结果，
并同时确认全局 8 维未发生漂移。预期输出：

```text
v11_no_risk_9d feature order: PASS
v11_no_util_9d feature order: PASS
v11_no_max_9d feature order: PASS
v11_no_priority_9d feature order: PASS
v11_no_risk_no_util_8d feature order: PASS
v11_lite_6d feature order: PASS
```

### 12.1 运行前说明

所有 DQN 命令都应在 `amc-repro` 环境中运行：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -c "import torch; import amc_py; print('ok')"
```

如果在 macOS 上导入 `torch` 时遇到 `libomp` / OpenMP 重复加载问题，可在命令前追加：

```bash
KMP_DUPLICATE_LIB_OK=TRUE
```

例如：

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro python -m pytest -q
```

### 12.2 Smoke 训练

最小 smoke 训练入口：

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro python scripts/train_dqn_smoke.py
```

可选参数：

- `--episodes`
- `--end-time`
- `--agent-period`
- `--seed`
- `--output-dir`
- `--batch-size`
- `--learning-rate`
- `--gamma`

默认输出目录：`outputs/dqn_smoke`

输出文件：

- `outputs/dqn_smoke/train_log.csv`
- `outputs/dqn_smoke/model.pt`

### 12.3 Smoke 评估

训练完成后，可加载 smoke 模型并与 baseline 统一比较：

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro python scripts/evaluate_dqn_smoke.py \
  --model outputs/dqn_smoke/model.pt
```

输出文件：

- `outputs/dqn_smoke/eval_summary.csv`

比较对象包括：

- `amc_plus_baseline`
- `noop_agent`
- `random_agent`
- `heuristic_agent`
- `dqn_agent`

### 12.4 正式训练 CLI

正式训练入口：

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro python scripts/train_dqn_amc.py \
  --episodes 10 \
  --end-time 100 \
  --agent-period 10 \
  --seed 0 \
  --batch-size 6 \
  --learning-rate 5e-5 \
  --gamma 0.99 \
  --target-update-freq 5 \
  --epsilon-start 1.0 \
  --epsilon-end 0.05 \
  --epsilon-decay-steps 1000 \
  --checkpoint 5 \
  --scenario stress \
  --output-dir outputs/dqn_amc
```

支持参数：

- `--episodes`
- `--end-time`
- `--agent-period`
- `--seed`
- `--batch-size`
- `--hidden-layers`
- `--learning-rate`
- `--gamma`
- `--target-update-freq`
- `--epsilon-start`
- `--epsilon-end`
- `--epsilon-decay-steps`
- `--output-dir`
- `--checkpoint`
- `--scenario`
- `--save-best-by`

输出文件：

- `outputs/dqn_amc/train_log.csv`
- `outputs/dqn_amc/model_final.pt`
- `outputs/dqn_amc/config.json`
- `outputs/dqn_amc/checkpoints/model_episode_XXXX.pt`（当 `--checkpoint > 0` 时）

### 12.4.1 `--save-best-by` 策略说明

`train_dqn_amc.py` 会在 validation 过程中维护 `model_best.pt`。  
`--save-best-by` 用于决定“什么叫更好的 checkpoint”。

当前支持：

1. `--save-best-by mode_changes`
- 目标：`mode_changes_mean` 越小越好。
- 约束：`deadline_misses_sum` 必须为 0，否则候选不会被选为 best。

2. `--save-best-by lo_cancellations`
- 目标：`lo_cancellations_mean` 越小越好。
- 额外门槛：候选必须满足 `mode_changes_mean <= baseline_mode_changes_mean`，即 mode-change 不能比 baseline 更差。
- 同样要求 `deadline_misses_sum == 0`。

3. `--save-best-by reward`
- 目标：`reward_mean` 越大越好。
- 同样要求 `deadline_misses_sum == 0`。

4. `--save-best-by relative_score`
- 目标：`relative_score` 越小越好。
- 定义（validation 聚合后）：
  `relative_score = relative_delta_lo_cancellations + alpha * relative_delta_mode_changes`
- 其中 `alpha` 由 `--relative-score-alpha` 控制。
- 同样要求 `deadline_misses_sum == 0`。

5. `--save-best-by pareto_relative_score`（温和版本）
- 目标：在 `relative_score` 基础上，对“劣于 baseline”的维度做软惩罚，分数越小越好。
- 定义：
  `pareto_relative_score = relative_score + 10 * max(0, relative_delta_mode_changes) + 10 * max(0, relative_delta_lo_cancellations)`
- 解释：
  - 如果两个 delta 都 `<= 0`（不劣于 baseline），该策略退化为 `relative_score` 比较；
- 如果某一维劣于 baseline，不会硬性淘汰，而是增加惩罚（更温和）。
- 同样要求 `deadline_misses_sum == 0`。

6. `--save-best-by conservative_qos`
- 目标：在 `HI` 安全前提下，优先最小化 `lc_service_loss_mean`。
- 约束：`hi_deadline_misses_sum == 0` 且 `mode_changes_mean <= baseline_mode_changes_mean`。
- 并列时继续比较：`min_lc_service_mean` 越大越好，`budget_adjust_count_mean` 越小越好，episode 越早越好。

7. `--save-best-by qos_stable`
- 目标：在 `HI` 安全前提下，按新的 QoS 稳定标准选择 best。
- 约束：`hi_deadline_misses_sum == 0` 且 `mode_changes_mean <= baseline_mode_changes_mean * (1 + --qos-stable-mode-delta)`。
- 默认 `--qos-stable-mode-delta 0.05`，即允许 mode change 相对 baseline 最多增加 5%。

8. `--save-best-by qos_best`
- 目标：只要求 `HI` 安全，然后按 `lc_service_loss_mean -> min_lc_service_mean -> mode_changes_mean -> budget_adjust_count_mean -> episode` 排序。
- 约束：`hi_deadline_misses_sum == 0`。

新增参数：

- `--qos-stable-mode-delta`：控制 `qos_stable` 的 mode-change 放宽比例，默认 `0.05`。
- `--save-all-best-types`：除主 `model_best.pt` 外，额外输出
  `model_best_conservative_qos.pt`、`model_best_qos_stable.pt`、`model_best_qos_best.pt`
  以及对应 metadata JSON。若某类型没有合格 checkpoint，只会写 `found_valid_checkpoint=false` 的 metadata，不会伪造 best 模型。

常用示例：

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro python scripts/train_dqn_amc.py \
  --save-best-by pareto_relative_score \
  --relative-score-alpha 1.0
```

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro python scripts/train_dqn_amc.py \
  --save-best-by qos_stable \
  --qos-stable-mode-delta 0.05 \
  --save-all-best-types
```

新的 validation / 选模输出说明：

- `validation_metrics.csv` 新增：
  `released_lo_jobs_mean`、`cancelled_lo_jobs_mean`、`completed_lo_jobs_mean`、
  `lo_deadline_misses_sum`、`hi_deadline_misses_sum`、
  `lc_service_loss_mean`、`lc_qos_mean`、`min_lc_service_mean`、
  `budget_adjust_count_mean`、`mean_abs_budget_change_mean`、
  `baseline_lc_service_loss_mean`、`relative_lc_loss_reduction`、`mode_change_delta_ratio`。
- `validation_unified_summary.csv` 新增：
  `qos_stable_valid_delta000`、`qos_stable_valid_delta005`、`qos_stable_valid_delta010`、
  `best_candidate_rank_key`、`dqn_lc_service_loss_mean`、`dqn_lc_qos_mean` 等字段。
- `best_model_metadata.json` 以及额外的 `best_model_metadata_*.json` 会明确记录：
  `best_type`、`found_valid_checkpoint`、`hi_deadline_misses_sum`、
  `lc_service_loss_mean`、`relative_lc_loss_reduction`、`mode_change_delta_ratio`。

对已有训练目录重新按新规则选模：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/select_qos_best_from_validation.py \
  --run-dir outputs/dqn_amc \
  --qos-stable-mode-delta 0.05
```

对多个训练目录比较新 best：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/compare_dqn_training_runs.py \
  --runs outputs/run_a,outputs/run_b \
  --best-type qos_stable \
  --qos-stable-mode-delta 0.05 \
  --output outputs/qos_compare.csv
```

配置参考文件：

- `configs/dqn_smoke.yaml`
- `configs/dqn_default.yaml`

### 12.5 正式评估 CLI

正式评估入口：

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro python scripts/evaluate_dqn_amc.py \
  --model outputs/dqn_amc/model_final.pt \
  --seeds 0,1,2 \
  --end-time 100 \
  --agent-period 10 \
  --scenario stress \
  --output outputs/dqn_amc/eval_summary.csv
```

输出文件：

- `outputs/dqn_amc/eval_summary.csv`
- `outputs/dqn_amc/eval_summary_unified_summary.csv`

评估明细 CSV 现已统一输出以下 QoS 字段，baseline / noop / random / heuristic / dqn 全部同口径：

- `released_lo_jobs`
- `cancelled_lo_jobs`
- `completed_lo_jobs`
- `lo_deadline_misses`
- `hi_deadline_misses`
- `lc_service_loss`
- `lc_qos`
- `min_lc_service`
- `budget_adjust_count`
- `mean_abs_budget_change`

### 12.6 训练诊断绘图

可以把正式训练日志直接绘制为诊断图：

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro python scripts/plot_dqn_training.py \
  --train-log outputs/dqn_amc/train_log.csv \
  --output-dir outputs/dqn_amc/plots
```

默认会生成：

- `episode_reward.png`
- `loss.png`
- `epsilon.png`
- `action_counts.png`

### 12.7 train_log.csv 关键字段

正式训练日志至少包含以下字段：

- `episode`
- `step`
- `sim_time`
- `reward`
- `episode_reward`
- `loss`
- `epsilon`
- `action_id`
- `accepted`
- `rejected`
- `reject_reason`
- `valid_action_count`
- `masked_action_count`
- `noop_due_to_no_valid_action`
- `mode_changes`
- `lo_cancellations`
- `deadline_misses`

这些字段可用于判断：

- reward 是否逐步改善
- loss 是否出现 `NaN`
- epsilon 是否按预期衰减
- agent 是否频繁选择被屏蔽动作或只能 NoOp

### 12.8 `interval_v3_dense_B` 使用说明（Step1 Dense Reward）

新增 reward mode 配置文件：

- `configs/reward_modes/interval_v3_dense_B.json`

该模式在原 interval reward 基础上新增两项 pressure 惩罚：

- `lo_pressure_penalty * lo_pressure_mean`
- `hi_mode_pressure_penalty * hi_mode_pressure_mean`

其中 pressure 计算口径为（按任务取均值）：

- `pressure_i = max(0, estimated_exec / budget_i - threshold)`
- `estimated_exec = max(recent_execution, ema_cost)`

运行方式（仅切换 reward mode，其余参数保持 v3 baseline 一致）：

```bash
cd /Users/x1ngchuan/Documents/AMC
python scripts/train_dqn_amc.py \
  --workload mc_fairgen \
  --action-space single \
  --budget-increase-ratio 0.025 \
  --budget-decrease-ratio 0.015 \
  --include-explicit-noop \
  --budget-floor-ratio 0.9 \
  --agent-period 50000 \
  --episodes 120 \
  --end-time 1000000 \
  --validation-end-time 1000000 \
  --reward-mode interval_v3_dense_B \
  --save-best-by pareto_relative_score
```

`train_log.csv` 新增以下字段用于观测 dense reward 实际生效强度：

- `lo_pressure_mean`
- `hi_mode_pressure_mean`
- `lo_pressure_penalty_value`
- `hi_mode_pressure_penalty_value`

### 12.9 `interval_v3_dense_LO_guard` 使用说明（Step2 Dense Reward）

新增 reward mode 配置文件：

- `configs/reward_modes/interval_v3_dense_LO_guard.json`

该模式在 `interval_v3_dense_B` 基础上，额外引入两个专门针对 LO 局部高风险的惩罚项：

- `lo_pressure_max_penalty * lo_pressure_max`
- `lo_near_cancel_penalty * lo_near_cancel_rate`

完整 reward 公式为：

- `original_interval_reward`
- `- lo_pressure_penalty * lo_pressure_mean`
- `- lo_pressure_max_penalty * lo_pressure_max`
- `- lo_near_cancel_penalty * lo_near_cancel_rate`
- `- hi_mode_pressure_penalty * hi_mode_pressure_mean`

其中新增指标定义如下：

- `lo_pressure_max`：所有 LO task 的 pressure 最大值，用于捕捉最危险任务；
- `lo_near_cancel_rate`：`estimated_exec / budget > lo_near_cancel_threshold` 的 LO 任务比例；
- `estimated_exec = max(recent_execution, ema_cost)`，用于避免低估风险。

运行方式（仅切换 reward mode，其余参数保持 v3 baseline 一致）：

```bash
cd /Users/x1ngchuan/Documents/AMC
python scripts/train_dqn_amc.py \
  --workload mc_fairgen \
  --action-space single \
  --budget-increase-ratio 0.025 \
  --budget-decrease-ratio 0.015 \
  --include-explicit-noop \
  --budget-floor-ratio 0.9 \
  --agent-period 50000 \
  --episodes 120 \
  --end-time 1000000 \
  --validation-end-time 1000000 \
  --reward-mode interval_v3_dense_LO_guard \
  --save-best-by pareto_relative_score
```

`train_log.csv` 中与该模式直接相关的字段包括：

- `lo_pressure_mean`
- `lo_pressure_max`
- `lo_near_cancel_rate`
- `hi_mode_pressure_mean`
- `lo_pressure_penalty_value`
- `lo_pressure_max_penalty_value`
- `lo_near_cancel_penalty_value`
- `hi_mode_pressure_penalty_value`

### 12.10 automotive workload 用法

当前仓库已经提供 automotive workload 生成器：

- `amc_py/automotive_workload.py`

支持能力：

- runnable period distribution
- ACET / BCET / WCET sampling
- Weibull execution-time sampling
- runnables -> tasks 聚合
- 每个 period、每个 criticality 最多一个 task
- `150 / 250` runnables 配置
- LO budget quantile 选择
- normalization bounds 输出
- AMC-rtb 可调度任务集筛选

最小 Python 用法示例：

```python
from amc_py.automotive_workload import build_automotive_experiment_config
from amc_py.dqn import build_env_from_experiment_config
from amc_py.runtime_models import RuntimeSemantics

experiment_config = build_automotive_experiment_config(
    num_runnables=150,
    require_schedulable=True,
    max_attempts=20,
)

env = build_env_from_experiment_config(
    experiment_config,
    seed=0,
    end_time=100,
    agent_period=10,
    semantics=RuntimeSemantics.AMC_PLUS,
)

obs = env.reset(seed=0)
step_result = env.step(None)
print(len(obs.state_vector), step_result.reward, step_result.done)
```

如果后续要把正式 DQN 训练入口切到 automotive workload，只需要把当前 small stress experiment config 替换为 `build_automotive_experiment_config(...)` 返回的配置对象，不需要重写 DQN agent。

## 13. MC-FairGen Workload（Step1-3）使用说明

已新增独立 workload 文件：`amc_py/workloads/mc_fairgen.py`，用于生成面向 LO cancellation 学习信号的 mixed-criticality 任务集。

### 13.1 当前实现范围（仅 Step1-3）

当前已实现：
- `MCFairGenWorkloadConfig` 与严格参数校验（非法字段直接 `ValueError`）。
- `uunifast_discard` 固定总 utilization 拆分。
- MC-FairGen 风格任务构造：
  - HI 任务：`c_lo` 为初始预算，`c_hi` 为 HI 上界；
  - LO 任务：强制 `c_hi == c_lo`。
- `build_mc_fairgen_execution_scenario`：
  - HI 任务小概率 overrun（不超过 `c_hi`）；
  - LO 任务高概率 overrun（允许 `actual_cost > c_lo`）。
- `build_mc_fairgen_normalization_bounds`：覆盖 HI 上界与 LO overrun 上界。
- `MCFairGenWorkloadProvider`：可直接产出 `WorkloadBundle`。

### 13.2 最小调用示例

```python
from amc_py.workloads.mc_fairgen import (
    MCFairGenWorkloadConfig,
    build_mc_fairgen_execution_scenario,
    build_mc_fairgen_workload,
)

config = MCFairGenWorkloadConfig(seed=0)
workload = build_mc_fairgen_workload(config)
scenario = build_mc_fairgen_execution_scenario(workload, scenario_seed=123)

print(len(workload.tasks))
print(workload.metadata)
print(scenario.actual_cost_for(workload.tasks[0], 0))
```

### 13.3 运行定向测试

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest tests/test_mc_fairgen_workload.py -q
```

该测试文件覆盖：配置校验、period 采样、UUniFast 总和约束、HI/LO 任务语义、metadata 字段、scenario 约束与 LO over-budget 信号。

### 13.4 Step4-6 新增接入说明

本次继续完成了 Step4-6：
- Step4：`MCFairGenWorkloadProvider` 已补齐可调度筛选元数据、`workload_family` 元数据、`__all__` 导出。
- Step5：`amc_py/dqn/experiment.py` 新增 `build_mc_fairgen_experiment_config(...)`，并可通过 `build_experiment_config(name="mc_fairgen", ...)` 解析。
- Step6：训练/评估 CLI 都支持 `--workload mc_fairgen`。

训练脚本新增参数（`scripts/train_dqn_amc.py`）：
- `--workload mc_fairgen`
- `--mc-fairgen-mode`
- `--mc-fairgen-num-tasks`
- `--mc-fairgen-hi-ratio`
- `--mc-fairgen-period-source`
- `--mc-fairgen-period-scale`
- `--mc-fairgen-u-hi-lo-min/--mc-fairgen-u-hi-lo-max`
- `--mc-fairgen-u-hi-hi-min/--mc-fairgen-u-hi-hi-max`
- `--mc-fairgen-u-lo-lo-min/--mc-fairgen-u-lo-lo-max`
- `--mc-fairgen-hi-budget-rho-min/--mc-fairgen-hi-budget-rho-max`
- `--mc-fairgen-lo-budget-rho-min/--mc-fairgen-lo-budget-rho-max`
- `--mc-fairgen-hi-overrun-prob/--mc-fairgen-lo-overrun-prob`
- `--mc-fairgen-hi-overrun-factor-min/--mc-fairgen-hi-overrun-factor-max`
- `--mc-fairgen-lo-overrun-factor-min/--mc-fairgen-lo-overrun-factor-max`

评估脚本新增同名参数（`scripts/evaluate_dqn_amc.py`），确保训练与评估配置口径一致。

最小训练 smoke 示例：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/train_dqn_amc.py \
  --workload mc_fairgen \
  --mc-fairgen-mode paper_learnable_headroom \
  --episodes 1 \
  --end-time 200 \
  --validate-every 1 \
  --validation-seeds 0 \
  --validation-end-time 200
```

### 13.5 Step7-8 新增接入说明

已完成以下脚本的 `mc_fairgen` 接入：
- `scripts/scan_taskset_headroom.py`
- `scripts/generate_learnable_tasksets.py`

`scan_taskset_headroom.py` 新增能力：
- `--workload automotive|mc_fairgen`
- 支持全套 `--mc-fairgen-*` 参数
- 新增输出列 `baseline_lo_cancellation_ratio_total`
- 当 `workload=mc_fairgen` 时，不再走 automotive 的 two-stage manifest 重写路径

`generate_learnable_tasksets.py` 新增能力：
- `--workload automotive|mc_fairgen`
- 支持全套 `--mc-fairgen-*` 参数
- 新增阈值 `--learnable-fast-min-lo-cancellation-ratio`
- 当 `workload=mc_fairgen` 时，生成逻辑使用 `mc_fairgen` provider，不走 `paper_exact -> rewrite` two-stage

最小 smoke（scan）：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/scan_taskset_headroom.py \
  --workload mc_fairgen \
  --mc-fairgen-mode paper_learnable_headroom \
  --mc-fairgen-num-tasks 16 \
  --mc-fairgen-hi-ratio 0.5 \
  --budget-scales 1.00 \
  --fixed-taskset-seeds 0 \
  --seeds 200:201 \
  --end-time 50000 \
  --agent-period 10000 \
  --reward-mode interval_v1 \
  --action-space single \
  --budget-increase-ratio 0.02 \
  --budget-decrease-ratio 0.0125 \
  --include-explicit-noop \
  --budget-floor-ratio 0.9 \
  --observation-mode v11_full_10d \
  --enable-constraint-guided-pair-diagnostic \
  --output outputs/taskset_slack_scan/smoke_mc_fairgen_scan.csv
```

最小 smoke（generate）：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/generate_learnable_tasksets.py \
  --workload mc_fairgen \
  --mc-fairgen-mode paper_learnable_headroom \
  --mc-fairgen-num-tasks 16 \
  --mc-fairgen-hi-ratio 0.5 \
  --num-tasksets 1 \
  --candidate-seed-start 0 \
  --learnable-max-attempts 3 \
  --learnable-fast-end-time 50000 \
  --learnable-fast-eval-seeds 1 \
  --learnable-fast-min-lo-cancellation-ratio 0.0 \
  --output-manifest outputs/tasksets/smoke_mc_fairgen_manifest.csv \
  --output-rejections outputs/tasksets/smoke_mc_fairgen_rejections.csv
```

### 13.6 MC-FairGen 修复版说明（seed/日志/manifest）

本次按修复计划补齐了以下关键行为：
- `scripts/generate_learnable_tasksets.py`：同一 `candidate_seed` 下，`learnable_fast_eval_seeds` 会固定 taskset、仅变化 scenario（不再复用同一个 scenario）。
- `scripts/train_dqn_amc.py` 与 `scripts/evaluate_dqn_amc.py`：`num_tasks` 按实际 workload 生效；`mc_fairgen_*` 参数写入配置输出。
- `scripts/generate_learnable_tasksets.py`：manifest/rejections 补齐 `mc_fairgen_*` 复现字段与 `fast_baseline_lo_cancellation_ratio_total`。
- `scripts/scan_taskset_headroom.py`：manifest roundtrip 时支持读取 `mc_fairgen` 参数并复现；输出包含 `baseline_lo_cancellation_ratio_total`。

新增回归测试：
- `tests/test_mc_fairgen_generator.py`
- `tests/test_mc_fairgen_cli_smoke.py`

推荐验证命令：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest \
  tests/test_mc_fairgen_workload.py \
  tests/test_mc_fairgen_experiment.py \
  tests/test_mc_fairgen_generator.py \
  tests/test_mc_fairgen_cli_smoke.py -q
```

## QoS-Pressure 任务集扫描与筛选（新增）

本仓库新增了 QoS-Pressure 工作流脚本，用于在 `mc_fairgen` 任务集上先做 baseline QoS 压力扫描，再筛选出训练 manifest，最后生成训练与对比命令。

### 1. QoS pressure 规则模块

- 文件：`amc_py/qos_pressure.py`
- 功能：
  - `classify_qos_pressure_bucket(lc_service_loss)`：按 `easy/medium/hard/overloaded/unknown` 分桶。
  - `classify_improvement_type(...)`：按 `stable005/stable010/tradeoff_only` 等类别输出改进类型。
  - `recommend_for_qos_dqn(...)`：支持 stable-improvement 与 tradeoff-only 约束，输出 `(是否推荐, 拒绝原因)`。

### 2. 扫描脚本

- 文件：`scripts/scan_qos_pressure_tasksets.py`
- 作用：扫描 candidate seed 的 AMCRTB baseline QoS 指标并输出 CSV；可选开启 static sweep。
- 说明：static sweep 是 **static budget scaling sweep**，仅在仿真前一次性缩放预算，用作低成本可学习性代理信号，不是运行中按周期动态动作策略。
- 新增：支持与 `--action-space single` 对齐的两类代理扫描：
  - `single-task sweep`：每次只扰动一个任务（increase/decrease，可重复多次）；
  - `single-sequence sweep`：按固定 single-action 序列逐步改预算后再仿真。

基础扫描示例：

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/scan_qos_pressure_tasksets.py \
  --workload mc_fairgen \
  --mc-fairgen-mode paper_learnable_headroom \
  --mc-fairgen-num-tasks 12 \
  --mc-fairgen-hi-ratio 0.5 \
  --mc-fairgen-period-source automotive \
  --mc-fairgen-u-hi-lo-min 0.20 \
  --mc-fairgen-u-hi-lo-max 0.35 \
  --mc-fairgen-u-hi-hi-min 0.45 \
  --mc-fairgen-u-hi-hi-max 0.70 \
  --mc-fairgen-u-lo-lo-min 0.25 \
  --mc-fairgen-u-lo-lo-max 0.45 \
  --mc-fairgen-hi-budget-rho-min 0.55 \
  --mc-fairgen-hi-budget-rho-max 0.75 \
  --mc-fairgen-lo-budget-rho-min 0.20 \
  --mc-fairgen-lo-budget-rho-max 0.40 \
  --mc-fairgen-hi-overrun-prob 0.08 \
  --mc-fairgen-lo-overrun-prob 0.12 \
  --mc-fairgen-hi-overrun-factor-min 1.02 \
  --mc-fairgen-hi-overrun-factor-max 1.25 \
  --mc-fairgen-lo-overrun-factor-min 1.02 \
  --mc-fairgen-lo-overrun-factor-max 1.25 \
  --require-schedulable \
  --seed-start 0 \
  --seed-end 500 \
  --eval-seeds 200:229 \
  --end-time 1000000 \
  --agent-period 50000 \
  --workers 1 \
  --output outputs/tasksets/qos_pressure_scan_v1.csv
```

开启 static sweep 示例：

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/scan_qos_pressure_tasksets.py \
  ...同上 mc_fairgen 参数... \
  --seed-start 0 \
  --seed-end 500 \
  --eval-seeds 200:229 \
  --end-time 1000000 \
  --agent-period 50000 \
  --enable-static-sweep \
  --static-sweep-stage quick \
  --sweep-inc-ratios 0,0.015,0.025,0.035 \
  --sweep-dec-ratios 0,0.010,0.015 \
  --stable-static-mode-deltas 0.05,0.10 \
  --stable-static-mode-abs-tolerance 0.0 \
  --min-static-sweep-reduction 0.05 \
  --min-stable-static-sweep-reduction 0.02 \
  --stable-static-delta 0.10 \
  --exclude-tradeoff-only \
  --output outputs/tasksets/qos_pressure_scan_medium_static.csv \
  --static-sweep-detail-output outputs/tasksets/qos_pressure_scan_medium_static_detail.csv
```

开启 single-task / single-sequence sweep 示例：

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/scan_qos_pressure_tasksets.py \
  ...同上 mc_fairgen 参数... \
  --seed-start 0 \
  --seed-end 200 \
  --eval-seeds 200:204 \
  --enable-static-sweep \
  --static-sweep-stage quick \
  --enable-single-task-sweep \
  --single-task-sweep-stage quick \
  --single-task-sweep-actions increase,decrease \
  --single-task-repeat-counts 1,2 \
  --single-task-top-k-by-headroom 6 \
  --single-task-sweep-detail-output outputs/tasksets/qos_pressure_single_task_detail.csv \
  --enable-single-sequence-sweep \
  --single-sequence-patterns inc_repeat,dec_repeat,inc_dec_pair,inc_dec_alternate \
  --single-sequence-lengths 2,4 \
  --single-sequence-top-k-tasks 4 \
  --single-sequence-sweep-detail-output outputs/tasksets/qos_pressure_sequence_detail.csv \
  --output outputs/tasksets/qos_pressure_scan_with_single_proxy.csv
```

新增关键输出字段（主 CSV）：
- `static_qos_best_*`：不约束 mode changes 的 static 最优解。
- `stable005_static_*`：约束 `mode_changes <= baseline * 1.05` 的最优解。
- `stable010_static_*`：约束 `mode_changes <= baseline * 1.10` 的最优解。
- `tradeoff_gap_005 / tradeoff_gap_010`：`static_qos_best` 与 `stable` 的改善差距。
- `tradeoff_only_flag_005 / tradeoff_only_flag_010`：trade-off-only 诊断标记。
- `improvement_type`：`stable005_improvable / stable010_improvable / tradeoff_only / weak_or_no_improvement / no_static_improvement`。
- `global_static_qos_best_* / global_stable005_static_* / global_stable010_static_*`：`static_*` 的全局静态别名字段。
- `single_static_qos_best_* / single_stable005_static_* / single_stable010_static_*`：single-task 代理最优结果。
- `sequence_static_qos_best_* / sequence_stable005_static_* / sequence_stable010_static_*`：single-sequence 代理最优结果。
- `single_improvement_type`：single 维度改进类型。
- `dqn_proxy_stable005_relative_lc_loss_reduction / dqn_proxy_stable010_relative_lc_loss_reduction`：single-task 与 sequence 的稳定改进上界代理。

### 3. manifest 筛选脚本

- 文件：`scripts/select_qos_pressure_tasksets.py`
- 作用：从扫描 CSV 按 bucket 和阈值筛选训练用 seed manifest，并输出拒绝原因 CSV。

示例（medium top20）：

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/select_qos_pressure_tasksets.py \
  --scan-csv outputs/tasksets/qos_pressure_scan_medium_static.csv \
  --bucket medium \
  --top-k 20 \
  --min-baseline-lc-service-loss 0.10 \
  --max-baseline-lc-service-loss 0.30 \
  --min-released-lo-jobs 100 \
  --min-cancelled-lo-jobs 10 \
  --min-mode-changes 1.0 \
  --min-static-sweep-reduction 0.05 \
  --min-stable-static-sweep-reduction 0.02 \
  --stable-static-delta 0.10 \
  --exclude-tradeoff-only \
  --require-single-action-improvement \
  --min-single-stable-sweep-reduction 0.005 \
  --single-stable-delta 0.10 \
  --allow-relaxed-single-stable \
  --prefer-single-action-stable \
  --exclude-single-tradeoff-only \
  --prefer-stable-static \
  --target-loss-center 0.20 \
  --output outputs/tasksets/mc_fairgen_qos_pressure_medium_top20.csv \
  --rejections-output outputs/tasksets/mc_fairgen_qos_pressure_medium_rejections.csv
```

新增筛选参数说明：
- `--min-stable-static-sweep-reduction`：要求稳定 static 改进至少达到给定阈值。
- `--stable-static-delta {0.05,0.10}`：指定主稳定约束字段使用 `stable005` 或 `stable010`。
- `--allow-relaxed-stable-static`：当主字段是 `stable005` 且不满足时，允许回退到 `stable010`。
- `--exclude-tradeoff-only`：剔除 `tradeoff_only_flag_* = true` 的样本。
- `--prefer-stable-static`：排序优先按 `stable005/stable010` 改进降序。
- `--require-single-action-improvement`：强制要求 single-action 稳定改进。
- `--min-single-stable-sweep-reduction`：single 稳定改进阈值。
- `--single-stable-delta {0.05,0.10}`：single 主稳定字段。
- `--allow-relaxed-single-stable`：single 主字段不满足时允许回退到 single010。
- `--prefer-single-action-stable`：排序优先按 single 稳定改进降序。
- `--exclude-single-tradeoff-only`：剔除 `single_improvement_type=single_tradeoff_only`。

### 4. 扫描汇总脚本

- 文件：`scripts/summarize_qos_pressure_scan.py`
- 作用：按 bucket 统计扫描分布并输出 summary CSV。

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/summarize_qos_pressure_scan.py \
  --scan-csv outputs/tasksets/qos_pressure_scan_v1.csv \
  --output outputs/tasksets/qos_pressure_scan_v1_summary.csv
```

新增汇总字段（按 bucket）：
- `stable005_static_found_valid_count`
- `stable010_static_found_valid_count`
- `stable005_static_relative_lc_loss_reduction_mean/median/max`
- `stable010_static_relative_lc_loss_reduction_mean/median/max`
- `static_qos_best_relative_lc_loss_reduction_mean`
- `tradeoff_gap_005_mean / tradeoff_gap_010_mean`
- `tradeoff_only_flag_005_count / tradeoff_only_flag_010_count`
- `improvement_type_counts`
- `single_stable005_static_relative_lc_loss_reduction_mean/max`
- `single_stable010_static_relative_lc_loss_reduction_mean/max`
- `single_stable005_found_count / single_stable010_found_count`
- `single_improvement_type_counts`

### 5. 训练命令生成脚本

- 文件：`scripts/make_train_single_v3_qos_pressure_commands.py`
- 作用：从 manifest 生成 single_v3 训练脚本，并在每次训练后自动调用 `scripts/select_qos_best_from_validation.py` 重选 best。

```bash
python -u scripts/make_train_single_v3_qos_pressure_commands.py \
  --manifest outputs/tasksets/mc_fairgen_qos_pressure_medium_top20.csv \
  --output-script /tmp/run_single_v3_qospressure_medium_top20_e120.sh \
  --output-dir-prefix outputs/train_single_v3_qospressure_medium \
  --episodes 120 \
  --end-time 1000000 \
  --agent-period 50000 \
  --validation-seeds 200:229 \
  --validation-end-time 1000000 \
  --qos-stable-mode-delta 0.05
```

生成的训练命令会显式包含 single_v3 对照关键参数：

```bash
--train-seed-mode per-episode
--validate-every 10
--validation-workers 1
--checkpoint 10
--save-best-by qos_stable
--qos-stable-mode-delta 0.05
--save-all-best-types
```

### 6. 对比命令生成脚本

- 文件：`scripts/make_compare_qos_pressure_commands.py`
- 作用：从 manifest 生成 `qos_stable/conservative_qos/qos_best` 三类 compare 命令脚本。

```bash
python -u scripts/make_compare_qos_pressure_commands.py \
  --manifest outputs/tasksets/mc_fairgen_qos_pressure_medium_top20.csv \
  --output-script /tmp/run_compare_qospressure_medium_top20_e120.sh \
  --run-dir-prefix outputs/train_single_v3_qospressure_medium \
  --episodes 120 \
  --output-prefix outputs/compare_single_v3_qospressure_medium \
  --qos-stable-mode-delta 0.05
```

### 13.6 QoS-Stable Reward 三版本接入说明

本次新增 3 个 reward mode（目录：`configs/reward_modes/`）：
- `qos_stable_v1_balanced`
- `qos_stable_v1_conservative`
- `qos_stable_v1_qoslean`

使用方法：训练时直接切换 `--reward-mode`。

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/train_dqn_amc.py \
  --workload mc_fairgen \
  --episodes 2 \
  --end-time 100000 \
  --agent-period 50000 \
  --reward-mode qos_stable_v1_balanced
```

新增 reward 配置 smoke test：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/smoke_test_reward_modes.py \
  --modes interval_v1,qos_stable_v1_balanced,qos_stable_v1_conservative,qos_stable_v1_qoslean
```

你会看到每个 mode 的 `step_reward` 计算结果；若变量名写错或 JSON 格式错误，脚本会返回非 0 退出码。

日志增强说明：
- `amc_py/rl/env.py` 的 `info` 新增 `mode_change_spike_penalty` 与 `mode_change_spike_penalty_value`。
- `scripts/train_dqn_amc.py` 的 `train_log.csv` 新增列 `mode_change_spike_penalty_value`。
- `scripts/train_dqn_amc.py` 的 `train_metrics.csv` 新增列 `reward_mode_change_spike_penalty_value_sum`。

命令生成脚本更新（`scripts/make_train_single_v3_qos_pressure_commands.py`）：
- 新增参数 `--reward-mode`（默认 `interval_v1`）。
- 生成命令会自动写入 `--reward-mode {值}`。
- 生成输出目录名会包含 reward mode，便于做多版本 ablation 对比。

示例：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/make_train_single_v3_qos_pressure_commands.py \
  --manifest outputs/tasksets/mc_fairgen_qos_pressure_strong_medium_top20_presweep_0_1200_lop020_lorho010_030.csv \
  --reward-mode qos_stable_v1_balanced \
  --output-dir-prefix outputs/train_single_v3_qosstable_strong_medium \
  --output-script /tmp/run_reward_test.sh
```

### 13.7 Controlled MC-FairGen + Stable Probe 使用说明

本次改动新增了 `controlled` 周期源、stable probe 脚本、probe-aware selector，并把扫描指标扩展为 `per-1M` 归一化字段。

1) `mc_fairgen` 新增周期源：
- `automotive`
- `controlled_sparse`
- `controlled_medium`
- `controlled_dense`

2) 生成/扫描时可直接使用：

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/generate_learnable_tasksets.py \
  --workload mc_fairgen \
  --mc-fairgen-period-source controlled_medium \
  --mc-fairgen-period-scale 100
```

3) `scan_taskset_headroom.py` 与 `scan_qos_pressure_tasksets.py` 新增输出字段：
- `baseline_total_events_per_1m`
- `baseline_mode_changes_per_1m`
- `baseline_lo_cancellations_per_1m`

并在 `end_time <= 0` 时直接报错：
- `ValueError("end_time must be positive for per-1M metrics")`

4) 新增 stable probe：`scripts/probe_stable_improvement_tasksets.py`

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/probe_stable_improvement_tasksets.py \
  --taskset-manifest outputs/tasksets/your_manifest.csv \
  --manifest-seed-column candidate_seed \
  --seeds 200:206 \
  --mc-fairgen-period-source controlled_medium \
  --end-time 3000000 \
  --output-summary outputs/taskset_probe/controlled_medium/stable_probe_summary.csv \
  --output-detail outputs/taskset_probe/controlled_medium/stable_probe_detail.csv
```

5) 新增 probe-aware selector：`scripts/select_probe_aware_tasksets.py`

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/select_probe_aware_tasksets.py \
  --headroom-summary outputs/taskset_slack_scan/controlled_medium/headroom_quick_summary.csv \
  --probe-summary outputs/taskset_probe/controlled_medium/stable_probe_summary.csv \
  --manifest-csv outputs/tasksets/controlled_medium/fullscan_0_1000.csv \
  --top-k 20 \
  --output-summary outputs/tasksets/controlled_medium/probe_aware_top20.csv \
  --output-manifest outputs/tasksets/controlled_medium/probe_aware_top20_manifest.csv \
  --output-rejections outputs/tasksets/controlled_medium/probe_aware_rejections.csv
```

6) `select_learnable_pressure_tasksets.py` 新增 `per-1M` 参数（默认启用 `--use-per-1m-metrics`）：
- `--min-events-per-1m`
- `--max-events-per-1m`
- `--min-lo-cancellations-per-1m`
- `--max-mode-changes-per-1m`

兼容旧参数，但会打印 warning。

7) 新增 smoke 命令生成器：`scripts/make_controlled_mc_fairgen_smoke_commands.py`

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python scripts/make_controlled_mc_fairgen_smoke_commands.py \
  --period-source controlled_medium \
  --out /tmp/run_controlled_medium_smoke.sh
bash /tmp/run_controlled_medium_smoke.sh
```

### 13.8 Task-level Cancellation Controllability 使用说明

本次新增了 task-level cancellation source 诊断能力，默认关闭；不加新参数时，`scan_taskset_headroom.py` 的旧输出口径保持不变。

1) `scan_taskset_headroom.py` 新增参数：
- `--enable-task-level-cancellation-diagnostic`：开启 task-level 聚合统计。
- `--task-level-output-dir`：可选；指定后输出每个 `seed/scale/eval_seed` 的 per-task 明细 CSV。

示例：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro env PYTHONPATH=. python -u scripts/scan_taskset_headroom.py \
  --workload mc_fairgen \
  --fixed-taskset-seeds 1484 \
  --budget-scales 1.0 \
  --seeds 200:202 \
  --enable-task-level-cancellation-diagnostic \
  --task-level-output-dir outputs/task_level_details \
  --output outputs/task_level_scan_summary.csv
```

新增 summary 字段（节选）：
- `task_level_top1_cancel_share_mean`
- `task_level_top2_cancel_share_mean`
- `task_level_top3_cancel_share_mean`
- `task_level_cancel_concentration_hhi_mean`
- `task_level_num_cancelled_lo_tasks_mean`
- `task_level_max_task_cancel_ratio_mean`
- `valid_increase_cancel_coverage_mean`
- `valid_decrease_cancel_coverage_mean`
- `valid_increase_top1_cancel_hit_rate`

per-task CSV 字段（节选）：
- `cancelled_jobs`
- `cancel_ratio_over_released`
- `cancel_share_of_total`
- `is_valid_increase_union`
- `is_valid_decrease_union`
- `valid_increase_seen_steps`
- `valid_decrease_seen_steps`
- `valid_increase_seen_fraction`
- `valid_decrease_seen_fraction`

2) 新增独立诊断脚本：`scripts/diagnose_task_level_controllability.py`

用于按指定 seed 列表直接输出 summary + task-level long table。

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro env PYTHONPATH=. python -u scripts/diagnose_task_level_controllability.py \
  --workload mc_fairgen \
  --fixed-taskset-seeds 1484,2429,2829,2221,1502,90,2574 \
  --seeds 200:210 \
  --output-summary outputs/task_level_diagnostic/summary.csv \
  --output-task-details outputs/task_level_diagnostic/details.csv
```

3) 新增 strong/weak 对照汇总脚本：`scripts/summarize_controllability_contrast.py`

将 task-level summary 与训练结果按 `candidate_seed` 关联，输出 joined CSV 和 markdown 报告。

```bash
cd /Users/x1ngchuan/Documents/AMC
python scripts/summarize_controllability_contrast.py \
  --summary-csv outputs/task_level_diagnostic/summary.csv \
  --result-csv outputs/controlled_medium_scale500/controlled_medium_scale500_top20_qos_stable_summary_with_qos.csv \
  --output-csv outputs/task_level_diagnostic/contrast_joined.csv \
  --output-md outputs/task_level_diagnostic/contrast_report.md
```

## 14. Decrease Controllability 诊断与 Static Probe 使用说明

本次修改新增了 decrease 侧可控性诊断与静态探测能力，核心目标是回答：
- 主要 cancellation source 是否可被 valid decrease 覆盖；
- `decrease_top_cancelled_lo` 与 `decrease_low_cancel_high_budget` 哪种更有效；
- decrease 改善是否伴随 mode-change tradeoff。

### 14.1 代码改动点

1. `amc_py/task_level_diagnostics.py`
- 修正 `task_level_top1/top2/top3_cancel_share` 与 `task_level_cancel_concentration_hhi` 的排序口径：统一按 `cancelled_jobs` 降序后的 LO 任务计算。
- 在 `compute_valid_action_cancel_coverage(...)` 中新增 decrease 侧字段：
  - `valid_decrease_top1_cancel_hit`
  - `valid_decrease_top2_cancel_hit_count`
  - `valid_decrease_top3_cancel_hit_count`
  - `valid_decrease_cancelled_task_count`
  - `valid_decrease_cancelled_task_share`
  - `valid_decrease_top_cancel_task_name`
  - `valid_decrease_top_cancel_task_index`

2. `scripts/diagnose_task_level_controllability.py`
- summary 新增 decrease 侧聚合字段：
  - `valid_decrease_top1_cancel_hit_rate`
  - `valid_decrease_top2_cancel_hit_count_mean`
  - `valid_decrease_top3_cancel_hit_count_mean`
  - `valid_decrease_cancelled_task_share_mean`
- 新增 `decrease_source_score`（诊断覆盖性指标，不等价于 decrease 一定有益）。

3. 新增脚本 `scripts/probe_static_decrease_controllability.py`
- 静态探测三类 probe：
  - `decrease_top_cancelled_lo`
  - `decrease_low_cancel_high_budget`
  - `increase_valid_cancel_source`（对照）
- 输出 detail/summary 两份 CSV，字段与计划文档保持一致。
- 支持：当 `--candidate-seeds` 非空时，直接使用该列表，不要求 manifest 必须包含这些 seed。

4. 新增脚本 `scripts/summarize_decrease_probe_contrast.py`
- 将 controllability summary 与 decrease probe summary 合并。
- 输出 joined CSV 与 Markdown 报告，按 `strong/medium/weak/opportunity_or_lightprobe_weak` 分组汇总。

### 14.2 运行 diagnose（含 decrease 侧指标）

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/diagnose_task_level_controllability.py \
  --workload mc_fairgen \
  --fixed-taskset-seeds 1484,2429,2829 \
  --seeds 200:210 \
  --end-time 1000000 \
  --output-summary outputs/taskset_slack_scan/controlled_medium_scale500/controllability_summary_e1m_s200_210.csv \
  --output-task-details outputs/taskset_slack_scan/controlled_medium_scale500/controllability_task_details_e1m_s200_210.csv
```

### 14.3 运行 static decrease probe

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/probe_static_decrease_controllability.py \
  --workload mc_fairgen \
  --taskset-manifest outputs/tasksets/controlled_medium_scale500/final_top20_medium_0_3000.csv \
  --manifest-seed-column candidate_seed \
  --candidate-seeds 1484,2429,2829,2221,1502,90,2574,57,185,2784,395,2505,367,1563,343,559 \
  --seeds 200:210 \
  --end-time 1000000 \
  --budget-increase-ratio 0.025 \
  --budget-decrease-ratio 0.015 \
  --budget-floor-ratio 0.9 \
  --repeat-counts 1,2,3 \
  --top-k-cancelled 3 \
  --top-k-low-cancel-high-budget 3 \
  --top-k-increase-reference 3 \
  --stable-mode-delta 0.05 \
  --mc-fairgen-mode paper_learnable_headroom \
  --mc-fairgen-num-tasks 12 \
  --mc-fairgen-hi-ratio 0.5 \
  --mc-fairgen-period-source controlled_medium \
  --mc-fairgen-period-scale 500 \
  --mc-fairgen-u-hi-lo-min 0.20 \
  --mc-fairgen-u-hi-lo-max 0.35 \
  --mc-fairgen-u-hi-hi-min 0.45 \
  --mc-fairgen-u-hi-hi-max 0.70 \
  --mc-fairgen-u-lo-lo-min 0.25 \
  --mc-fairgen-u-lo-lo-max 0.45 \
  --mc-fairgen-hi-budget-rho-min 0.55 \
  --mc-fairgen-hi-budget-rho-max 0.75 \
  --mc-fairgen-lo-budget-rho-min 0.20 \
  --mc-fairgen-lo-budget-rho-max 0.40 \
  --mc-fairgen-hi-overrun-prob 0.08 \
  --mc-fairgen-lo-overrun-prob 0.12 \
  --mc-fairgen-hi-overrun-factor-min 1.02 \
  --mc-fairgen-hi-overrun-factor-max 1.25 \
  --mc-fairgen-lo-overrun-factor-min 1.02 \
  --mc-fairgen-lo-overrun-factor-max 1.25 \
  --output-summary outputs/taskset_slack_scan/controlled_medium_scale500/decrease_probe/static_decrease_probe_summary_e1m_s200_210.csv \
  --output-detail outputs/taskset_slack_scan/controlled_medium_scale500/decrease_probe/static_decrease_probe_detail_e1m_s200_210.csv
```

### 14.4 运行对照汇总脚本

```bash
cd /Users/x1ngchuan/Documents/AMC
PYTHONPATH=. python -u scripts/summarize_decrease_probe_contrast.py \
  --task-level-summary outputs/taskset_slack_scan/controlled_medium_scale500/controllability_summary_e1m_s200_210.csv \
  --decrease-probe-summary outputs/taskset_slack_scan/controlled_medium_scale500/decrease_probe/static_decrease_probe_summary_e1m_s200_210.csv \
  --output-csv outputs/taskset_slack_scan/controlled_medium_scale500/decrease_probe/decrease_probe_contrast_joined.csv \
  --output-md outputs/taskset_slack_scan/controlled_medium_scale500/decrease_probe/decrease_probe_contrast_report.md
```

### 14.5 输出文件说明

1. `static_decrease_probe_detail_*.csv`
- 每行对应一个 `candidate_seed + probe_type + target_task + repeat_count`。
- 关键字段：`lo_cancellation_rel_reduction`、`lc_service_loss_abs_reduction`、`mode_change_delta_ratio`、`stable_candidate`、`tradeoff_risk`。

2. `static_decrease_probe_summary_*.csv`
- 每个 candidate seed 一行。
- 包含：
  - `best_decrease_*`
  - `best_decrease_top_cancelled_lo_*`
  - `best_decrease_low_cancel_high_budget_*`
  - `best_increase_reference_*`
  - `decrease_probe_has_stable_positive`
  - `decrease_probe_best_beats_increase_reference`

3. `decrease_probe_contrast_report.md`
- 按 manual group 输出组级均值与关键问答，直接用于 strong/weak/opportunity weak 对照分析。

## 13.9 Task-Level Controllability Selector 使用说明

本节对应 `controllability_selector_codex_plan.md` 的必做实现，目标是将任务集筛选从 pressure 指标升级为 task-level controllability 指标。

### 1) 诊断脚本字段补齐（`scripts/diagnose_task_level_controllability.py`）

本次修改后，诊断脚本支持两种指定候选任务集的方式：

- 显式传入 `--fixed-taskset-seeds`；
- 仅传 `--taskset-manifest`（脚本会自动读取 `--manifest-seed-column` 对应列作为 seed 列表）。

输出的 `summary` 关键字段现在包含：

- `baseline_total_events_per_1m`
- `baseline_mode_changes_per_1m`
- `baseline_lo_cancellations_per_1m`
- `baseline_lo_cancellation_ratio_total`
- `task_level_top1_cancel_share_mean`
- `task_level_top2_cancel_share_mean`
- `task_level_top3_cancel_share_mean`
- `task_level_cancel_concentration_hhi_mean`
- `cancel_concentration_hhi_mean`（兼容别名）
- `valid_increase_cancel_coverage_mean`
- `valid_increase_top1_cancel_hit_rate`
- `valid_increase_top2_cancel_hit_count_mean`
- `valid_increase_top3_cancel_hit_count_mean`
- `valid_decrease_cancel_coverage_mean`
- `valid_decrease_top1_cancel_hit_rate`
- `valid_decrease_top2_cancel_hit_count_mean`
- `valid_decrease_top3_cancel_hit_count_mean`

`task details` 输出继续包含并保留以下字段：

- `candidate_seed`
- `eval_seed`
- `task_index`
- `task_name`
- `criticality`
- `period`
- `released_jobs`
- `cancelled_jobs`
- `cancel_ratio_over_released`
- `cancel_share_of_total`
- `is_valid_increase_union`
- `is_valid_decrease_union`
- `valid_increase_seen_fraction`
- `valid_decrease_seen_fraction`

### 2) 新增 selector 脚本（`scripts/select_controllable_tasksets.py`）

脚本输入：

- `--summary-csv`：原 headroom summary
- `--manifest-csv`：原 taskset manifest
- `--controllability-summary-csv`：task-level diagnostics summary

脚本输出：

- `--output-summary`：controllability topK 的筛选摘要
- `--output-manifest`：可直接给 `train_dqn_amc.py` 使用的 manifest（保持原列顺序）
- `--output-rejections`：未入选种子及拒绝原因

默认阈值（与计划一致）：

- `--top-k 10`
- `--min-valid-increase 3`
- `--min-valid-decrease 3`
- `--min-baseline-lo-cancellations-per-1m 22`
- `--max-baseline-mode-changes-per-1m 30`
- `--min-baseline-lo-cancellation-ratio 0.45`
- `--max-baseline-lo-cancellation-ratio 0.75`
- `--min-valid-increase-cancel-coverage 0.35`
- `--min-valid-increase-top2-hit-count 1.0`
- `--min-valid-increase-top3-hit-count 1.5`
- `--max-top1-cancel-share 0.75`
- `--max-deadline-misses 0`

支持 dry-run：

- `--dry-run`：不写文件，只打印通过样本 topN、拒绝原因计数、关键指标 describe
- `--print-top-n`：dry-run 打印行数（默认 `30`）

### 3) 示例命令

```bash
cd /Users/x1ngchuan/Documents/AMC

python scripts/select_controllable_tasksets.py \
  --summary-csv outputs/taskset_slack_scan/controlled_medium_scale500/top30_medium_summary_0_3000_e5m_s200_210.csv \
  --manifest-csv outputs/tasksets/controlled_medium_scale500/learnable_top30_0_3000_quick_e3m_s200_206.csv \
  --controllability-summary-csv outputs/taskset_slack_scan/controlled_medium_scale500/controllability/controllability_summary_top30_e1m_s200_210.csv \
  --top-k 10 \
  --min-valid-increase 3 \
  --min-valid-decrease 3 \
  --min-baseline-lo-cancellations-per-1m 22 \
  --max-baseline-mode-changes-per-1m 30 \
  --min-baseline-lo-cancellation-ratio 0.45 \
  --max-baseline-lo-cancellation-ratio 0.75 \
  --min-valid-increase-cancel-coverage 0.35 \
  --min-valid-increase-top2-hit-count 1.0 \
  --min-valid-increase-top3-hit-count 1.5 \
  --max-top1-cancel-share 0.75 \
  --output-summary outputs/taskset_slack_scan/controlled_medium_scale500/controllability/controllable_top10_summary_0_3000.csv \
  --output-manifest outputs/tasksets/controlled_medium_scale500/controllable_top10_0_3000.csv \
  --output-rejections outputs/taskset_slack_scan/controlled_medium_scale500/controllability/controllable_top10_rejections_0_3000.csv
```

dry-run 示例：

```bash
cd /Users/x1ngchuan/Documents/AMC

python scripts/select_controllable_tasksets.py \
  --summary-csv outputs/taskset_slack_scan/controlled_medium_scale500/top30_medium_summary_0_3000_e5m_s200_210.csv \
  --manifest-csv outputs/tasksets/controlled_medium_scale500/learnable_top30_0_3000_quick_e3m_s200_206.csv \
  --controllability-summary-csv outputs/taskset_slack_scan/controlled_medium_scale500/controllability/controllability_summary_top30_e1m_s200_210.csv \
  --dry-run --print-top-n 30
```
