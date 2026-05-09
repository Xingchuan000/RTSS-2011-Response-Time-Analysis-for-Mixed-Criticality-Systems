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

在接入 DQN 之前，当前仓库已经提供可直接用于训练循环的运行时环境封装：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q tests/test_rl_env.py
conda run -n amc-repro python scripts/run_pre_dqn_runtime_baselines.py --end-time 100 --seed 0
```

相关文档：`docs/pre_dqn_runtime_interface.md`。

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

### 12.0 v11 observation（per-task 10d + global 8d）配置

当前训练/评估 CLI 已支持通过参数启用：

- `--observation-mode v10_basic`（默认，旧模式，`state_dim = 2 * n_tasks`）
- `--observation-mode v11_full_10d`（新模式，`state_dim = 10 * n_tasks + 8`）

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

日志与产物中已记录：

- step info / 训练日志：`observation_mode`、`state_dim`
- validation 指标：`observation_mode`、`state_dim_mean`
- 评估 CSV：`observation_mode`、`state_dim`
- 训练配置快照 `config.json`：`observation_mode` 与 `feature_config`

### 12.0.1 阶段 8 测试（v11 observation）

新增测试文件：

- `tests/test_v11_observation.py`

覆盖项：

1. `v10_basic` 长度保持 `2 * n_tasks`
2. `v11_full_10d` 长度为 `10 * n_tasks + 8`
3. `v11_full_10d` 所有特征值在 `[0, 1]`
4. `step` 后 `v11_full_10d` 维度保持正确
5. `feature_state` 在 step 后存在且任务键集合保持一致
6. event window 长度不超过 `event_window`

运行命令：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q tests/test_v11_observation.py
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

常用示例：

```bash
cd /Users/x1ngchuan/Documents/AMC
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro python scripts/train_dqn_amc.py \
  --save-best-by pareto_relative_score \
  --relative-score-alpha 1.0
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

### 12.8 automotive workload 用法

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
