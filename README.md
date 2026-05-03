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

输出文件：

- `outputs/dqn_amc/train_log.csv`
- `outputs/dqn_amc/model_final.pt`
- `outputs/dqn_amc/config.json`
- `outputs/dqn_amc/checkpoints/model_episode_XXXX.pt`（当 `--checkpoint > 0` 时）

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
