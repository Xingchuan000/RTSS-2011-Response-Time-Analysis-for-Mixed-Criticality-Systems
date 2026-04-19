# 任务集生成说明（修订版）

## 1. 核心参数语义

当前生成器以 `GenerationConfig` 为唯一参数入口，关键字段如下：

1. `time_scale`：
   把论文单位周期（如 `10~1000`）统一缩放到内部 tick（如 `*100`），降低整数离散化误差。
2. `deadline_mode`：
   截止期策略枚举，支持 `implicit`、`ratio_uniform`、`arbitrary_paper`。
3. `lo_hi_budget_policy`：
   LO 任务是否拥有独立分析预算 `c_hi`，支持 `equal_lo` 与 `scaled_by_cf`。
4. `criticality_assignment`：
   HI 任务分配方式，支持 `fixed_count` 与 `bernoulli`。

## 2. deadline_mode 三种模式

1. `implicit`：`D = T`。
   该模式下允许出现 `C(HI) > D`，这类任务是“可生成但可能不可调度”的样本，
   是否可调度交给后续分析器判断，不在生成阶段提前剔除。
2. `ratio_uniform`：`D` 在 `[max(ceil(r*T), c_required), T]` 均匀采样。
3. `arbitrary_paper`：
   - HI 任务：`D ~ U[c_hi, T]`
   - LO 任务：`D ~ U[c_lo, T]`

其中 `c_required` 对 HI 任务取 `c_hi`，对 LO 任务取 `c_lo`。
因此 `ratio_uniform / arbitrary_paper` 会在采样阶段显式保证预算下界，
而 `implicit` 模式仅固定 `D=T`。

## 2.1 生成器与分析器的职责边界

- 生成器负责：参数合法性、采样区间合法性、任务结构合法性（如 `deadline >= c_lo`）。
- 生成器不负责：提前过滤所有不可调度任务。
- 特别地，在 `implicit` 模式下，不以 `c_hi <= deadline` 作为硬约束。

## 3. LO 任务预算策略

1. `equal_lo`：`c_hi = c_lo`（兼容旧语义）。
2. `scaled_by_cf`：`c_hi = max(c_lo, round(c_lo * cf))`。

`scaled_by_cf` 可以让 `SMC` 与 `SMC-NO` 在统计上自然分离，更接近论文比较目的。

## 4. time_scale 的作用

当总利用率较低时，如果不放大时间粒度，`c_lo = round(util * period)` 可能被 `max(1, ...)` 明显抬高。

引入 `time_scale` 后，内部 `period` 更大，`c_lo/period` 分辨率更细，`actual_total_util_lo` 更接近目标利用率。

建议默认值：

- `paper`：`time_scale = 100`
- `fast`：`time_scale = 10`

## 5. 配置文件

当前配置文件：

- `/Users/x1ngchuan/Documents/AMC/configs/generator_fast.yaml`
- `/Users/x1ngchuan/Documents/AMC/configs/generator_paper.yaml`

两个文件都包含：

- `time_scale`
- `deadline_mode`
- `lo_hi_budget_policy`

## 6. Python 调用示例

```python
from amc_py.generator import make_generation_config, generate_taskset

cfg = make_generation_config("paper")
taskset = generate_taskset(
    num_tasks=cfg.num_tasks,
    total_util=cfg.total_util,
    min_period=cfg.min_period,
    max_period=cfg.max_period,
    time_scale=cfg.time_scale,
    cf=cfg.cf,
    cp=cfg.cp,
    seed=2026,
    deadline_mode=cfg.deadline_mode,
    deadline_ratio_min=cfg.deadline_ratio_min,
    criticality_assignment=cfg.criticality_assignment,
    lo_hi_budget_policy=cfg.lo_hi_budget_policy,
)
```

## 7. 生成器校验脚本

运行：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python scripts/validate_generator.py --mode paper --num-tasksets 200 --seed 2026
```

输出目录（默认）：

- `reports/generator_validation/taskset_stats.csv`
- `reports/generator_validation/task_stats.csv`
- `reports/generator_validation/generator_validation_plots.png`
- `reports/generator_validation/generator_validation_report.md`
