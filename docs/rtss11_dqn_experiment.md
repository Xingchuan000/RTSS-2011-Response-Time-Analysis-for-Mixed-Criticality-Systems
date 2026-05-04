# RTSS2011 + DQN 实验说明

## 1. 第一轮正式实验配置

- `total_util ∈ {0.55, 0.65, 0.75}`
- `num_tasks = 20`
- `cf = 2.0`
- `cp = 0.5`
- `train_seeds ∈ {0, 1, 2}`
- `test_seeds = 100~129`
- `episodes = 50` 或 `100`
- `end_time = 10000` 或 `50000`
- `agent_period = 1000`

## 2. 每组实验输出

- `outputs/dqn_rtss11/u055_seed0/model.pt`
- `outputs/dqn_rtss11/u055_seed0/train_log.csv`
- `outputs/dqn_rtss11/u055_eval.csv`
- `outputs/dqn_rtss11/u055_summary.csv`

- `outputs/dqn_rtss11/u065_seed0/model.pt`
- `outputs/dqn_rtss11/u065_seed0/train_log.csv`
- `outputs/dqn_rtss11/u065_eval.csv`
- `outputs/dqn_rtss11/u065_summary.csv`

- `outputs/dqn_rtss11/u075_seed0/model.pt`
- `outputs/dqn_rtss11/u075_seed0/train_log.csv`
- `outputs/dqn_rtss11/u075_eval.csv`
- `outputs/dqn_rtss11/u075_summary.csv`

## 3. 第一轮结果判断标准

至少检查以下指标：

1. `deadline_misses_sum == 0`
2. `dqn_agent` 的 `mode_changes_mean` 是否低于 `amc_plus_baseline`
3. `dqn_agent` 的 `lo_cancellations_mean` 是否低于 `amc_plus_baseline`
4. `dqn_agent` 是否优于 `random_agent`
5. `rejection_rate` 是否过高

若 `rejection_rate > 0.8`，说明大部分动作被 safety mask 拒绝，即便总体指标有改善，也应谨慎解释结论。

## 4. 需要避免的实现错误

1. 把 `AMC-rtb` 当作运行时调度器：`AMC-rtb` 是离线分析方法，运行时 baseline 应使用 `AMC/AMC+`。
2. 训练中混入离线不可调度任务集：会让结论失去可解释性。
3. 缺少实际执行时间分布：RTSS2011 taskset 不包含 job 实际执行时间，必须显式引入 scenario/sampler。
4. 只报告平均改善：还应报告可调度筛选通过率、样本数量、`deadline_misses`、动作接受/拒绝比例及 `random/noop` 对照。
5. 忽略 seed 管理：训练 seed、taskset seed、scenario seed 必须分开记录。

## 5. 端到端 smoke 测试

仓库内端到端最小闭环测试：

- `tests/test_rtss11_dqn_end_to_end_smoke.py`

测试流程：

1. 生成 RTSS2011 taskset
2. AMC-rtb 可调度筛选
3. 短训练
4. 短评估
5. 汇总 CSV

可通过以下命令运行：

```bash
cd /Users/x1ngchuan/Documents/AMC
conda run -n amc-repro python -m pytest -q tests/test_rtss11_dqn_end_to_end_smoke.py
```

## 6. 最终验收清单

- [ ] `pytest` 全部通过
- [ ] RTSS2011 taskset 可按 seed 稳定复现
- [ ] 可调度筛选函数可返回 AMC-rtb 可调度任务集
- [ ] execution scenario 能产生预算超限事件
- [ ] HI task actual execution time 永不超过 `C_HI`
- [ ] `train_dqn_amc.py` 支持 `--workload rtss11`
- [ ] `evaluate_dqn_amc.py` 支持 `--workload rtss11`
- [ ] 训练可生成 checkpoint
- [ ] 评估可生成完整 CSV
- [ ] 汇总脚本可生成 summary CSV
- [ ] `noop_agent` 与 `amc_plus_baseline` 结果一致或近似一致
- [ ] safety check 开启时 `deadline_misses == 0`
- [ ] DQN-agent 与 AMC+ baseline 在同一批 seed 对比
- [ ] 输出 `accepted/rejected/noop` 动作统计
- [ ] 第一轮小规模正式实验可复现
