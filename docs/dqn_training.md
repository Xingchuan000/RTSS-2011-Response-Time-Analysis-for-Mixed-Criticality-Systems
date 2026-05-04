# RTSS11 DQN 训练流程（Phase 8）

本文固定 RTSS11 的可复现实验流程，避免手工拼接命令导致训练口径漂移。

## 1. 训练（u=0.65）

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro env PYTHONPATH=. python scripts/train_dqn_amc.py \
  --workload rtss11 \
  --total-util 0.65 \
  --num-tasks 20 \
  --cf 2.0 \
  --cp 0.5 \
  --require-schedulable \
  --episodes 500 \
  --end-time 50000 \
  --agent-period 1000 \
  --seed 0 \
  --train-seed-mode per-episode \
  --batch-size 64 \
  --replay-capacity 10000 \
  --min-replay-size 500 \
  --hidden-layers 128,128 \
  --epsilon-decay-steps 5000 \
  --validation-seeds 100:129 \
  --validate-every 50 \
  --reward-mode event_delta_no_job_start \
  --action-space pair \
  --budget-increase-ratio 0.05 \
  --budget-decrease-ratio 0.05 \
  --include-explicit-noop \
  --output-dir outputs/dqn_rtss11/u065_multiseed_pair_v1
```

训练完成后，输出目录最少应包含：

- `train_metrics.csv`
- `validation_metrics.csv`
- `model_best.pt`
- `model_final.pt`

## 2. 评估 best 模型

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro env PYTHONPATH=. python scripts/evaluate_dqn_amc.py \
  --workload rtss11 \
  --total-util 0.65 \
  --num-tasks 20 \
  --cf 2.0 \
  --cp 0.5 \
  --require-schedulable \
  --model outputs/dqn_rtss11/u065_multiseed_pair_v1/model_best.pt \
  --seeds 200:299 \
  --end-time 50000 \
  --agent-period 1000 \
  --reward-mode event_delta_no_job_start \
  --action-space pair \
  --budget-increase-ratio 0.05 \
  --budget-decrease-ratio 0.05 \
  --include-explicit-noop \
  --output outputs/dqn_rtss11/u065_multiseed_pair_v1/test_eval.csv
```

## 3. 汇总评估结果

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro env PYTHONPATH=. python scripts/summarize_dqn_rtss11_results.py \
  --input outputs/dqn_rtss11/u065_multiseed_pair_v1/test_eval.csv \
  --output outputs/dqn_rtss11/u065_multiseed_pair_v1/test_summary.csv
```

## 4. 对比多次训练 run

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n amc-repro env PYTHONPATH=. python scripts/compare_dqn_training_runs.py \
  --runs outputs/dqn_rtss11/u065_multiseed_pair_v1,outputs/dqn_rtss11/u065_multiseed_v1 \
  --output outputs/dqn_rtss11/training_comparison.csv
```

输出字段：

- `run,total_util,reward_mode,action_space,episodes,end_time,train_seed_mode`
- `best_episode,best_deadline_misses_sum,best_mode_changes_mean,best_lo_cancellations_mean`
- `baseline_mode_changes_mean,baseline_lo_cancellations_mean,dqn_mode_delta,dqn_lo_cancel_delta`
- `valid_action_count_mean,masked_action_count_mean,no_safe_action_steps`

