# reward 配置文件说明

- `interval_v1`：当前历史主线版本。
- `interval_v2`：在 `interval_v1` 基础上增加 `budget_change_penalty` 与 `budget_drift_penalty`。

## QoS-Stable reward modes

- `qos_stable_v1_balanced`：主推荐版本，平衡 QoS 改善与 mode stability。
- `qos_stable_v1_conservative`：更强 mode/budget 稳定约束，适合 `balanced` 仍过于激进时使用。
- `qos_stable_v1_qoslean`：更偏 LC QoS，适合 `balanced` 过于保守时使用。

训练侧说明：
- 这三种 reward 不直接写入 `baseline_mode_changes * 1.05` 这类 episode 级硬约束；
- 训练时通过 `mode_change_per_job`、`mode_change_spike_penalty`、`budget_change_penalty`、`budget_drift_penalty` 提供近似信号；
- 最终是否满足 QoS-Stable 约束，仍由 validation 与 `model_selection.py` 的 `qos_stable` 规则决定。
