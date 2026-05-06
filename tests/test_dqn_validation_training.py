"""DQN 训练内 validation 流程测试。"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

from amc_py.dqn import DqnBudgetAgent, DqnConfig, build_small_stress_experiment_config, build_env_from_experiment_config
from amc_py.runtime_models import RuntimeSemantics
from scripts.train_dqn_amc import _run_validation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_writes_validation_metrics_and_best_checkpoint(tmp_path: Path) -> None:
    """开启 validate-every 后应产出 validation CSV 与 best/final 模型。"""

    output_dir = tmp_path / "dqn_validation"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "rtss11",
            "--total-util",
            "0.55",
            "--num-tasks",
            "20",
            "--cf",
            "2.0",
            "--cp",
            "0.5",
            "--require-schedulable",
            "--episodes",
            "3",
            "--end-time",
            "2000",
            "--agent-period",
            "1000",
            "--seed",
            "0",
            "--validation-seeds",
            "100,101",
            "--validate-every",
            "1",
            "--validation-end-time",
            "2000",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    validation_metrics_path = output_dir / "validation_metrics.csv"
    validation_unified_summary_path = output_dir / "validation_unified_summary.csv"
    best_model_path = output_dir / "model_best.pt"
    final_model_path = output_dir / "model_final.pt"
    assert validation_metrics_path.exists()
    assert validation_unified_summary_path.exists()
    assert best_model_path.exists()
    assert final_model_path.exists()

    with validation_metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 3
    assert "deadline_misses_sum" in rows[0]
    with validation_unified_summary_path.open("r", encoding="utf-8", newline="") as f:
        unified_rows = list(csv.DictReader(f))
    assert unified_rows
    assert "noop_action_rate_mean" in unified_rows[0]
    assert "noop_q_rank_mean" in rows[0]
    assert "noop_q_rank_mean" in unified_rows[0]
    assert "noop_q_sample_count" in unified_rows[0]


def test_validation_metrics_match_between_serial_and_parallel_workers() -> None:
    """固定 agent 与 validation seeds 时，串行/并行 validation 聚合结果应一致。"""

    experiment_config = build_small_stress_experiment_config()
    initial_env = build_env_from_experiment_config(
        experiment_config,
        seed=0,
        end_time=100,
        agent_period=1000,
        semantics=RuntimeSemantics.AMC_PLUS,
        reward_mode="mendes",
        action_space="triple",
        budget_increase_ratio=0.10,
        budget_decrease_ratio=0.05,
        include_explicit_noop=False,
        budget_floor_ratio=0.0,
        forbid_decreasing_hi_budgets=False,
        mask_detail_mode="minimal",
    )
    initial_obs = initial_env.reset(seed=0)
    agent = DqnBudgetAgent(
        observation_dim=len(initial_obs.state_vector),
        action_dim=initial_env.action_space_size,
        config=DqnConfig(
            gamma=0.99,
            learning_rate=1e-3,
            replay_capacity=32,
            min_replay_size=2,
            batch_size=2,
            target_update_freq=2,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=1,
            hidden_layers=(16, 16),
            seed=7,
        ),
    )

    serial_metrics, _, _ = _run_validation(
        agent=agent,
        experiment_config=experiment_config,
        validation_seeds=[100, 101],
        validation_end_time=100,
        agent_period=1000,
        reward_mode="mendes",
        action_space="triple",
        budget_increase_ratio=0.10,
        budget_decrease_ratio=0.05,
        include_explicit_noop=False,
        budget_floor_ratio=0.0,
        forbid_decreasing_hi_budgets=False,
        mask_detail_mode="minimal",
        validation_workers=1,
    )
    parallel_metrics, _, _ = _run_validation(
        agent=agent,
        experiment_config=experiment_config,
        validation_seeds=[100, 101],
        validation_end_time=100,
        agent_period=1000,
        reward_mode="mendes",
        action_space="triple",
        budget_increase_ratio=0.10,
        budget_decrease_ratio=0.05,
        include_explicit_noop=False,
        budget_floor_ratio=0.0,
        forbid_decreasing_hi_budgets=False,
        mask_detail_mode="minimal",
        validation_workers=2,
    )

    comparable_fields = [
        "deadline_misses_sum",
        "mode_changes_mean",
        "lo_cancellations_mean",
        "baseline_mode_changes_mean",
        "baseline_lo_cancellations_mean",
        "dqn_mode_changes_delta_mean",
        "dqn_lo_cancellations_delta_mean",
        "accepted_actions_mean",
        "rejected_actions_mean",
        "step_count_mean",
        "selected_action_count_mean",
        "noop_actions_mean",
        "explicit_noop_actions_mean",
        "noop_action_rate_mean",
        "explicit_noop_action_rate_mean",
        "accepted_action_rate_mean",
        "rejected_action_rate_mean",
        "valid_action_count_mean",
        "masked_action_count_mean",
        "no_safe_action_steps_mean",
        "reward_mean",
    ]
    for field in comparable_fields:
        assert parallel_metrics[field] == pytest.approx(serial_metrics[field])
