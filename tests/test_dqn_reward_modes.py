"""DQN reward mode 测试。"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_single_hi_overrun_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tasks() -> list[Task]:
    """构造可触发模式切换的任务集。"""

    return [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l1", 12, 12, 2, 2, Criticality.LO),
        Task("l2", 15, 15, 2, 2, Criticality.LO),
    ]


def test_mendes_reward_mode_keeps_original_component_semantics() -> None:
    """mendes 模式下，总奖励应等于组件和。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks(),
        scenario=make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi"),
        runtime_config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        reward_mode="mendes",
    )
    env.reset(seed=0)
    step = env.step(None)
    total = float(step.info["step_reward_total"])
    comp = (
        float(step.info["step_reward_job_start"])
        + float(step.info["step_reward_lo_overrun"])
        + float(step.info["step_reward_hi_overrun"])
        + float(step.info["step_reward_mode_change"])
        + float(step.info["step_reward_lo_cancellation"])
        + float(step.info["step_reward_deadline_miss"])
    )
    assert total == comp


def test_event_delta_mode_penalizes_mode_changes() -> None:
    """event_delta 模式下发生 mode change 时应产生负向奖励。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks(),
        scenario=make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi"),
        runtime_config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        reward_mode="event_delta",
    )
    env.reset(seed=0)
    step = env.step(None)
    if int(step.info["mode_changes"]) > 0:
        assert float(step.info["step_reward_mode_change"]) < 0.0


def test_event_delta_no_job_start_excludes_job_start_positive_reward() -> None:
    """event_delta_no_job_start 不应包含 job_start 正奖励。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks(),
        scenario=make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi"),
        runtime_config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        reward_mode="event_delta_no_job_start",
    )
    env.reset(seed=0)
    step = env.step(None)
    assert float(step.info["step_reward_job_start"]) == 0.0


def test_train_metrics_contains_reward_component_columns(tmp_path: Path) -> None:
    """训练输出的 train_metrics.csv 应包含 reward 组件汇总字段。"""

    output_dir = tmp_path / "dqn_reward_mode"
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
            "2",
            "--end-time",
            "2000",
            "--agent-period",
            "1000",
            "--seed",
            "0",
            "--reward-mode",
            "event_delta_no_job_start",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    with (output_dir / "train_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "reward_job_start_sum" in rows[0]
    assert "reward_lo_overrun_sum" in rows[0]
    assert "reward_hi_overrun_sum" in rows[0]
    assert "reward_mode_change_sum" in rows[0]
    assert "reward_lo_cancellation_sum" in rows[0]
    assert "reward_deadline_miss_sum" in rows[0]
