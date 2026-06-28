"""Level 4 reason-split reward 训练环境测试。"""

from __future__ import annotations

import json
from pathlib import Path

from amc_py.models import Criticality, Task
from amc_py.rl import reward_config as reward_config_module
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import (
    LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH,
    LO_LOSS_BUDGET_CANCELLATION,
    LoJobLossEvent,
    RuntimeConfig,
    RuntimeSemantics,
    SimulationResult,
)
from amc_py.runtime_scenarios import make_single_hi_overrun_scenario


def _tasks() -> list[Task]:
    """构造一个最小可用任务集。"""

    return [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l1", 12, 12, 2, 2, Criticality.LO),
    ]


class _StubState:
    """提供 env.step 所需的最小 active_jobs 容器。"""

    active_jobs: list[object] = []


class _StubEngine:
    """把 env.step 需要的 engine 接口最小化出来。"""

    def __init__(self, *, runtime_budgets: object, result: SimulationResult, target_time: int) -> None:
        self.runtime_budgets = runtime_budgets
        self._result = result
        self.current_time = 0
        self._target_time = target_time
        self.state = _StubState()

    def run_until(self, target_time: int, include_boundary: bool = True) -> None:  # noqa: ARG002
        """测试里不做真实推进，只把时间推进到目标点。"""

        self.current_time = min(target_time, self._target_time)

    def finish(self) -> SimulationResult:
        """直接返回预先构造好的结果对象。"""

        return self._result

    def apply_budget_updates(self, updates: dict[str, int]) -> None:  # noqa: ARG002
        """本组测试只走 noop 路径，不会真的改预算。"""


def _build_env_with_stubbed_result(result: SimulationResult, *, reward_mode: str = "mendes") -> AmcBudgetEnv:
    """构造环境，并把底层 engine 替换成固定结果的 stub。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks(),
        scenario=make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi"),
        runtime_config=RuntimeConfig(
            end_time=10,
            semantics=RuntimeSemantics.AMC_PLUS,
            record_dropped_lo_releases=True,
        ),
        agent_period=10,
        reward_mode=reward_mode,
    )
    env.reset(seed=0)
    assert env._engine is not None
    env._monitor.job_start_count = 1
    env._engine = _StubEngine(
        runtime_budgets=env._engine.runtime_budgets,
        result=result,
        target_time=10,
    )
    return env


def _loss_event(*, reason: str, release_index: int) -> LoJobLossEvent:
    """构造单个 LO loss 事件。"""

    return LoJobLossEvent(
        loss_time=5,
        task="l1",
        release_index=release_index,
        release_time=release_index * 12,
        executed_at_loss=0,
        budget_at_loss=2,
        reason=reason,
    )


def test_env_step_exposes_budget_cancellation_reason_split_fields() -> None:
    """当 runtime result 含有 budget cancellation loss 时，应把 Level 4 字段写入 info。"""

    env = _build_env_with_stubbed_result(
        SimulationResult(lo_job_losses=[_loss_event(reason=LO_LOSS_BUDGET_CANCELLATION, release_index=0)])
    )

    step = env.step(None)

    assert step.info["lo_budget_cancellations"] == 1
    assert step.info["delta_lo_budget_cancellations"] == 1.0
    assert step.info["lo_budget_cancellation_rate"] == 1.0
    assert "step_reward_lo_budget_cancellation" in step.info


def test_env_step_exposes_active_drop_reason_split_fields() -> None:
    """当 runtime result 含有 mode-switch active drop 时，应把 active drop 字段写入 info。"""

    env = _build_env_with_stubbed_result(
        SimulationResult(
            lo_job_losses=[_loss_event(reason=LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH, release_index=1)]
        )
    )

    step = env.step(None)

    assert step.info["lo_active_dropped_on_mode_switch"] == 1
    assert step.info["delta_lo_active_dropped_on_mode_switch"] == 1.0
    assert step.info["lo_active_drop_rate"] == 1.0
    assert "step_reward_lo_active_drop" in step.info


def test_reward_expression_can_reference_new_reason_split_variables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """新 reward 公式应能直接引用 lo_active_drop_rate 等 Level 4 变量。"""

    reward_dir = tmp_path / "reward_modes"
    reward_dir.mkdir()
    (reward_dir / "level4_test.json").write_text(
        json.dumps(
            {
                "event_weights": {"job_start": 0.0, "lo_overrun": 0.0, "hi_overrun": 0.0},
                "step_reward_formula": (
                    "- lo_budget_cancellation_penalty * lo_budget_cancellation_rate "
                    "- lo_active_drop_penalty * lo_active_drop_rate"
                ),
                "reward_parameters": {
                    "lo_budget_cancellation_penalty": 2.5,
                    "lo_active_drop_penalty": 5.0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(reward_config_module, "reward_config_dir", lambda: reward_dir)
    reward_config_module.available_reward_modes.cache_clear()
    reward_config_module.load_reward_mode_config.cache_clear()

    env = _build_env_with_stubbed_result(
        SimulationResult(
            lo_job_losses=[
                _loss_event(reason=LO_LOSS_BUDGET_CANCELLATION, release_index=0),
                _loss_event(reason=LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH, release_index=1),
            ]
        ),
        reward_mode="level4_test",
    )

    step = env.step(None)

    assert step.reward == -7.5
    reward_config_module.available_reward_modes.cache_clear()
    reward_config_module.load_reward_mode_config.cache_clear()
