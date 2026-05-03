"""DQN smoke 模型评估脚本。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from amc_py.dqn import DqnBudgetAgent
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.agents import HeuristicBudgetAgent, NoOpBudgetAgent, RandomBudgetAgent
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.runtime_wrapper import AgentRuntimeConfig, simulate_ordered_taskset_with_agent
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import ExecutionScenario, make_table_scenario


def _build_small_taskset() -> list[Task]:
    """构造 smoke 评估使用的小任务集。"""

    return [
        Task("T1", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
        Task("T2", period=15, deadline=15, c_lo=2, c_hi=2, criticality=Criticality.LO),
        Task("T3", period=20, deadline=20, c_lo=3, c_hi=3, criticality=Criticality.LO),
    ]


def _build_stress_scenario() -> ExecutionScenario:
    """构造与 smoke 训练一致的压力场景。"""

    return make_table_scenario(
        actual_costs={
            ("T2", 0): 5,
            ("T2", 1): 5,
            ("T2", 2): 4,
            ("T2", 3): 5,
            ("T3", 1): 5,
            ("T1", 0): 3,
            ("T1", 2): 3,
        },
        default_hi="c_lo",
        default_lo="c_lo",
    )


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/dqn_smoke/eval_summary.csv"))
    return parser


def _evaluate_dqn_agent(
    *,
    model_path: Path,
    tasks: list[Task],
    scenario: ExecutionScenario,
    runtime_config: RuntimeConfig,
    agent_period: int,
    seed: int,
) -> dict[str, int | float | str]:
    """以评估模式运行 DQN agent，并汇总关键统计。"""

    env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=scenario,
        runtime_config=runtime_config,
        agent_period=agent_period,
        check_safety=True,
    )
    agent = DqnBudgetAgent.load(model_path)

    obs = env.reset(seed=seed)
    done = False
    accepted_actions = 0
    rejected_actions = 0
    noop_actions = 0
    total_reward = 0.0
    total_valid_action_count = 0
    total_masked_action_count = 0
    noop_due_to_no_valid_action = 0
    last_info: dict[str, int | float | str | bool | None] = {
        "mode_changes": 0,
        "lo_cancellations": 0,
        "deadline_misses": 0,
    }

    while not done:
        mask = env.valid_action_mask()
        valid_action_count = sum(mask)
        masked_action_count = len(mask) - valid_action_count
        action_id = agent.select_action_id(
            obs.state_vector,
            valid_action_mask=mask,
            training=False,
        )
        if action_id is None:
            noop_actions += 1
            noop_due_to_no_valid_action += 1

        result = env.step(action_id)
        total_reward += result.reward
        last_info = result.info
        total_valid_action_count += valid_action_count
        total_masked_action_count += masked_action_count

        if action_id is not None:
            if bool(result.info.get("accepted")):
                accepted_actions += 1
            else:
                rejected_actions += 1

        obs = result.observation
        done = result.done

    return {
        "method": "dqn_agent",
        "seed": seed,
        "end_time": runtime_config.end_time,
        "agent_period": agent_period,
        "mode_changes": int(last_info.get("mode_changes", 0)),
        "lo_cancellations": int(last_info.get("lo_cancellations", 0)),
        "deadline_misses": int(last_info.get("deadline_misses", 0)),
        "accepted_actions": accepted_actions,
        "rejected_actions": rejected_actions,
        "noop_actions": noop_actions,
        "total_reward": total_reward,
        "valid_action_count": total_valid_action_count,
        "masked_action_count": total_masked_action_count,
        "noop_due_to_no_valid_action": noop_due_to_no_valid_action,
    }


def main() -> None:
    """运行 DQN smoke 模型与各基线的统一评估。"""

    args = build_parser().parse_args()
    tasks = _build_small_taskset()
    scenario = _build_stress_scenario()
    runtime_config = RuntimeConfig(end_time=args.end_time, semantics=RuntimeSemantics.AMC_PLUS)
    actions = build_budget_action_space(tasks)

    baseline_result = simulate_ordered_taskset_event_driven(
        ordered_tasks=tasks,
        scenario=scenario,
        config=runtime_config,
    )
    noop_result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=scenario,
        agent=NoOpBudgetAgent(),
        runtime_config=runtime_config,
        agent_config=AgentRuntimeConfig(agent_period=args.agent_period, end_time=args.end_time, check_safety=True),
    )
    random_result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=scenario,
        agent=RandomBudgetAgent(actions=actions, seed=args.seed),
        runtime_config=runtime_config,
        agent_config=AgentRuntimeConfig(agent_period=args.agent_period, end_time=args.end_time, check_safety=True),
    )
    heuristic_result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=scenario,
        agent=HeuristicBudgetAgent(actions=actions),
        runtime_config=runtime_config,
        agent_config=AgentRuntimeConfig(agent_period=args.agent_period, end_time=args.end_time, check_safety=True),
    )
    dqn_row = _evaluate_dqn_agent(
        model_path=args.model,
        tasks=tasks,
        scenario=scenario,
        runtime_config=runtime_config,
        agent_period=args.agent_period,
        seed=args.seed,
    )

    rows = [
        {
            "method": "amc_plus_baseline",
            "seed": args.seed,
            "end_time": args.end_time,
            "agent_period": args.agent_period,
            "mode_changes": baseline_result.mode_change_count(),
            "lo_cancellations": baseline_result.lo_job_cancellation_count(),
            "deadline_misses": len(baseline_result.deadline_misses),
            "accepted_actions": 0,
            "rejected_actions": 0,
            "noop_actions": 0,
            "total_reward": 0.0,
            "valid_action_count": 0,
            "masked_action_count": 0,
            "noop_due_to_no_valid_action": 0,
        },
        {
            "method": "noop_agent",
            "seed": args.seed,
            "end_time": args.end_time,
            "agent_period": args.agent_period,
            "mode_changes": noop_result.runtime_result.mode_change_count(),
            "lo_cancellations": noop_result.runtime_result.lo_job_cancellation_count(),
            "deadline_misses": len(noop_result.runtime_result.deadline_misses),
            "accepted_actions": noop_result.accepted_actions,
            "rejected_actions": noop_result.rejected_actions,
            "noop_actions": noop_result.noop_actions,
            "total_reward": noop_result.total_reward,
            "valid_action_count": 0,
            "masked_action_count": 0,
            "noop_due_to_no_valid_action": 0,
        },
        {
            "method": "random_agent",
            "seed": args.seed,
            "end_time": args.end_time,
            "agent_period": args.agent_period,
            "mode_changes": random_result.runtime_result.mode_change_count(),
            "lo_cancellations": random_result.runtime_result.lo_job_cancellation_count(),
            "deadline_misses": len(random_result.runtime_result.deadline_misses),
            "accepted_actions": random_result.accepted_actions,
            "rejected_actions": random_result.rejected_actions,
            "noop_actions": random_result.noop_actions,
            "total_reward": random_result.total_reward,
            "valid_action_count": 0,
            "masked_action_count": 0,
            "noop_due_to_no_valid_action": 0,
        },
        {
            "method": "heuristic_agent",
            "seed": args.seed,
            "end_time": args.end_time,
            "agent_period": args.agent_period,
            "mode_changes": heuristic_result.runtime_result.mode_change_count(),
            "lo_cancellations": heuristic_result.runtime_result.lo_job_cancellation_count(),
            "deadline_misses": len(heuristic_result.runtime_result.deadline_misses),
            "accepted_actions": heuristic_result.accepted_actions,
            "rejected_actions": heuristic_result.rejected_actions,
            "noop_actions": heuristic_result.noop_actions,
            "total_reward": heuristic_result.total_reward,
            "valid_action_count": 0,
            "masked_action_count": 0,
            "noop_due_to_no_valid_action": 0,
        },
        dqn_row,
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "seed",
        "end_time",
        "agent_period",
        "mode_changes",
        "lo_cancellations",
        "deadline_misses",
        "accepted_actions",
        "rejected_actions",
        "noop_actions",
        "total_reward",
        "valid_action_count",
        "masked_action_count",
        "noop_due_to_no_valid_action",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
