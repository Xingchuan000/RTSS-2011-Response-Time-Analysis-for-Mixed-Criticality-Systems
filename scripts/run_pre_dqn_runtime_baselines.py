"""阶段 8：运行 pre-DQN baseline 并导出 CSV。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.agents import HeuristicBudgetAgent, NoOpBudgetAgent, RandomBudgetAgent
from amc_py.rl.runtime_wrapper import AgentRuntimeConfig, simulate_ordered_taskset_with_agent
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import ExecutionScenario, make_nominal_scenario, make_table_scenario


def _build_small_taskset() -> list[Task]:
    """构造 baseline 对比使用的小任务集。"""

    return [
        Task("T1", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
        Task("T2", period=15, deadline=15, c_lo=2, c_hi=2, criticality=Criticality.LO),
        Task("T3", period=20, deadline=20, c_lo=3, c_hi=3, criticality=Criticality.LO),
    ]


def _build_stress_scenario() -> ExecutionScenario:
    """构造可稳定触发退化事件的压力场景。"""

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


def _build_scenarios(scenario_name: str) -> list[tuple[str, ExecutionScenario]]:
    """按命令行参数返回待执行场景列表。"""

    if scenario_name == "nominal":
        return [("nominal", make_nominal_scenario())]
    if scenario_name == "stress":
        return [("stress", _build_stress_scenario())]
    return [("nominal", make_nominal_scenario()), ("stress", _build_stress_scenario())]


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=int, default=1000)
    parser.add_argument("--agent-period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=["nominal", "stress", "all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/pre_dqn_baselines.csv"))
    args = parser.parse_args()

    tasks = _build_small_taskset()
    runtime_config = RuntimeConfig(end_time=args.end_time, semantics=RuntimeSemantics.AMC_PLUS)
    rows: list[dict[str, int | float | str]] = []
    actions = build_budget_action_space(tasks)
    for scenario_label, scenario in _build_scenarios(args.scenario):
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
            agent_config=AgentRuntimeConfig(
                agent_period=args.agent_period,
                end_time=args.end_time,
                check_safety=True,
            ),
        )
        random_result = simulate_ordered_taskset_with_agent(
            ordered_tasks=tasks,
            scenario=scenario,
            agent=RandomBudgetAgent(actions=actions, seed=args.seed),
            runtime_config=runtime_config,
            agent_config=AgentRuntimeConfig(
                agent_period=args.agent_period,
                end_time=args.end_time,
                check_safety=True,
            ),
        )
        heuristic_result = simulate_ordered_taskset_with_agent(
            ordered_tasks=tasks,
            scenario=scenario,
            agent=HeuristicBudgetAgent(actions=actions),
            runtime_config=runtime_config,
            agent_config=AgentRuntimeConfig(
                agent_period=args.agent_period,
                end_time=args.end_time,
                check_safety=True,
            ),
        )
        rows.extend(
            [
                {
                    "scenario": scenario_label,
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
                },
                {
                    "scenario": scenario_label,
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
                },
                {
                    "scenario": scenario_label,
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
                },
                {
                    "scenario": scenario_label,
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
                },
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
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
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
