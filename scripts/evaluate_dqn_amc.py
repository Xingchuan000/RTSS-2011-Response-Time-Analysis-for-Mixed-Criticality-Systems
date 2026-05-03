"""正式 DQN 评估命令行入口。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from amc_py.dqn import (
    DqnBudgetAgent,
    build_env_from_experiment_config,
    build_small_nominal_experiment_config,
    build_small_stress_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Task
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.agents import HeuristicBudgetAgent, NoOpBudgetAgent, RandomBudgetAgent
from amc_py.rl.runtime_wrapper import AgentRuntimeConfig, simulate_ordered_taskset_with_agent
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics


def _parse_seeds(raw_value: str) -> list[int]:
    """将逗号分隔的种子列表解析为整数列表。"""

    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def _evaluate_dqn_once(
    *,
    model_path: Path,
    experiment_config,
    agent_period: int,
    seed: int,
    end_time: int,
) -> dict[str, int | float | str]:
    """以评估模式运行一次 DQN agent。"""

    env = build_env_from_experiment_config(
        experiment_config,
        seed=seed,
        end_time=end_time,
        agent_period=agent_period,
        semantics=RuntimeSemantics.AMC_PLUS,
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
        action_id = agent.select_action_id(obs.state_vector, valid_action_mask=mask, training=False)
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
        "end_time": end_time,
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


def build_parser() -> argparse.ArgumentParser:
    """构建正式评估 CLI 的参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=10)
    parser.add_argument("--scenario", choices=["nominal", "stress"], default="stress")
    parser.add_argument("--output", type=Path, default=Path("outputs/dqn_amc/eval_summary.csv"))
    return parser


def main() -> None:
    """运行正式 DQN 评估，并输出统一 CSV。"""

    args = build_parser().parse_args()
    experiment_config = (
        build_small_nominal_experiment_config()
        if args.scenario == "nominal"
        else build_small_stress_experiment_config()
    )

    rows: list[dict[str, int | float | str]] = []
    for seed in _parse_seeds(args.seeds):
        bundle = resolve_experiment_bundle(experiment_config, seed)
        runtime_config = RuntimeConfig(end_time=args.end_time, semantics=RuntimeSemantics.AMC_PLUS)
        actions = build_budget_action_space(list(bundle.ordered_tasks))
        baseline_result = simulate_ordered_taskset_event_driven(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            config=runtime_config,
        )
        noop_result = simulate_ordered_taskset_with_agent(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            agent=NoOpBudgetAgent(),
            runtime_config=runtime_config,
            agent_config=AgentRuntimeConfig(agent_period=args.agent_period, end_time=args.end_time, check_safety=True),
            bounds=bundle.normalization_bounds,
        )
        random_result = simulate_ordered_taskset_with_agent(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            agent=RandomBudgetAgent(actions=actions, seed=seed),
            runtime_config=runtime_config,
            agent_config=AgentRuntimeConfig(agent_period=args.agent_period, end_time=args.end_time, check_safety=True),
            bounds=bundle.normalization_bounds,
        )
        heuristic_result = simulate_ordered_taskset_with_agent(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            agent=HeuristicBudgetAgent(actions=actions),
            runtime_config=runtime_config,
            agent_config=AgentRuntimeConfig(agent_period=args.agent_period, end_time=args.end_time, check_safety=True),
            bounds=bundle.normalization_bounds,
        )

        rows.extend(
            [
                {
                    "method": "amc_plus_baseline",
                    "seed": seed,
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
                    "seed": seed,
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
                    "seed": seed,
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
                    "seed": seed,
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
                _evaluate_dqn_once(
                    model_path=args.model,
                    experiment_config=experiment_config,
                    agent_period=args.agent_period,
                    seed=seed,
                    end_time=args.end_time,
                ),
            ]
        )

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
