"""正式 DQN 评估命令行入口。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from amc_py.automotive_workload import build_automotive_experiment_config
from amc_py.dqn import (
    DqnBudgetAgent,
    build_env_from_experiment_config,
    build_rtss11_experiment_config,
    build_rtss11_taskset,
    build_schedulable_rtss11_taskset,
    build_small_nominal_experiment_config,
    build_small_stress_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.experiments import evaluate_taskset
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.agents import HeuristicBudgetAgent, NoOpBudgetAgent, RandomBudgetAgent
from amc_py.rl.runtime_wrapper import AgentRuntimeConfig, simulate_ordered_taskset_with_agent
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SimulationResult


def _parse_seeds(raw_value: str) -> list[int]:
    """将逗号分隔的种子列表解析为整数列表。"""

    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def _parse_baselines(raw_value: str) -> list[str]:
    """将逗号分隔 baseline 列表解析为方法名列表。"""

    return [part.strip() for part in raw_value.split(",") if part.strip()]


def _derive_taskset_seed(seed: int) -> int:
    """与 RTSS2011 experiment config 保持一致的 taskset seed 派生规则。"""

    return seed * 2


def _derive_scenario_seed(seed: int) -> int:
    """与 RTSS2011 experiment config 保持一致的 scenario seed 派生规则。"""

    return seed * 2 + 1


def _build_rtss11_metadata(
    *,
    seed: int,
    total_util: float,
    num_tasks: int,
    cf: float,
    cp: float,
    require_schedulable: bool,
) -> tuple[bool, int]:
    """生成 RTSS2011 评估行所需的 AMC-rtb 可调度性元数据。"""

    taskset_seed = _derive_taskset_seed(seed)
    if require_schedulable:
        bundle = build_schedulable_rtss11_taskset(
            seed=taskset_seed,
            total_util=total_util,
            num_tasks=num_tasks,
            cf=cf,
            cp=cp,
        )
        return bundle.analysis.schedulable, bundle.attempts

    tasks = build_rtss11_taskset(
        seed=taskset_seed,
        total_util=total_util,
        num_tasks=num_tasks,
        cf=cf,
        cp=cp,
    )
    analysis = evaluate_taskset(tasks, method="amc_rtb", priority_policy="opa")
    return analysis.schedulable, 1


def _budget_overruns_from_result(result: SimulationResult) -> int:
    """在 AMC_PLUS 口径下估算 budget_overruns。"""

    return result.mode_change_count() + result.lo_job_cancellation_count()


def _evaluate_dqn_once(
    *,
    model_path: Path,
    experiment_config,
    agent_period: int,
    seed: int,
    end_time: int,
    row_base: dict[str, int | float | str | bool],
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

        result = env.step(action_id)
        total_reward += result.reward
        last_info = result.info

        if action_id is not None:
            if bool(result.info.get("accepted")):
                accepted_actions += 1
            else:
                rejected_actions += 1

        obs = result.observation
        done = result.done

    budget_overruns = 0
    if hasattr(env, "_engine") and env._engine is not None:
        budget_overruns = _budget_overruns_from_result(env._engine.finish())

    return {
        **row_base,
        "method": "dqn_agent",
        "mode_changes": int(last_info.get("mode_changes", 0)),
        "lo_cancellations": int(last_info.get("lo_cancellations", 0)),
        "deadline_misses": int(last_info.get("deadline_misses", 0)),
        "budget_overruns": budget_overruns,
        "accepted_actions": accepted_actions,
        "rejected_actions": rejected_actions,
        "noop_actions": noop_actions,
        "total_reward": total_reward,
    }


def build_parser() -> argparse.ArgumentParser:
    """构建正式评估 CLI 的参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["small", "rtss11", "automotive"], default="small")
    parser.add_argument("--total-util", type=float, default=0.65)
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--cf", type=float, default=2.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--require-schedulable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=1000)
    parser.add_argument("--scenario", choices=["nominal", "stress"], default="stress")
    parser.add_argument(
        "--baselines",
        type=str,
        default="amc_plus_baseline,noop_agent,random_agent,heuristic_agent,dqn_agent",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/dqn_amc/eval_summary.csv"))
    return parser


def main() -> None:
    """运行正式 DQN 评估，并输出统一 CSV。"""

    args = build_parser().parse_args()
    if args.workload == "small":
        experiment_config = (
            build_small_nominal_experiment_config()
            if args.scenario == "nominal"
            else build_small_stress_experiment_config()
        )
    elif args.workload == "rtss11":
        experiment_config = build_rtss11_experiment_config(
            total_util=args.total_util,
            num_tasks=args.num_tasks,
            cf=args.cf,
            cp=args.cp,
            require_schedulable=args.require_schedulable,
        )
    else:
        experiment_config = build_automotive_experiment_config(
            num_runnables=150,
            require_schedulable=args.require_schedulable,
        )

    enabled_methods = set(_parse_baselines(args.baselines))
    valid_methods = {"amc_plus_baseline", "noop_agent", "random_agent", "heuristic_agent", "dqn_agent"}
    unsupported_methods = sorted(enabled_methods - valid_methods)
    if unsupported_methods:
        raise ValueError(f"不支持的 baselines: {unsupported_methods}")

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

        if args.workload == "rtss11":
            amc_rtb_schedulable, attempts = _build_rtss11_metadata(
                seed=seed,
                total_util=args.total_util,
                num_tasks=args.num_tasks,
                cf=args.cf,
                cp=args.cp,
                require_schedulable=args.require_schedulable,
            )
            taskset_seed = _derive_taskset_seed(seed)
            scenario_seed = _derive_scenario_seed(seed)
        else:
            amc_rtb_schedulable = True
            attempts = 1
            taskset_seed = seed
            scenario_seed = seed

        row_base: dict[str, int | float | str | bool] = {
            "workload": args.workload,
            "total_util": args.total_util,
            "num_tasks": args.num_tasks,
            "cf": args.cf,
            "cp": args.cp,
            "seed": seed,
            "taskset_seed": taskset_seed,
            "scenario_seed": scenario_seed,
            "amc_rtb_schedulable": amc_rtb_schedulable,
            "attempts": attempts,
            "end_time": args.end_time,
            "agent_period": args.agent_period,
        }

        if "amc_plus_baseline" in enabled_methods:
            rows.append(
                {
                    **row_base,
                    "method": "amc_plus_baseline",
                    "mode_changes": baseline_result.mode_change_count(),
                    "lo_cancellations": baseline_result.lo_job_cancellation_count(),
                    "deadline_misses": len(baseline_result.deadline_misses),
                    "budget_overruns": _budget_overruns_from_result(baseline_result),
                    "accepted_actions": 0,
                    "rejected_actions": 0,
                    "noop_actions": 0,
                    "total_reward": 0.0,
                }
            )

        if "noop_agent" in enabled_methods:
            rows.append(
                {
                    **row_base,
                    "method": "noop_agent",
                    "mode_changes": noop_result.runtime_result.mode_change_count(),
                    "lo_cancellations": noop_result.runtime_result.lo_job_cancellation_count(),
                    "deadline_misses": len(noop_result.runtime_result.deadline_misses),
                    "budget_overruns": _budget_overruns_from_result(noop_result.runtime_result),
                    "accepted_actions": noop_result.accepted_actions,
                    "rejected_actions": noop_result.rejected_actions,
                    "noop_actions": noop_result.noop_actions,
                    "total_reward": noop_result.total_reward,
                }
            )

        if "random_agent" in enabled_methods:
            rows.append(
                {
                    **row_base,
                    "method": "random_agent",
                    "mode_changes": random_result.runtime_result.mode_change_count(),
                    "lo_cancellations": random_result.runtime_result.lo_job_cancellation_count(),
                    "deadline_misses": len(random_result.runtime_result.deadline_misses),
                    "budget_overruns": _budget_overruns_from_result(random_result.runtime_result),
                    "accepted_actions": random_result.accepted_actions,
                    "rejected_actions": random_result.rejected_actions,
                    "noop_actions": random_result.noop_actions,
                    "total_reward": random_result.total_reward,
                }
            )

        if "heuristic_agent" in enabled_methods:
            rows.append(
                {
                    **row_base,
                    "method": "heuristic_agent",
                    "mode_changes": heuristic_result.runtime_result.mode_change_count(),
                    "lo_cancellations": heuristic_result.runtime_result.lo_job_cancellation_count(),
                    "deadline_misses": len(heuristic_result.runtime_result.deadline_misses),
                    "budget_overruns": _budget_overruns_from_result(heuristic_result.runtime_result),
                    "accepted_actions": heuristic_result.accepted_actions,
                    "rejected_actions": heuristic_result.rejected_actions,
                    "noop_actions": heuristic_result.noop_actions,
                    "total_reward": heuristic_result.total_reward,
                }
            )

        if "dqn_agent" in enabled_methods:
            rows.append(
                _evaluate_dqn_once(
                    model_path=args.model,
                    experiment_config=experiment_config,
                    agent_period=args.agent_period,
                    seed=seed,
                    end_time=args.end_time,
                    row_base=row_base,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload",
        "total_util",
        "num_tasks",
        "cf",
        "cp",
        "seed",
        "taskset_seed",
        "scenario_seed",
        "method",
        "amc_rtb_schedulable",
        "attempts",
        "mode_changes",
        "lo_cancellations",
        "deadline_misses",
        "budget_overruns",
        "accepted_actions",
        "rejected_actions",
        "noop_actions",
        "total_reward",
        "end_time",
        "agent_period",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
