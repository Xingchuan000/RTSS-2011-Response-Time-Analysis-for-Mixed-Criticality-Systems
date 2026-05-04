"""正式 DQN 评估命令行入口。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from amc_py.automotive_workload import build_automotive_experiment_config
from amc_py.dqn import (
    DqnBudgetAgent,
    build_env_from_experiment_config,
    build_rtss11_experiment_config,
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
    """将种子字符串解析为整数列表，支持 `a:b` 与逗号列表。"""

    seeds: list[int] = []
    for part in (item.strip() for item in raw_value.split(",")):
        if not part:
            continue
        if ":" in part:
            begin_text, end_text = (token.strip() for token in part.split(":", maxsplit=1))
            begin = int(begin_text)
            end = int(end_text)
            if end < begin:
                raise ValueError(f"seed 区间必须满足 begin<=end，收到: {part}")
            seeds.extend(range(begin, end + 1))
        else:
            seeds.append(int(part))
    return seeds


def _parse_baselines(raw_value: str) -> list[str]:
    """将逗号分隔 baseline 列表解析为方法名列表。"""

    return [part.strip() for part in raw_value.split(",") if part.strip()]


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
    reward_mode: str,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    trace_dir: Path | None = None,
    debug_log_dir: Path | None = None,
    trace_enabled: bool = False,
) -> tuple[dict[str, int | float | str | bool], SimulationResult, list[dict[str, object]]]:
    """以评估模式运行一次 DQN agent。"""

    env = build_env_from_experiment_config(
        experiment_config,
        seed=seed,
        end_time=end_time,
        agent_period=agent_period,
        semantics=RuntimeSemantics.AMC_PLUS,
        reward_mode=reward_mode,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
    )
    agent = DqnBudgetAgent.load(model_path)
    if agent.action_dim != env.action_space_size:
        raise ValueError(
            "模型动作空间与环境不兼容："
            f"model.action_dim={agent.action_dim}, env.action_space_size={env.action_space_size}"
        )

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
    debug_stats = env.debug_statistics()
    action_total = accepted_actions + rejected_actions + noop_actions
    rejection_rate = (rejected_actions / action_total) if action_total > 0 else 0.0
    if trace_enabled and hasattr(env, "_engine") and env._engine is not None:
        _write_agent_debug_files(
            trace_dir=trace_dir,
            debug_log_dir=debug_log_dir,
            seed=seed,
            method="dqn_agent",
            action_log=env.action_log,
            mask_log=env.mask_log,
            runtime_result=env._engine.finish(),
        )

    runtime_result = env._engine.finish() if env._engine is not None else SimulationResult()
    return (
        {
            **row_base,
            "method": "dqn_agent",
            "mode_changes": int(last_info.get("mode_changes", 0)),
            "lo_cancellations": int(last_info.get("lo_cancellations", 0)),
            "deadline_misses": int(last_info.get("deadline_misses", 0)),
            "budget_overruns": budget_overruns,
            "accepted_actions": accepted_actions,
            "rejected_actions": rejected_actions,
            "noop_actions": noop_actions,
            "rejection_rate": rejection_rate,
            "total_reward": total_reward,
            "check_safety": bool(debug_stats["check_safety"]),
            "safety_checked_actions": int(debug_stats["safety_checked_actions"]),
            "safety_accepted_actions": int(debug_stats["safety_accepted_actions"]),
            "safety_rejected_actions": int(debug_stats["safety_rejected_actions"]),
            "valid_action_count_mean": float(debug_stats["valid_action_count_mean"]),
            "masked_action_count_mean": float(debug_stats["masked_action_count_mean"]),
            "masked_action_count_max": int(debug_stats["masked_action_count_max"]),
            "mask_rejection_rate_mean": float(debug_stats["mask_rejection_rate_mean"]),
            "selected_invalid_mask_actions": int(debug_stats["selected_invalid_mask_actions"]),
            "action_space_type": str(debug_stats["action_space_type"]),
            "action_count": int(debug_stats["action_count"]),
            "budget_increase_ratio": float(debug_stats["budget_increase_ratio"]),
            "budget_decrease_ratio": float(debug_stats["budget_decrease_ratio"]),
            "no_safe_action_steps": int(debug_stats["no_safe_action_steps"]),
        },
        runtime_result,
        env.action_log,
    )


def _parse_csv_set(raw_value: str) -> set[str]:
    """将逗号分隔字符串转为去空白集合。"""

    return {part.strip() for part in raw_value.split(",") if part.strip()}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """写 jsonl 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _trace_rows_from_runtime(result: SimulationResult) -> list[dict]:
    """将 runtime 结果转为 jsonl trace 行。

    输出顺序上先写逐 tick 调度快照，再追加事件级 debug 日志。这样一份文件里
    同时包含“CPU 在跑谁”和“为什么发生切换/更新/miss”的两条视角。
    """

    rows: list[dict] = []
    for tick in result.trace:
        rows.append(
            {
                "event": "schedule_tick",
                "time": tick.time,
                "executing_task": tick.executing_task,
                "executing_release_index": tick.executing_release_index,
                "mode": tick.mode.name,
            }
        )
    rows.extend(result.debug_events)
    for miss in result.deadline_misses:
        rows.append(
            {
                "event": "deadline_miss",
                "task": miss.task,
                "release_index": miss.release_index,
                "release_time": miss.release_time,
                "absolute_deadline": miss.absolute_deadline,
                "mode_at_miss": miss.mode_at_miss.name,
                "executed_at_miss": miss.executed_at_miss,
            }
        )
    return rows


def _build_runtime_budget_timeline(result: SimulationResult) -> dict[str, list[tuple[int, int]]]:
    """为每个任务构造预算时间线，供 deadline miss 详情回溯使用。"""

    if not result.jobs:
        return {}
    initial_budgets = {job.task.name: job.task.c_lo for job in result.jobs}
    timeline: dict[str, list[tuple[int, int]]] = {task_name: [(0, budget)] for task_name, budget in initial_budgets.items()}
    for update in result.budget_update_events:
        for task_name, budget in update.updates.items():
            timeline.setdefault(task_name, [(0, budget)])
            timeline[task_name].append((update.time, budget))
    return timeline


def _budget_at_time(timeline: dict[str, list[tuple[int, int]]], task_name: str, time: int) -> int | None:
    """查询某任务在指定时刻的全局预算值。"""

    budget = None
    for update_time, candidate_budget in timeline.get(task_name, []):
        if update_time > time:
            break
        budget = candidate_budget
    return budget


def _last_action_before(action_log: list[dict], time: int) -> dict | None:
    """返回 miss 发生前最近一次 agent 决策。"""

    last_action: dict | None = None
    for row in action_log:
        if int(row.get("time", -1)) <= time:
            last_action = row
        else:
            break
    return last_action


def _deadline_miss_detail_rows(
    *,
    row_base: dict[str, int | float | str | bool],
    method: str,
    runtime_result: SimulationResult,
    action_log: list[dict],
) -> list[dict[str, object]]:
    """展开 deadline miss 详情。

    这里不只输出 miss 数量，而是把 task/job/budget/action 四条信息链拼到一起，
    这样看到某条 miss 记录时就能直接回答：
    - 是哪个 job miss；
    - 释放时预算是多少；
    - miss 当刻全局预算又是多少；
    - miss 前最近一次动作是谁、何时发生、是否被接受。
    """

    jobs_by_key = {(job.task.name, job.release_index): job for job in runtime_result.jobs}
    budget_timeline = _build_runtime_budget_timeline(runtime_result)
    detail_rows: list[dict[str, object]] = []
    for miss in runtime_result.deadline_misses:
        job = jobs_by_key[(miss.task, miss.release_index)]
        last_action = _last_action_before(action_log, miss.absolute_deadline)
        last_budget_update_time = None
        for event in runtime_result.budget_update_events:
            if event.time <= miss.absolute_deadline:
                last_budget_update_time = event.time
            else:
                break
        detail_rows.append(
            {
                "workload": row_base["workload"],
                "total_util": row_base["total_util"],
                "seed": row_base["seed"],
                "method": method,
                "time": miss.absolute_deadline,
                "task": job.task.name,
                "criticality": job.task.criticality.value,
                "release_index": job.release_index,
                "release_time": job.release_time,
                "absolute_deadline": job.absolute_deadline,
                "actual_cost": job.actual_cost,
                "executed_at_miss": miss.executed_at_miss,
                "runtime_budget_at_release": job.runtime_budget_at_release,
                "current_global_budget": _budget_at_time(budget_timeline, job.task.name, miss.absolute_deadline),
                "completion_time": job.completion_time,
                "dropped": job.dropped,
                "drop_time": job.drop_time,
                "mode_at_miss": miss.mode_at_miss.name,
                "last_action_time": None if last_action is None else last_action.get("time"),
                "last_action_id": None if last_action is None else last_action.get("action_id"),
                "last_action_accepted": None if last_action is None else last_action.get("accepted"),
                "last_action_updates": None if last_action is None else last_action.get("updates"),
                "last_budget_update_time": last_budget_update_time,
            }
        )
    return detail_rows


def _write_agent_debug_files(
    *,
    trace_dir: Path | None,
    debug_log_dir: Path | None,
    seed: int,
    method: str,
    action_log: list[dict],
    runtime_result: SimulationResult,
    mask_log: list[dict] | None = None,
) -> None:
    """按方法/seed 写出 action、trace、debug 三类文件。"""

    if trace_dir is not None:
        _write_jsonl(trace_dir / f"seed{seed}_{method}_action_log.jsonl", action_log)
        _write_jsonl(trace_dir / f"seed{seed}_{method}_runtime_trace.jsonl", _trace_rows_from_runtime(runtime_result))
        if mask_log is not None:
            _write_jsonl(trace_dir / f"seed{seed}_{method}_mask_log.jsonl", mask_log)
    if debug_log_dir is not None:
        _write_jsonl(debug_log_dir / f"seed{seed}_{method}_debug_events.jsonl", runtime_result.debug_events)


def _deadline_miss_rows(rows: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    """筛选 deadline_misses > 0 的记录。"""

    return [row for row in rows if int(row["deadline_misses"]) > 0]


def build_parser() -> argparse.ArgumentParser:
    """构建正式评估 CLI 的参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["small", "rtss11", "automotive"], default="small")
    parser.add_argument("--total-util", type=float, default=0.65)
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--cf", type=float, default=2.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--scenario-seed-offset", type=int, default=100000)
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
    parser.add_argument("--fail-on-deadline-miss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--trace-seeds", type=str, default="")
    parser.add_argument("--trace-methods", type=str, default="")
    parser.add_argument("--debug-log-dir", type=Path, default=None)
    parser.add_argument(
        "--reward-mode",
        choices=["mendes", "event_delta", "event_delta_no_job_start"],
        default="mendes",
    )
    parser.add_argument("--action-space", choices=["triple", "pair", "single"], default="triple")
    parser.add_argument("--budget-increase-ratio", type=float, default=0.10)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.05)
    parser.add_argument("--include-explicit-noop", action=argparse.BooleanOptionalAction, default=False)
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
            scenario_seed_offset=args.scenario_seed_offset,
        )
    else:
        experiment_config = build_automotive_experiment_config(
            num_runnables=150,
            require_schedulable=args.require_schedulable,
        )

    enabled_methods = set(_parse_baselines(args.baselines))
    trace_seed_set = {int(s) for s in _parse_csv_set(args.trace_seeds)}
    trace_method_set = _parse_csv_set(args.trace_methods)
    valid_methods = {"amc_plus_baseline", "noop_agent", "random_agent", "heuristic_agent", "dqn_agent"}
    unsupported_methods = sorted(enabled_methods - valid_methods)
    if unsupported_methods:
        raise ValueError(f"不支持的 baselines: {unsupported_methods}")

    rows: list[dict[str, int | float | str]] = []
    deadline_miss_details: list[dict[str, object]] = []
    for seed in _parse_seeds(args.seeds):
        bundle = resolve_experiment_bundle(experiment_config, seed)
        runtime_config = RuntimeConfig(end_time=args.end_time, semantics=RuntimeSemantics.AMC_PLUS)
        actions = build_budget_action_space(
            list(bundle.ordered_tasks),
            action_space=args.action_space,
            budget_increase_ratio=args.budget_increase_ratio,
            budget_decrease_ratio=args.budget_decrease_ratio,
            include_explicit_noop=args.include_explicit_noop,
        )

        if args.workload == "rtss11":
            if args.require_schedulable:
                amc_rtb_schedulable = True
            else:
                amc_rtb_schedulable = evaluate_taskset(
                    list(bundle.ordered_tasks),
                    method="amc_rtb",
                    priority_policy="opa",
                ).schedulable
            attempts = bundle.taskset_attempts
            taskset_seed = bundle.taskset_seed if bundle.taskset_seed is not None else seed
            scenario_seed = bundle.scenario_seed if bundle.scenario_seed is not None else seed
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
        trace_enabled_for_seed = (
            (args.trace_dir is not None or args.debug_log_dir is not None)
            and (not trace_seed_set or seed in trace_seed_set)
        )

        if "amc_plus_baseline" in enabled_methods:
            baseline_result = simulate_ordered_taskset_event_driven(
                ordered_tasks=list(bundle.ordered_tasks),
                scenario=bundle.scenario,
                config=runtime_config,
            )
            baseline_rejection_rate = 0.0
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
                    "rejection_rate": baseline_rejection_rate,
                    "total_reward": 0.0,
                    "check_safety": True,
                    "safety_checked_actions": 0,
                    "safety_accepted_actions": 0,
                    "safety_rejected_actions": 0,
                    "valid_action_count_mean": 0.0,
                    "masked_action_count_mean": 0.0,
                    "masked_action_count_max": 0,
                    "mask_rejection_rate_mean": 0.0,
                    "selected_invalid_mask_actions": 0,
                    "action_space_type": args.action_space,
                    "action_count": len(actions),
                    "budget_increase_ratio": args.budget_increase_ratio,
                    "budget_decrease_ratio": args.budget_decrease_ratio,
                    "no_safe_action_steps": 0,
                }
            )
            deadline_miss_details.extend(
                _deadline_miss_detail_rows(
                    row_base=row_base,
                    method="amc_plus_baseline",
                    runtime_result=baseline_result,
                    action_log=[],
                )
            )

        if "noop_agent" in enabled_methods:
            noop_result = simulate_ordered_taskset_with_agent(
                ordered_tasks=list(bundle.ordered_tasks),
                scenario=bundle.scenario,
                agent=NoOpBudgetAgent(),
                runtime_config=runtime_config,
                agent_config=AgentRuntimeConfig(
                    agent_period=args.agent_period,
                    end_time=args.end_time,
                    check_safety=True,
                    reward_mode=args.reward_mode,
                ),
                bounds=bundle.normalization_bounds,
            )
            noop_total_actions = (
                noop_result.accepted_actions + noop_result.rejected_actions + noop_result.noop_actions
            )
            noop_rejection_rate = (
                (noop_result.rejected_actions / noop_total_actions) if noop_total_actions > 0 else 0.0
            )
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
                    "rejection_rate": noop_rejection_rate,
                    "total_reward": noop_result.total_reward,
                    "check_safety": True,
                    "safety_checked_actions": noop_result.safety_checked_actions,
                    "safety_accepted_actions": noop_result.safety_accepted_actions,
                    "safety_rejected_actions": noop_result.safety_rejected_actions,
                    "valid_action_count_mean": 0.0,
                    "masked_action_count_mean": 0.0,
                    "masked_action_count_max": 0,
                    "mask_rejection_rate_mean": 0.0,
                    "selected_invalid_mask_actions": 0,
                    "action_space_type": args.action_space,
                    "action_count": len(actions),
                    "budget_increase_ratio": args.budget_increase_ratio,
                    "budget_decrease_ratio": args.budget_decrease_ratio,
                    "no_safe_action_steps": 0,
                }
            )
            deadline_miss_details.extend(
                _deadline_miss_detail_rows(
                    row_base=row_base,
                    method="noop_agent",
                    runtime_result=noop_result.runtime_result,
                    action_log=noop_result.action_log,
                )
            )
            if (
                trace_enabled_for_seed
                and (not trace_method_set or "noop_agent" in trace_method_set)
            ):
                _write_agent_debug_files(
                    trace_dir=args.trace_dir,
                    debug_log_dir=args.debug_log_dir,
                    seed=seed,
                    method="noop_agent",
                    action_log=noop_result.action_log,
                    runtime_result=noop_result.runtime_result,
                )

        if "random_agent" in enabled_methods:
            random_result = simulate_ordered_taskset_with_agent(
                ordered_tasks=list(bundle.ordered_tasks),
                scenario=bundle.scenario,
                agent=RandomBudgetAgent(actions=actions, seed=seed),
                runtime_config=runtime_config,
                agent_config=AgentRuntimeConfig(
                    agent_period=args.agent_period,
                    end_time=args.end_time,
                    check_safety=True,
                    reward_mode=args.reward_mode,
                ),
                bounds=bundle.normalization_bounds,
            )
            random_total_actions = (
                random_result.accepted_actions + random_result.rejected_actions + random_result.noop_actions
            )
            random_rejection_rate = (
                (random_result.rejected_actions / random_total_actions) if random_total_actions > 0 else 0.0
            )
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
                    "rejection_rate": random_rejection_rate,
                    "total_reward": random_result.total_reward,
                    "check_safety": True,
                    "safety_checked_actions": random_result.safety_checked_actions,
                    "safety_accepted_actions": random_result.safety_accepted_actions,
                    "safety_rejected_actions": random_result.safety_rejected_actions,
                    "valid_action_count_mean": 0.0,
                    "masked_action_count_mean": 0.0,
                    "masked_action_count_max": 0,
                    "mask_rejection_rate_mean": 0.0,
                    "selected_invalid_mask_actions": 0,
                    "action_space_type": args.action_space,
                    "action_count": len(actions),
                    "budget_increase_ratio": args.budget_increase_ratio,
                    "budget_decrease_ratio": args.budget_decrease_ratio,
                    "no_safe_action_steps": 0,
                }
            )
            deadline_miss_details.extend(
                _deadline_miss_detail_rows(
                    row_base=row_base,
                    method="random_agent",
                    runtime_result=random_result.runtime_result,
                    action_log=random_result.action_log,
                )
            )
            if (
                trace_enabled_for_seed
                and (not trace_method_set or "random_agent" in trace_method_set)
            ):
                _write_agent_debug_files(
                    trace_dir=args.trace_dir,
                    debug_log_dir=args.debug_log_dir,
                    seed=seed,
                    method="random_agent",
                    action_log=random_result.action_log,
                    runtime_result=random_result.runtime_result,
                )

        if "heuristic_agent" in enabled_methods:
            heuristic_result = simulate_ordered_taskset_with_agent(
                ordered_tasks=list(bundle.ordered_tasks),
                scenario=bundle.scenario,
                agent=HeuristicBudgetAgent(actions=actions),
                runtime_config=runtime_config,
                agent_config=AgentRuntimeConfig(
                    agent_period=args.agent_period,
                    end_time=args.end_time,
                    check_safety=True,
                    reward_mode=args.reward_mode,
                ),
                bounds=bundle.normalization_bounds,
            )
            heuristic_total_actions = (
                heuristic_result.accepted_actions
                + heuristic_result.rejected_actions
                + heuristic_result.noop_actions
            )
            heuristic_rejection_rate = (
                (heuristic_result.rejected_actions / heuristic_total_actions)
                if heuristic_total_actions > 0
                else 0.0
            )
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
                    "rejection_rate": heuristic_rejection_rate,
                    "total_reward": heuristic_result.total_reward,
                    "check_safety": True,
                    "safety_checked_actions": heuristic_result.safety_checked_actions,
                    "safety_accepted_actions": heuristic_result.safety_accepted_actions,
                    "safety_rejected_actions": heuristic_result.safety_rejected_actions,
                    "valid_action_count_mean": 0.0,
                    "masked_action_count_mean": 0.0,
                    "masked_action_count_max": 0,
                    "mask_rejection_rate_mean": 0.0,
                    "selected_invalid_mask_actions": 0,
                    "action_space_type": args.action_space,
                    "action_count": len(actions),
                    "budget_increase_ratio": args.budget_increase_ratio,
                    "budget_decrease_ratio": args.budget_decrease_ratio,
                    "no_safe_action_steps": 0,
                }
            )
            deadline_miss_details.extend(
                _deadline_miss_detail_rows(
                    row_base=row_base,
                    method="heuristic_agent",
                    runtime_result=heuristic_result.runtime_result,
                    action_log=heuristic_result.action_log,
                )
            )
            if (
                trace_enabled_for_seed
                and (not trace_method_set or "heuristic_agent" in trace_method_set)
            ):
                _write_agent_debug_files(
                    trace_dir=args.trace_dir,
                    debug_log_dir=args.debug_log_dir,
                    seed=seed,
                    method="heuristic_agent",
                    action_log=heuristic_result.action_log,
                    runtime_result=heuristic_result.runtime_result,
                )

        if "dqn_agent" in enabled_methods:
            dqn_row, dqn_runtime_result, dqn_action_log = _evaluate_dqn_once(
                model_path=args.model,
                experiment_config=experiment_config,
                agent_period=args.agent_period,
                seed=seed,
                end_time=args.end_time,
                row_base=row_base,
                reward_mode=args.reward_mode,
                action_space=args.action_space,
                budget_increase_ratio=args.budget_increase_ratio,
                budget_decrease_ratio=args.budget_decrease_ratio,
                include_explicit_noop=args.include_explicit_noop,
                trace_dir=args.trace_dir,
                debug_log_dir=args.debug_log_dir,
                trace_enabled=trace_enabled_for_seed and (not trace_method_set or "dqn_agent" in trace_method_set),
            )
            rows.append(dqn_row)
            deadline_miss_details.extend(
                _deadline_miss_detail_rows(
                    row_base=row_base,
                    method="dqn_agent",
                    runtime_result=dqn_runtime_result,
                    action_log=dqn_action_log,
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
        "rejection_rate",
        "total_reward",
        "check_safety",
        "safety_checked_actions",
        "safety_accepted_actions",
        "safety_rejected_actions",
        "valid_action_count_mean",
        "masked_action_count_mean",
        "masked_action_count_max",
        "mask_rejection_rate_mean",
        "selected_invalid_mask_actions",
        "action_space_type",
        "action_count",
        "budget_increase_ratio",
        "budget_decrease_ratio",
        "no_safe_action_steps",
        "end_time",
        "agent_period",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    miss_rows = _deadline_miss_rows(rows)
    if miss_rows:
        detail_output_path = args.output.with_name(f"{args.output.stem}_deadline_misses.jsonl")
        _write_jsonl(detail_output_path, deadline_miss_details)
    if args.fail_on_deadline_miss and miss_rows:
        print("Deadline miss detected under check_safety=True:")
        for row in miss_rows:
            print(
                "  "
                f"total_util={row['total_util']} seed={row['seed']} "
                f"method={row['method']} deadline_misses={row['deadline_misses']}"
            )
        print("First deadline miss details:")
        printed = 0
        for detail_row in deadline_miss_details:
            if printed >= 3:
                break
            print(
                "  "
                f"seed={detail_row['seed']} method={detail_row['method']} task={detail_row['task']} "
                f"rel={detail_row['release_index']} deadline={detail_row['absolute_deadline']} "
                f"executed={detail_row['executed_at_miss']} "
                f"budget_at_release={detail_row['runtime_budget_at_release']}"
            )
            printed += 1
        print("Evaluation failed because --fail-on-deadline-miss is enabled.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
