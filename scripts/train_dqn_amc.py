"""正式 DQN 训练命令行入口。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from amc_py.automotive_workload import build_automotive_experiment_config
from amc_py.dqn import (
    DqnBudgetAgent,
    DqnConfig,
    ExperimentConfig,
    Transition,
    build_env_from_experiment_config,
    build_rtss11_experiment_config,
    build_small_nominal_experiment_config,
    build_small_stress_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SimulationResult


def _parse_hidden_layers(raw_value: str | None) -> tuple[int, ...] | None:
    """将逗号分隔的隐藏层字符串解析为整数元组。"""

    if raw_value is None or raw_value == "":
        return None
    return tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())


def _parse_seed_spec(raw_value: str) -> list[int]:
    """解析 seed 规格，支持 `0:9` 与 `0,1,2`。"""

    text = raw_value.strip()
    if not text:
        return []
    seeds: list[int] = []
    for part in (item.strip() for item in text.split(",")):
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


def _build_episode_seed_schedule(
    *,
    episodes: int,
    seed: int,
    mode: str,
    train_seeds: list[int],
) -> list[int]:
    """构建每个 episode 使用的 seed 调度表。"""

    if episodes <= 0:
        raise ValueError("episodes 必须为正整数")
    if mode == "fixed":
        return [seed for _ in range(episodes)]
    if mode == "per-episode":
        return [seed + episode for episode in range(episodes)]
    if mode == "cycle":
        if not train_seeds:
            raise ValueError("train-seed-mode=cycle 时，--train-seeds 不能为空")
        return [train_seeds[episode % len(train_seeds)] for episode in range(episodes)]
    raise ValueError(f"不支持的 train-seed-mode: {mode}")


def _serialize_tasks(tasks: list[Task]) -> list[dict[str, int | str]]:
    """将任务集转换为可写入 JSON 的结构。"""

    return [
        {
            "name": task.name,
            "period": task.period,
            "deadline": task.deadline,
            "c_lo": task.c_lo,
            "c_hi": task.c_hi,
            "criticality": task.criticality.value,
        }
        for task in tasks
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """写入 jsonl 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _trace_rows_from_runtime(result: SimulationResult) -> list[dict]:
    """将 runtime 结果转换为逐行 trace。"""

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


def _build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """按 CLI 参数构建实验配置。"""

    if args.workload == "small":
        return (
            build_small_nominal_experiment_config()
            if args.scenario == "nominal"
            else build_small_stress_experiment_config()
        )
    if args.workload == "rtss11":
        return build_rtss11_experiment_config(
            total_util=args.total_util,
            num_tasks=args.num_tasks,
            cf=args.cf,
            cp=args.cp,
            require_schedulable=args.require_schedulable,
            hi_overrun_prob=args.hi_overrun_prob,
            lo_overrun_prob=args.lo_overrun_prob,
            lo_overrun_factor=args.lo_overrun_factor,
            scenario_seed_offset=args.scenario_seed_offset,
        )
    return build_automotive_experiment_config(
        num_runnables=150,
        require_schedulable=args.require_schedulable,
    )


def _run_validation(
    *,
    agent: DqnBudgetAgent,
    experiment_config: ExperimentConfig,
    validation_seeds: list[int],
    validation_end_time: int,
    agent_period: int,
    reward_mode: str,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
) -> dict[str, int | float]:
    """在验证集上评估当前 agent，并返回聚合指标。"""

    dqn_mode_changes: list[int] = []
    dqn_lo_cancellations: list[int] = []
    dqn_deadline_misses: list[int] = []
    dqn_accepted_actions: list[int] = []
    dqn_noop_actions: list[int] = []
    dqn_valid_action_count_mean: list[float] = []
    dqn_masked_action_count_mean: list[float] = []
    dqn_total_reward: list[float] = []
    dqn_no_safe_action_steps: list[int] = []
    baseline_mode_changes: list[int] = []
    baseline_lo_cancellations: list[int] = []

    for seed in validation_seeds:
        bundle = resolve_experiment_bundle(experiment_config, seed)
        baseline_result = simulate_ordered_taskset_event_driven(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            config=RuntimeConfig(end_time=validation_end_time, semantics=RuntimeSemantics.AMC_PLUS),
        )
        baseline_mode_changes.append(baseline_result.mode_change_count())
        baseline_lo_cancellations.append(baseline_result.lo_job_cancellation_count())

        env = build_env_from_experiment_config(
            experiment_config,
            seed=seed,
            end_time=validation_end_time,
            agent_period=agent_period,
            semantics=RuntimeSemantics.AMC_PLUS,
            reward_mode=reward_mode,
            action_space=action_space,
            budget_increase_ratio=budget_increase_ratio,
            budget_decrease_ratio=budget_decrease_ratio,
            include_explicit_noop=include_explicit_noop,
        )
        obs = env.reset(seed=seed)
        done = False
        accepted_actions = 0
        noop_actions = 0
        total_reward = 0.0
        last_info: dict[str, int | float | str | bool | None] = {
            "mode_changes": 0,
            "lo_cancellations": 0,
            "deadline_misses": 0,
        }

        while not done:
            mask = env.valid_action_mask()
            action_id = agent.select_action_id(
                obs.state_vector,
                valid_action_mask=mask,
                training=False,
            )
            if action_id is None:
                noop_actions += 1
            result = env.step(action_id)
            total_reward += result.reward
            if action_id is not None and bool(result.info.get("accepted")):
                accepted_actions += 1
            obs = result.observation
            done = result.done
            last_info = result.info

        dqn_mode_changes.append(int(last_info.get("mode_changes", 0)))
        dqn_lo_cancellations.append(int(last_info.get("lo_cancellations", 0)))
        dqn_deadline_misses.append(int(last_info.get("deadline_misses", 0)))
        dqn_accepted_actions.append(accepted_actions)
        dqn_noop_actions.append(noop_actions)
        dqn_total_reward.append(total_reward)
        debug_stats = env.debug_statistics()
        dqn_valid_action_count_mean.append(float(debug_stats["valid_action_count_mean"]))
        dqn_masked_action_count_mean.append(float(debug_stats["masked_action_count_mean"]))
        dqn_no_safe_action_steps.append(int(debug_stats["no_safe_action_steps"]))

    seed_count = len(validation_seeds)
    mode_delta_sum = sum(dqn - base for dqn, base in zip(dqn_mode_changes, baseline_mode_changes, strict=True))
    cancel_delta_sum = sum(
        dqn - base for dqn, base in zip(dqn_lo_cancellations, baseline_lo_cancellations, strict=True)
    )
    return {
        "validation_seed_count": seed_count,
        "deadline_misses_sum": sum(dqn_deadline_misses),
        "mode_changes_mean": sum(dqn_mode_changes) / seed_count,
        "lo_cancellations_mean": sum(dqn_lo_cancellations) / seed_count,
        "baseline_mode_changes_mean": sum(baseline_mode_changes) / seed_count,
        "baseline_lo_cancellations_mean": sum(baseline_lo_cancellations) / seed_count,
        "dqn_mode_changes_delta_mean": mode_delta_sum / seed_count,
        "dqn_lo_cancellations_delta_mean": cancel_delta_sum / seed_count,
        "accepted_actions_mean": sum(dqn_accepted_actions) / seed_count,
        "noop_actions_mean": sum(dqn_noop_actions) / seed_count,
        "valid_action_count_mean": sum(dqn_valid_action_count_mean) / seed_count,
        "masked_action_count_mean": sum(dqn_masked_action_count_mean) / seed_count,
        "no_safe_action_steps_mean": sum(dqn_no_safe_action_steps) / seed_count,
        "reward_mean": sum(dqn_total_reward) / seed_count,
    }


def _is_better_validation_row(
    *,
    candidate_row: dict[str, int | float],
    best_row: dict[str, int | float] | None,
    save_best_by: str,
) -> bool:
    """判断候选验证结果是否优于当前 best。"""

    if int(candidate_row["deadline_misses_sum"]) != 0:
        return False
    if best_row is None:
        return True
    if int(best_row["deadline_misses_sum"]) != 0:
        return True
    metric_field = {
        "lo_cancellations": "lo_cancellations_mean",
        "mode_changes": "mode_changes_mean",
        "reward": "reward_mean",
    }[save_best_by]
    return float(candidate_row[metric_field]) < float(best_row[metric_field])


def build_parser() -> argparse.ArgumentParser:
    """构建正式训练 CLI 的命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--min-replay-size", type=int, default=500)
    parser.add_argument("--hidden-layers", type=str, default="128,128")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument("--target-update-frequency", type=int, default=None)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--workload", choices=["small", "rtss11", "automotive"], default="small")
    parser.add_argument("--scenario", choices=["nominal", "stress"], default="stress")
    parser.add_argument("--total-util", type=float, default=0.65)
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--cf", type=float, default=2.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--require-schedulable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hi-overrun-prob", type=float, default=0.05)
    parser.add_argument("--lo-overrun-prob", type=float, default=0.10)
    parser.add_argument("--lo-overrun-factor", type=float, default=1.5)
    parser.add_argument("--log-train-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-every", type=int, default=0)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--train-seed-mode", choices=["fixed", "per-episode", "cycle"], default="fixed")
    parser.add_argument("--train-seeds", type=str, default="")
    parser.add_argument("--scenario-seed-offset", type=int, default=100000)
    parser.add_argument("--validation-seeds", type=str, default="100:129")
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--validation-end-time", type=int, default=10000)
    parser.add_argument("--save-best-by", choices=["lo_cancellations", "mode_changes", "reward"], default="lo_cancellations")
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
    """运行正式 DQN 训练并产出完整目录结构。"""

    args = build_parser().parse_args()
    if args.trace_every < 0:
        raise ValueError("--trace-every 必须为非负整数")
    if args.trace_every > 0 and args.trace_dir is None:
        raise ValueError("设置 --trace-every 时必须同时提供 --trace-dir")
    if args.validate_every < 0:
        raise ValueError("--validate-every 必须为非负整数")
    if args.target_update_frequency is not None:
        args.target_update_freq = int(args.target_update_frequency)

    experiment_config = _build_experiment_config(args)
    train_seed_candidates = _parse_seed_spec(args.train_seeds)
    episode_seed_schedule = _build_episode_seed_schedule(
        episodes=args.episodes,
        seed=args.seed,
        mode=args.train_seed_mode,
        train_seeds=train_seed_candidates,
    )
    validation_seeds = _parse_seed_spec(args.validation_seeds)

    hidden_layers = _parse_hidden_layers(args.hidden_layers)
    config = DqnConfig(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        target_update_freq=args.target_update_freq,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        hidden_layers=hidden_layers,
        seed=args.seed,
    )

    initial_seed = episode_seed_schedule[0]
    initial_bundle = resolve_experiment_bundle(experiment_config, initial_seed)
    initial_env = build_env_from_experiment_config(
        experiment_config,
        seed=initial_seed,
        end_time=args.end_time,
        agent_period=args.agent_period,
        semantics=RuntimeSemantics.AMC_PLUS,
        reward_mode=args.reward_mode,
        action_space=args.action_space,
        budget_increase_ratio=args.budget_increase_ratio,
        budget_decrease_ratio=args.budget_decrease_ratio,
        include_explicit_noop=args.include_explicit_noop,
    )
    initial_obs = initial_env.reset(seed=initial_seed)
    agent = DqnBudgetAgent(
        observation_dim=len(initial_obs.state_vector),
        action_dim=initial_env.action_space_size,
        config=config,
        hidden_layers=hidden_layers,
    )

    if args.output_dir is None:
        output_dir = (
            Path(f"outputs/dqn_rtss11/u{int(round(args.total_util * 1000)):03d}_seed{args.seed}")
            if args.workload == "rtss11"
            else Path("outputs/dqn_amc")
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    if args.checkpoint > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, int | float | str | bool]] = []
    train_metric_rows: list[dict[str, int | float | str]] = []
    action_hist_rows: list[dict[str, int]] = []
    validation_rows: list[dict[str, int | float]] = []
    best_validation_row: dict[str, int | float] | None = None
    model_best_path = output_dir / "model_best.pt"

    global_step = 0
    for episode in range(args.episodes):
        episode_seed = episode_seed_schedule[episode]
        env = build_env_from_experiment_config(
            experiment_config,
            seed=episode_seed,
            end_time=args.end_time,
            agent_period=args.agent_period,
            semantics=RuntimeSemantics.AMC_PLUS,
            reward_mode=args.reward_mode,
            action_space=args.action_space,
            budget_increase_ratio=args.budget_increase_ratio,
            budget_decrease_ratio=args.budget_decrease_ratio,
            include_explicit_noop=args.include_explicit_noop,
        )
        bundle = resolve_experiment_bundle(experiment_config, episode_seed)
        obs = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0.0
        episode_losses: list[float] = []
        episode_accepted_actions = 0
        episode_rejected_actions = 0
        episode_noop_actions = 0
        reward_job_start_sum = 0.0
        reward_lo_overrun_sum = 0.0
        reward_hi_overrun_sum = 0.0
        reward_mode_change_sum = 0.0
        reward_lo_cancellation_sum = 0.0
        reward_deadline_miss_sum = 0.0
        episode_action_hist: dict[int, dict[str, int]] = defaultdict(lambda: {"count": 0, "accepted": 0, "rejected": 0})
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
                training=True,
            )
            result = env.step(action_id)
            if action_id is not None and not bool(result.info.get("selected_action_was_mask_valid", False)):
                raise RuntimeError(f"检测到动作选择与 mask 语义不一致，episode={episode}, action_id={action_id}")
            next_mask = (
                env.valid_action_mask()
                if not result.done
                else tuple(False for _ in range(env.action_space_size))
            )

            loss: float | None = None
            if action_id is not None:
                transition = Transition(
                    state=obs.state_vector,
                    action_id=action_id,
                    reward=result.reward,
                    next_state=result.observation.state_vector,
                    done=result.done,
                    valid_action_mask=tuple(mask),
                    next_valid_action_mask=tuple(next_mask),
                )
                agent.remember(transition)
                loss = agent.optimize_one_step()
                if loss is not None:
                    episode_losses.append(loss)
                episode_action_hist[action_id]["count"] += 1
                if bool(result.info.get("accepted")):
                    episode_action_hist[action_id]["accepted"] += 1
                else:
                    episode_action_hist[action_id]["rejected"] += 1

            accepted = bool(result.info.get("accepted"))
            rejected = action_id is not None and not accepted
            noop = action_id is None
            if accepted:
                episode_accepted_actions += 1
            if rejected:
                episode_rejected_actions += 1
            if noop:
                episode_noop_actions += 1

            episode_reward += result.reward
            reward_job_start_sum += float(result.info.get("step_reward_job_start", 0.0))
            reward_lo_overrun_sum += float(result.info.get("step_reward_lo_overrun", 0.0))
            reward_hi_overrun_sum += float(result.info.get("step_reward_hi_overrun", 0.0))
            reward_mode_change_sum += float(result.info.get("step_reward_mode_change", 0.0))
            reward_lo_cancellation_sum += float(result.info.get("step_reward_lo_cancellation", 0.0))
            reward_deadline_miss_sum += float(result.info.get("step_reward_deadline_miss", 0.0))
            step_rows.append(
                {
                    "episode": episode,
                    "step": global_step,
                    "sim_time": int(result.info.get("time", 0)),
                    "reward": float(result.reward),
                    "episode_reward": float(episode_reward),
                    "total_reward": float(episode_reward),
                    "loss": "" if loss is None else loss,
                    "epsilon": agent.current_epsilon,
                    "action_id": "" if action_id is None else action_id,
                    "accepted": accepted,
                    "rejected": rejected,
                    "reject_reason": "no_valid_action" if action_id is None else str(result.info.get("reject_reason", "")),
                    "valid_action_count": valid_action_count,
                    "masked_action_count": masked_action_count,
                    "noop_due_to_no_valid_action": action_id is None,
                    "mode_changes": int(result.info.get("mode_changes", 0)),
                    "lo_cancellations": int(result.info.get("lo_cancellations", 0)),
                    "deadline_misses": int(result.info.get("deadline_misses", 0)),
                    "step_reward_total": float(result.info.get("step_reward_total", 0.0)),
                    "step_reward_job_start": float(result.info.get("step_reward_job_start", 0.0)),
                    "step_reward_lo_overrun": float(result.info.get("step_reward_lo_overrun", 0.0)),
                    "step_reward_hi_overrun": float(result.info.get("step_reward_hi_overrun", 0.0)),
                    "step_reward_mode_change": float(result.info.get("step_reward_mode_change", 0.0)),
                    "step_reward_lo_cancellation": float(result.info.get("step_reward_lo_cancellation", 0.0)),
                    "step_reward_deadline_miss": float(result.info.get("step_reward_deadline_miss", 0.0)),
                    "workload": args.workload,
                    "total_util": args.total_util,
                    "num_tasks": args.num_tasks,
                    "cf": args.cf,
                    "cp": args.cp,
                    "taskset_seed": bundle.taskset_seed if bundle.taskset_seed is not None else episode_seed,
                    "scenario_seed": bundle.scenario_seed if bundle.scenario_seed is not None else episode_seed,
                    "require_schedulable": args.require_schedulable,
                }
            )
            obs = result.observation
            done = result.done
            last_info = result.info
            global_step += 1

        debug_stats = env.debug_statistics()
        if int(debug_stats["selected_invalid_mask_actions"]) > 0:
            raise RuntimeError(
                f"selected_invalid_mask_actions 必须为 0，episode={episode}, value={debug_stats['selected_invalid_mask_actions']}"
            )

        loss_mean = sum(episode_losses) / len(episode_losses) if episode_losses else ""
        loss_last: float | str = episode_losses[-1] if episode_losses else ""
        taskset_seed = bundle.taskset_seed if bundle.taskset_seed is not None else episode_seed
        scenario_seed = bundle.scenario_seed if bundle.scenario_seed is not None else episode_seed
        train_metric_rows.append(
            {
                "episode": episode,
                "episode_seed": episode_seed,
                "taskset_seed": taskset_seed,
                "scenario_seed": scenario_seed,
                "total_util": args.total_util,
                "num_tasks": args.num_tasks,
                "taskset_fingerprint": bundle.taskset_fingerprint or "",
                "steps": episode_accepted_actions + episode_rejected_actions + episode_noop_actions,
                "total_reward": episode_reward,
                "epsilon": agent.current_epsilon,
                "loss_mean": loss_mean,
                "loss_last": loss_last,
                "accepted_actions": episode_accepted_actions,
                "rejected_actions": episode_rejected_actions,
                "noop_actions": episode_noop_actions,
                "safety_checked_actions": int(debug_stats["safety_checked_actions"]),
                "selected_invalid_mask_actions": int(debug_stats["selected_invalid_mask_actions"]),
                "action_space_type": str(debug_stats["action_space_type"]),
                "action_count": int(debug_stats["action_count"]),
                "budget_increase_ratio": float(debug_stats["budget_increase_ratio"]),
                "budget_decrease_ratio": float(debug_stats["budget_decrease_ratio"]),
                "valid_action_count_mean": float(debug_stats["valid_action_count_mean"]),
                "masked_action_count_mean": float(debug_stats["masked_action_count_mean"]),
                "no_safe_action_steps": int(debug_stats["no_safe_action_steps"]),
                "mode_changes": int(last_info.get("mode_changes", 0)),
                "lo_cancellations": int(last_info.get("lo_cancellations", 0)),
                "deadline_misses": int(last_info.get("deadline_misses", 0)),
                "job_starts": int(env._monitor.job_start_count),
                "lo_overruns": int(env._monitor.lo_overrun_count),
                "hi_overruns": int(env._monitor.hi_overrun_count),
                "reward_job_start_sum": reward_job_start_sum,
                "reward_lo_overrun_sum": reward_lo_overrun_sum,
                "reward_hi_overrun_sum": reward_hi_overrun_sum,
                "reward_mode_change_sum": reward_mode_change_sum,
                "reward_lo_cancellation_sum": reward_lo_cancellation_sum,
                "reward_deadline_miss_sum": reward_deadline_miss_sum,
            }
        )
        for action_id in sorted(episode_action_hist):
            action_hist = episode_action_hist[action_id]
            action_hist_rows.append(
                {
                    "episode": episode,
                    "action_id": action_id,
                    "count": action_hist["count"],
                    "accepted_count": action_hist["accepted"],
                    "rejected_count": action_hist["rejected"],
                }
            )

        if args.trace_every > 0 and (episode + 1) % args.trace_every == 0 and args.trace_dir is not None:
            runtime_result = env._engine.finish() if env._engine is not None else SimulationResult()
            _write_jsonl(args.trace_dir / f"episode_{episode + 1:04d}_action_log.jsonl", env.action_log)
            _write_jsonl(args.trace_dir / f"episode_{episode + 1:04d}_mask_log.jsonl", env.mask_log)
            _write_jsonl(
                args.trace_dir / f"episode_{episode + 1:04d}_runtime_trace.jsonl",
                _trace_rows_from_runtime(runtime_result),
            )

        if args.checkpoint > 0 and (episode + 1) % args.checkpoint == 0:
            agent.save(checkpoint_dir / f"model_episode_{episode + 1:04d}.pt")

        if args.validate_every > 0 and validation_seeds and (episode + 1) % args.validate_every == 0:
            validation_row = _run_validation(
                agent=agent,
                experiment_config=experiment_config,
                validation_seeds=validation_seeds,
                validation_end_time=args.validation_end_time,
                agent_period=args.agent_period,
                reward_mode=args.reward_mode,
                action_space=args.action_space,
                budget_increase_ratio=args.budget_increase_ratio,
                budget_decrease_ratio=args.budget_decrease_ratio,
                include_explicit_noop=args.include_explicit_noop,
            )
            validation_row["episode"] = episode + 1
            validation_rows.append(validation_row)
            if _is_better_validation_row(
                candidate_row=validation_row,
                best_row=best_validation_row,
                save_best_by=args.save_best_by,
            ):
                best_validation_row = validation_row
                agent.save(model_best_path)

    train_log_path = output_dir / "train_log.csv"
    train_metrics_path = output_dir / "train_metrics.csv"
    action_hist_path = output_dir / "train_action_histogram.csv"
    validation_metrics_path = output_dir / "validation_metrics.csv"
    model_path = output_dir / "model_final.pt"
    config_path = output_dir / "config.json"

    step_fieldnames = [
        "episode",
        "step",
        "sim_time",
        "reward",
        "episode_reward",
        "total_reward",
        "loss",
        "epsilon",
        "action_id",
        "accepted",
        "rejected",
        "reject_reason",
        "valid_action_count",
        "masked_action_count",
        "noop_due_to_no_valid_action",
        "mode_changes",
        "lo_cancellations",
        "deadline_misses",
        "step_reward_total",
        "step_reward_job_start",
        "step_reward_lo_overrun",
        "step_reward_hi_overrun",
        "step_reward_mode_change",
        "step_reward_lo_cancellation",
        "step_reward_deadline_miss",
        "workload",
        "total_util",
        "num_tasks",
        "cf",
        "cp",
        "taskset_seed",
        "scenario_seed",
        "require_schedulable",
    ]
    with train_log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_fieldnames)
        writer.writeheader()
        writer.writerows(step_rows)

    if args.log_train_metrics:
        metric_fieldnames = [
            "episode",
            "episode_seed",
            "taskset_seed",
            "scenario_seed",
            "total_util",
            "num_tasks",
            "taskset_fingerprint",
            "steps",
            "total_reward",
            "epsilon",
            "loss_mean",
            "loss_last",
            "accepted_actions",
            "rejected_actions",
            "noop_actions",
            "safety_checked_actions",
            "selected_invalid_mask_actions",
            "action_space_type",
            "action_count",
            "budget_increase_ratio",
            "budget_decrease_ratio",
            "valid_action_count_mean",
            "masked_action_count_mean",
            "no_safe_action_steps",
            "mode_changes",
            "lo_cancellations",
            "deadline_misses",
            "job_starts",
            "lo_overruns",
            "hi_overruns",
            "reward_job_start_sum",
            "reward_lo_overrun_sum",
            "reward_hi_overrun_sum",
            "reward_mode_change_sum",
            "reward_lo_cancellation_sum",
            "reward_deadline_miss_sum",
        ]
        with train_metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
            writer.writeheader()
            writer.writerows(train_metric_rows)

    action_hist_fieldnames = ["episode", "action_id", "count", "accepted_count", "rejected_count"]
    with action_hist_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=action_hist_fieldnames)
        writer.writeheader()
        writer.writerows(action_hist_rows)

    if validation_rows:
        validation_fieldnames = [
            "episode",
            "validation_seed_count",
            "deadline_misses_sum",
            "mode_changes_mean",
            "lo_cancellations_mean",
            "baseline_mode_changes_mean",
            "baseline_lo_cancellations_mean",
            "dqn_mode_changes_delta_mean",
            "dqn_lo_cancellations_delta_mean",
            "accepted_actions_mean",
            "noop_actions_mean",
            "valid_action_count_mean",
            "masked_action_count_mean",
            "no_safe_action_steps_mean",
            "reward_mean",
        ]
        with validation_metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=validation_fieldnames)
            writer.writeheader()
            writer.writerows(validation_rows)

    agent.save(model_path)
    config_payload = {
        "dqn_config": asdict(config),
        "workload": args.workload,
        "scenario": args.scenario,
        "scenario_name": initial_bundle.scenario.name,
        "total_util": args.total_util,
        "num_tasks": args.num_tasks,
        "cf": args.cf,
        "cp": args.cp,
        "require_schedulable": args.require_schedulable,
        "hi_overrun_prob": args.hi_overrun_prob,
        "lo_overrun_prob": args.lo_overrun_prob,
        "lo_overrun_factor": args.lo_overrun_factor,
        "train_seed_mode": args.train_seed_mode,
        "train_seeds": episode_seed_schedule,
        "scenario_seed_offset": args.scenario_seed_offset,
        "validation_seeds": validation_seeds,
        "validate_every": args.validate_every,
        "validation_end_time": args.validation_end_time,
        "save_best_by": args.save_best_by,
        "reward_mode": args.reward_mode,
        "action_space": args.action_space,
        "budget_increase_ratio": args.budget_increase_ratio,
        "budget_decrease_ratio": args.budget_decrease_ratio,
        "include_explicit_noop": args.include_explicit_noop,
        "log_train_metrics": args.log_train_metrics,
        "trace_every": args.trace_every,
        "normalization_bounds": {
            task_name: {"min_cost": bound.min_cost, "max_cost": bound.max_cost}
            for task_name, bound in initial_bundle.normalization_bounds.items()
        },
        "action_space_size": initial_env.action_space_size,
        "observation_dim": len(initial_obs.state_vector),
        "tasks": _serialize_tasks(list(initial_bundle.ordered_tasks)),
        "runtime_config": {
            "end_time": args.end_time,
            "agent_period": args.agent_period,
            "semantics": RuntimeSemantics.AMC_PLUS.value,
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
