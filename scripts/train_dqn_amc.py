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
from amc_py.rl.reward_config import available_reward_modes, load_reward_mode_config
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


def _get_noop_action_id(env) -> int | None:
    """从环境动作空间中解析显式 noop 的 action_id。"""

    for action in env._actions:  # noqa: SLF001
        if bool(getattr(action, "is_noop", False)):
            return int(action.action_id)
    return None


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
            fixed_taskset_seed=args.fixed_taskset_seed,
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
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    mask_detail_mode: str,
    baseline_cache: dict[str, float] | None = None,
) -> tuple[dict[str, int | float], dict[str, float], bool]:
    """在验证集上评估当前 agent，并返回聚合指标。"""

    # 每个列表都保存“按 seed 的单次验证结果”，最后再做均值聚合写入 validation CSV。
    dqn_mode_changes: list[int] = []
    dqn_lo_cancellations: list[int] = []
    dqn_deadline_misses: list[int] = []
    dqn_accepted_actions: list[int] = []
    dqn_rejected_actions: list[int] = []
    dqn_step_counts: list[int] = []
    dqn_selected_action_counts: list[int] = []
    dqn_noop_actions: list[int] = []
    dqn_explicit_noop_actions: list[int] = []
    dqn_noop_action_rate: list[float] = []
    dqn_explicit_noop_action_rate: list[float] = []
    dqn_accepted_action_rate: list[float] = []
    dqn_rejected_action_rate: list[float] = []
    dqn_valid_action_count_mean: list[float] = []
    dqn_masked_action_count_mean: list[float] = []
    dqn_total_reward: list[float] = []
    dqn_no_safe_action_steps: list[int] = []
    baseline_mode_changes: list[int] = []
    baseline_lo_cancellations: list[int] = []
    used_baseline_cache = baseline_cache is not None

    if baseline_cache is None:
        for seed in validation_seeds:
            bundle = resolve_experiment_bundle(experiment_config, seed)
            baseline_result = simulate_ordered_taskset_event_driven(
                ordered_tasks=list(bundle.ordered_tasks),
                scenario=bundle.scenario,
                config=RuntimeConfig(end_time=validation_end_time, semantics=RuntimeSemantics.AMC_PLUS),
            )
            baseline_mode_changes.append(baseline_result.mode_change_count())
            baseline_lo_cancellations.append(baseline_result.lo_job_cancellation_count())
        baseline_cache = {
            "baseline_mode_changes_mean": sum(baseline_mode_changes) / len(validation_seeds),
            "baseline_lo_cancellations_mean": sum(baseline_lo_cancellations) / len(validation_seeds),
        }

    for seed in validation_seeds:
        bundle = resolve_experiment_bundle(experiment_config, seed)

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
            budget_floor_ratio=budget_floor_ratio,
            forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
            mask_detail_mode=mask_detail_mode,
        )
        obs = env.reset(seed=seed)
        done = False
        # 统一阶段 2 统计口径：
        # - step_count：决策步总数（每个 while 循环 +1）；
        # - selected_action_count：action_id 非空的步数；
        # - accepted/rejected：仅统计 action_id 非空时是否被接受；
        # - noop_actions：任何 is_noop=True 的步；
        # - explicit_noop_actions：仅 is_explicit_noop_action=True 的步。
        accepted_actions = 0
        rejected_actions = 0
        step_count = 0
        selected_action_count = 0
        noop_actions = 0
        explicit_noop_actions = 0
        total_reward = 0.0
        last_info: dict[str, int | float | str | bool | None] = {
            "mode_changes": 0,
            "lo_cancellations": 0,
            "deadline_misses": 0,
        }

        while not done:
            # 每进入一次循环就代表完成一个 agent 决策步，该值是所有 rate 的唯一分母。
            step_count += 1
            mask = env.valid_action_mask()
            action_id = agent.select_action_id(
                obs.state_vector,
                valid_action_mask=mask,
                training=False,
            )
            # selected_action_count 仅反映“是否给出离散动作编号”，不关心动作是否被接受。
            selected_action_count += int(action_id is not None)
            result = env.step(action_id)
            total_reward += result.reward
            is_noop = bool(result.info.get("is_noop", False))
            if is_noop:
                noop_actions += 1
                if bool(result.info.get("is_explicit_noop_action", False)):
                    explicit_noop_actions += 1
            if action_id is not None:
                if bool(result.info.get("accepted")):
                    accepted_actions += 1
                else:
                    rejected_actions += 1
            obs = result.observation
            done = result.done
            last_info = result.info

        dqn_mode_changes.append(int(last_info.get("mode_changes", 0)))
        dqn_lo_cancellations.append(int(last_info.get("lo_cancellations", 0)))
        dqn_deadline_misses.append(int(last_info.get("deadline_misses", 0)))
        dqn_step_counts.append(step_count)
        dqn_selected_action_counts.append(selected_action_count)
        dqn_accepted_actions.append(accepted_actions)
        dqn_rejected_actions.append(rejected_actions)
        dqn_noop_actions.append(noop_actions)
        dqn_explicit_noop_actions.append(explicit_noop_actions)
        # 阶段 2：所有 rate 一律用 step_count 做分母，避免 explicit noop 在 accepted/noop 双重计数。
        # 例：explicit noop 同时满足 accepted=True 与 is_noop=True，
        # 如果分母错误写成 accepted+rejected+noop，会把同一步算两次。
        dqn_noop_action_rate.append((noop_actions / step_count) if step_count > 0 else 0.0)
        dqn_explicit_noop_action_rate.append((explicit_noop_actions / step_count) if step_count > 0 else 0.0)
        dqn_accepted_action_rate.append((accepted_actions / step_count) if step_count > 0 else 0.0)
        dqn_rejected_action_rate.append((rejected_actions / step_count) if step_count > 0 else 0.0)
        dqn_total_reward.append(total_reward)
        debug_stats = env.debug_statistics()
        dqn_valid_action_count_mean.append(float(debug_stats["valid_action_count_mean"]))
        dqn_masked_action_count_mean.append(float(debug_stats["masked_action_count_mean"]))
        dqn_no_safe_action_steps.append(int(debug_stats["no_safe_action_steps"]))

    seed_count = len(validation_seeds)
    baseline_mode_changes_mean = float(baseline_cache["baseline_mode_changes_mean"])
    baseline_lo_cancellations_mean = float(baseline_cache["baseline_lo_cancellations_mean"])
    mode_delta_sum = sum(dqn - baseline_mode_changes_mean for dqn in dqn_mode_changes)
    cancel_delta_sum = sum(dqn - baseline_lo_cancellations_mean for dqn in dqn_lo_cancellations)
    return ({
        "validation_seed_count": seed_count,
        "deadline_misses_sum": sum(dqn_deadline_misses),
        "mode_changes_mean": sum(dqn_mode_changes) / seed_count,
        "lo_cancellations_mean": sum(dqn_lo_cancellations) / seed_count,
        "baseline_mode_changes_mean": baseline_mode_changes_mean,
        "baseline_lo_cancellations_mean": baseline_lo_cancellations_mean,
        "dqn_mode_changes_delta_mean": mode_delta_sum / seed_count,
        "dqn_lo_cancellations_delta_mean": cancel_delta_sum / seed_count,
        "accepted_actions_mean": sum(dqn_accepted_actions) / seed_count,
        "rejected_actions_mean": sum(dqn_rejected_actions) / seed_count,
        "step_count_mean": sum(dqn_step_counts) / seed_count,
        "selected_action_count_mean": sum(dqn_selected_action_counts) / seed_count,
        "noop_actions_mean": sum(dqn_noop_actions) / seed_count,
        "explicit_noop_actions_mean": sum(dqn_explicit_noop_actions) / seed_count,
        "noop_action_rate_mean": sum(dqn_noop_action_rate) / seed_count,
        "explicit_noop_action_rate_mean": sum(dqn_explicit_noop_action_rate) / seed_count,
        "accepted_action_rate_mean": sum(dqn_accepted_action_rate) / seed_count,
        "rejected_action_rate_mean": sum(dqn_rejected_action_rate) / seed_count,
        "valid_action_count_mean": sum(dqn_valid_action_count_mean) / seed_count,
        "masked_action_count_mean": sum(dqn_masked_action_count_mean) / seed_count,
        "no_safe_action_steps_mean": sum(dqn_no_safe_action_steps) / seed_count,
        "reward_mean": sum(dqn_total_reward) / seed_count,
    }, baseline_cache, used_baseline_cache)


def _is_better_validation_row(
    *,
    candidate_row: dict[str, int | float],
    best_row: dict[str, int | float] | None,
    save_best_by: str,
    relative_score_alpha: float = 1.0,
    require_better_than_baseline_for_best: bool = False,
) -> bool:
    """判断候选验证结果是否优于当前 best。"""

    if int(candidate_row["deadline_misses_sum"]) != 0:
        return False
    if best_row is None:
        return True
    if int(best_row["deadline_misses_sum"]) != 0:
        return True
    if save_best_by == "reward":
        # 阶段 3：reward 指标方向应为“越大越好”，与 cancellation/mode-change 相反。
        return float(candidate_row["reward_mean"]) > float(best_row["reward_mean"])
    if save_best_by == "relative_score":
        # relative_score 越小越好，<0 表示综合优于 baseline。
        # 本轮要求：即便 relative_score>=0，也要保留“验证集里最好的那个”checkpoint。
        _ = (relative_score_alpha, require_better_than_baseline_for_best)
        candidate_score = float(candidate_row["relative_score"])
        return candidate_score < float(best_row["relative_score"])
    metric_field = {
        "lo_cancellations": "lo_cancellations_mean",
        "mode_changes": "mode_changes_mean",
    }[save_best_by]
    return float(candidate_row[metric_field]) < float(best_row[metric_field])


def _build_validation_unified_summary_rows(
    validation_rows: list[dict[str, int | float]],
) -> list[dict[str, int | float | str]]:
    """把训练期 validation 指标转换为阶段 0/1 的统一 summary 口径。"""

    summary_rows: list[dict[str, int | float | str]] = []
    for row in validation_rows:
        baseline_mode_changes_mean = float(row["baseline_mode_changes_mean"])
        baseline_lo_cancellations_mean = float(row["baseline_lo_cancellations_mean"])
        dqn_mode_changes_mean = float(row["mode_changes_mean"])
        dqn_lo_cancellations_mean = float(row["lo_cancellations_mean"])
        mode_change_ratio: float | str
        lo_cancellation_ratio: float | str
        if dqn_mode_changes_mean == 0.0:
            mode_change_ratio = "inf" if baseline_mode_changes_mean > 0.0 else "nan"
        else:
            mode_change_ratio = baseline_mode_changes_mean / dqn_mode_changes_mean
        if dqn_lo_cancellations_mean == 0.0:
            lo_cancellation_ratio = "inf" if baseline_lo_cancellations_mean > 0.0 else "nan"
        else:
            lo_cancellation_ratio = baseline_lo_cancellations_mean / dqn_lo_cancellations_mean

        summary_rows.append(
            {
                "episode": int(row["episode"]),
                "validation_seed_count": int(row["validation_seed_count"]),
                "baseline_mode_changes_mean": baseline_mode_changes_mean,
                "baseline_lo_cancellations_mean": baseline_lo_cancellations_mean,
                "dqn_mode_changes_mean": dqn_mode_changes_mean,
                "dqn_lo_cancellations_mean": dqn_lo_cancellations_mean,
                # 这三列用于与文档要求一致地直观看到“相对 baseline 的差值与综合得分”。
                "relative_score": float(row["relative_score"]),
                "relative_score_alpha": float(row["relative_score_alpha"]),
                "relative_delta_mode_changes": float(row["relative_delta_mode_changes"]),
                "relative_delta_lo_cancellations": float(row["relative_delta_lo_cancellations"]),
                "is_better_than_baseline": bool(row["is_better_than_baseline"]),
                "mode_changes_delta_vs_baseline": float(row["dqn_mode_changes_delta_mean"]),
                "lo_cancellations_delta_vs_baseline": float(row["dqn_lo_cancellations_delta_mean"]),
                "mode_change_ratio": mode_change_ratio,
                "lo_cancellation_ratio": lo_cancellation_ratio,
                "accepted_action_count_mean": float(row["accepted_actions_mean"]),
                "rejected_action_count_mean": float(row["rejected_actions_mean"]),
                "noop_action_count_mean": float(row["noop_actions_mean"]),
                "noop_action_rate_mean": float(row["noop_action_rate_mean"]),
                "explicit_noop_action_rate_mean": float(row["explicit_noop_action_rate_mean"]),
                "accepted_action_rate_mean": float(row["accepted_action_rate_mean"]),
                "rejected_action_rate_mean": float(row["rejected_action_rate_mean"]),
                "masked_action_count_mean": float(row["masked_action_count_mean"]),
                "valid_action_count_mean": float(row["valid_action_count_mean"]),
            }
        )
    return summary_rows


def build_parser() -> argparse.ArgumentParser:
    """构建正式训练 CLI 的命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--network-seed", type=int, default=None)
    parser.add_argument("--exploration-seed", type=int, default=None)
    parser.add_argument("--replay-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--min-replay-size", type=int, default=500)
    parser.add_argument("--hidden-layers", type=str, default="128,128")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    # 稳定性修改 C：默认 target network 更新频率改为 5（优化步为单位）。
    # 该值可继续通过 CLI 覆盖，用于后续对照实验。
    parser.add_argument("--target-update-freq", type=int, default=5)
    parser.add_argument("--target-update-frequency", type=int, default=None)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=10.0,
        help="DQN 梯度裁剪阈值（L2 norm），用于限制反向传播后的梯度范数。",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=5000)
    parser.add_argument(
        "--noop-exploration-prob",
        type=float,
        default=0.0,
        help=(
            "During epsilon exploration, if explicit noop is valid, choose noop "
            "with this probability before sampling other valid actions."
        ),
    )
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
    parser.add_argument(
        "--fixed-taskset-seed",
        type=int,
        default=None,
        help=(
            "如果设置该参数，RTSS11 任务集生成固定使用该 seed；"
            "每个 episode 变化的 seed 仅用于 scenario 生成。"
        ),
    )
    parser.add_argument("--train-seeds", type=str, default="")
    parser.add_argument("--scenario-seed-offset", type=int, default=100000)
    parser.add_argument("--validation-seeds", type=str, default="100:129")
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--validation-end-time", type=int, default=10000)
    parser.add_argument(
        "--save-best-by",
        choices=["lo_cancellations", "mode_changes", "reward", "relative_score"],
        default="lo_cancellations",
    )
    parser.add_argument(
        "--reward-mode",
        choices=list(available_reward_modes()),
        default="mendes",
    )
    parser.add_argument(
        "--relative-score-alpha",
        type=float,
        default=1.0,
        help="relative_score 中 mode_changes 差值的权重 alpha。",
    )
    parser.add_argument(
        "--require-better-than-baseline-for-best",
        action="store_true",
        help="仅当 relative_score < 0 时才允许刷新 model_best.pt。",
    )
    parser.add_argument("--action-space", choices=["triple", "pair", "single"], default="triple")
    parser.add_argument("--budget-increase-ratio", type=float, default=0.10)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.05)
    parser.add_argument("--include-explicit-noop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--budget-floor-ratio",
        type=float,
        default=0.0,
        help=(
            "Reject budget actions that would reduce any task budget below "
            "initial_budget * this ratio. 0 disables the floor."
        ),
    )
    parser.add_argument(
        "--forbid-decreasing-hi-budgets",
        action="store_true",
        # 阶段 3：开启后会在 action mask 中屏蔽“decrease 命中 HI 任务”的动作。
        # 这是硬约束，不是软惩罚，不依赖 reward 权重调节。
        help="If set, action masks reject budget actions whose decrease tasks include any HI-criticality task.",
    )
    parser.add_argument("--mask-detail-mode", choices=["minimal", "full"], default="minimal")
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
    if args.budget_floor_ratio < 0.0 or args.budget_floor_ratio > 1.0:
        raise ValueError("--budget-floor-ratio must be in [0, 1]")
    reward_mode_config = load_reward_mode_config(args.reward_mode)

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
    # 阶段 0：显式拆分 DQN 侧随机源，保证同配置可复现且可追踪。
    # - network_seed:     只用于网络初始化；
    # - exploration_seed: 只用于 epsilon-greedy 探索；
    # - replay_seed:      只用于 replay 抽样。
    network_seed = args.seed if args.network_seed is None else int(args.network_seed)
    exploration_seed = args.seed if args.exploration_seed is None else int(args.exploration_seed)
    replay_seed = args.seed if args.replay_seed is None else int(args.replay_seed)
    config = DqnConfig(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        target_update_freq=args.target_update_freq,
        grad_clip_norm=args.grad_clip_norm,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        noop_exploration_prob=args.noop_exploration_prob,
        hidden_layers=hidden_layers,
        seed=args.seed,
        network_seed=network_seed,
        exploration_seed=exploration_seed,
        replay_seed=replay_seed,
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
        budget_floor_ratio=args.budget_floor_ratio,
        forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
        mask_detail_mode=args.mask_detail_mode,
    )
    initial_obs = initial_env.reset(seed=initial_seed)
    agent = DqnBudgetAgent(
        observation_dim=len(initial_obs.state_vector),
        action_dim=initial_env.action_space_size,
        config=config,
        noop_action_id=_get_noop_action_id(initial_env),
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
    best_model_metadata_path = output_dir / "best_model_metadata.json"
    best_model_saved = False
    baseline_validation_cache: dict[str, float] | None = None

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
            budget_floor_ratio=args.budget_floor_ratio,
            forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
            mask_detail_mode=args.mask_detail_mode,
        )
        bundle = resolve_experiment_bundle(experiment_config, episode_seed)
        obs = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0.0
        episode_losses: list[float] = []
        episode_accepted_actions = 0
        episode_rejected_actions = 0
        episode_step_count = 0
        episode_selected_action_count = 0
        episode_noop_actions = 0
        episode_explicit_noop_actions = 0
        reward_job_start_sum = 0.0
        reward_lo_overrun_sum = 0.0
        reward_hi_overrun_sum = 0.0
        reward_mode_change_sum = 0.0
        reward_lo_cancellation_sum = 0.0
        reward_deadline_miss_sum = 0.0
        reward_paper_sum = 0.0
        reward_noop_bonus_sum = 0.0
        reward_budget_change_penalty_sum = 0.0
        reward_budget_change_norm_sum = 0.0
        reward_budget_drift_penalty_sum = 0.0
        reward_budget_drift_mean_sum = 0.0
        exploration_action_count_before = int(agent.exploration_action_count)
        exploration_noop_action_count_before = int(agent.exploration_noop_action_count)
        episode_action_hist: dict[int, dict[str, int]] = defaultdict(lambda: {"count": 0, "accepted": 0, "rejected": 0})
        last_info: dict[str, int | float | str | bool | None] = {
            "mode_changes": 0,
            "lo_cancellations": 0,
            "deadline_misses": 0,
        }

        while not done:
            # episode_step_count 是训练期指标的主分母，定义为“完成了多少个环境 step”。
            episode_step_count += 1
            mask = env.valid_action_mask()
            valid_action_count = sum(mask)
            masked_action_count = len(mask) - valid_action_count
            action_id = agent.select_action_id(
                obs.state_vector,
                valid_action_mask=mask,
                training=True,
            )
            # selected_action_count 只表示“agent 有没有给出 action_id”，
            # 与是否 accepted/noop 是正交维度。
            episode_selected_action_count += int(action_id is not None)
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
            noop = bool(result.info.get("is_noop", False))
            explicit_noop = bool(result.info.get("is_explicit_noop_action", False))
            # 统计口径说明：
            # - explicit noop 既可能是 accepted，也必须计入 noop；
            # - 因此 rate 分母必须用 step_count，不能用 accepted+rejected+noop。
            if accepted:
                episode_accepted_actions += 1
            if rejected:
                episode_rejected_actions += 1
            if noop:
                episode_noop_actions += 1
            if explicit_noop:
                episode_explicit_noop_actions += 1

            episode_reward += result.reward
            reward_job_start_sum += float(result.info.get("step_reward_job_start", 0.0))
            reward_lo_overrun_sum += float(result.info.get("step_reward_lo_overrun", 0.0))
            reward_hi_overrun_sum += float(result.info.get("step_reward_hi_overrun", 0.0))
            reward_mode_change_sum += float(result.info.get("step_reward_mode_change", 0.0))
            reward_lo_cancellation_sum += float(result.info.get("step_reward_lo_cancellation", 0.0))
            reward_deadline_miss_sum += float(result.info.get("step_reward_deadline_miss", 0.0))
            reward_paper_sum += float(result.info.get("paper_reward", 0.0))
            reward_noop_bonus_sum += float(result.info.get("noop_reward_bonus", 0.0))
            reward_budget_change_penalty_sum += float(result.info.get("budget_change_penalty_value", 0.0))
            reward_budget_change_norm_sum += float(result.info.get("budget_change_norm", 0.0))
            reward_budget_drift_penalty_sum += float(result.info.get("budget_drift_penalty_value", 0.0))
            reward_budget_drift_mean_sum += float(result.info.get("budget_drift_mean", 0.0))
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
                    "is_noop": noop,
                    "is_explicit_noop": explicit_noop,
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
                    "paper_reward": float(result.info.get("paper_reward", 0.0)),
                    "noop_reward_bonus": float(result.info.get("noop_reward_bonus", 0.0)),
                    "budget_change_norm": float(result.info.get("budget_change_norm", 0.0)),
                    "budget_change_penalty_value": float(result.info.get("budget_change_penalty_value", 0.0)),
                    "budget_drift_mean": float(result.info.get("budget_drift_mean", 0.0)),
                    "budget_drift_penalty_value": float(result.info.get("budget_drift_penalty_value", 0.0)),
                    "reward_after_regularization": float(result.info.get("reward_after_regularization", 0.0)),
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
                "network_seed": network_seed,
                "exploration_seed": exploration_seed,
                "replay_seed": replay_seed,
                "total_util": args.total_util,
                "num_tasks": args.num_tasks,
                "taskset_fingerprint": bundle.taskset_fingerprint or "",
                "steps": episode_accepted_actions + episode_rejected_actions + episode_noop_actions,
                # `steps` 保留旧字段以兼容历史脚本；其值可能与 step_count 不同（存在重复口径）。
                # 新分析请优先使用 `step_count`。
                "step_count": episode_step_count,
                "selected_action_count": episode_selected_action_count,
                "total_reward": episode_reward,
                "epsilon": agent.current_epsilon,
                "loss_mean": loss_mean,
                "loss_last": loss_last,
                "accepted_actions": episode_accepted_actions,
                "rejected_actions": episode_rejected_actions,
                "noop_actions": episode_noop_actions,
                "explicit_noop_actions": episode_explicit_noop_actions,
                "noop_action_rate": (
                    episode_noop_actions
                    / episode_step_count
                    if episode_step_count > 0
                    else 0.0
                ),
                # 显式 noop rate 单独输出，便于区分“主动不调预算”和“无动作可用”。
                "explicit_noop_action_rate": (
                    episode_explicit_noop_actions / episode_step_count if episode_step_count > 0 else 0.0
                ),
                "accepted_action_rate": (
                    episode_accepted_actions / episode_step_count if episode_step_count > 0 else 0.0
                ),
                "rejected_action_rate": (
                    episode_rejected_actions / episode_step_count if episode_step_count > 0 else 0.0
                ),
                "safety_checked_actions": int(debug_stats["safety_checked_actions"]),
                "selected_invalid_mask_actions": int(debug_stats["selected_invalid_mask_actions"]),
                "action_space_type": str(debug_stats["action_space_type"]),
                "action_count": int(debug_stats["action_count"]),
                "budget_increase_ratio": float(debug_stats["budget_increase_ratio"]),
                "budget_decrease_ratio": float(debug_stats["budget_decrease_ratio"]),
                "budget_floor_ratio": float(debug_stats["budget_floor_ratio"]),
                "valid_action_count_mean": float(debug_stats["valid_action_count_mean"]),
                "masked_action_count_mean": float(debug_stats["masked_action_count_mean"]),
                "masked_decrease_hi_forbidden_count": int(debug_stats["masked_decrease_hi_forbidden_count"]),
                "masked_decrease_hi_forbidden_rate": float(debug_stats["masked_decrease_hi_forbidden_rate"]),
                "masked_budget_floor_violation_count": int(debug_stats["masked_budget_floor_violation_count"]),
                "masked_budget_floor_violation_rate": float(debug_stats["masked_budget_floor_violation_rate"]),
                "no_safe_action_steps": int(debug_stats["no_safe_action_steps"]),
                "selected_explicit_noop_actions": int(debug_stats["selected_explicit_noop_actions"]),
                "selected_explicit_noop_rate": float(debug_stats["selected_explicit_noop_rate"]),
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
                "reward_paper_sum": reward_paper_sum,
                "reward_noop_bonus_sum": reward_noop_bonus_sum,
                "reward_budget_change_penalty_sum": reward_budget_change_penalty_sum,
                "reward_budget_change_norm_sum": reward_budget_change_norm_sum,
                "reward_budget_drift_penalty_sum": reward_budget_drift_penalty_sum,
                "reward_budget_drift_mean_sum": reward_budget_drift_mean_sum,
                "noop_exploration_prob": args.noop_exploration_prob,
                "exploration_action_count": int(agent.exploration_action_count - exploration_action_count_before),
                "exploration_noop_action_count": int(
                    agent.exploration_noop_action_count - exploration_noop_action_count_before
                ),
                "exploration_noop_action_rate": (
                    float(agent.exploration_noop_action_count - exploration_noop_action_count_before)
                    / float(agent.exploration_action_count - exploration_action_count_before)
                    if int(agent.exploration_action_count - exploration_action_count_before) > 0
                    else 0.0
                ),
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
            validation_row, baseline_validation_cache, used_baseline_cache = _run_validation(
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
                budget_floor_ratio=args.budget_floor_ratio,
                forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
                mask_detail_mode=args.mask_detail_mode,
                baseline_cache=baseline_validation_cache,
            )
            if used_baseline_cache:
                print("Using cached baseline validation metrics")
            validation_row["episode"] = episode + 1
            # 阶段 2：显式记录相对 baseline 的两类 delta，并按 alpha 计算综合分数。
            # 这里严格保持“relative_score 越小越好”的选模口径。
            # 公式：relative_score = delta_lo + alpha * delta_mode
            # 其中：
            # - delta_lo   = dqn_lo_cancellations_mean - baseline_lo_cancellations_mean
            # - delta_mode = dqn_mode_changes_mean    - baseline_mode_changes_mean
            # is_better_than_baseline 仅做标记，不参与是否保存 best 的硬门槛。
            delta_lo = (
                float(validation_row["lo_cancellations_mean"])
                - float(validation_row["baseline_lo_cancellations_mean"])
            )
            delta_mode = float(validation_row["mode_changes_mean"]) - float(
                validation_row["baseline_mode_changes_mean"]
            )
            validation_row["relative_score_alpha"] = args.relative_score_alpha
            validation_row["relative_delta_lo_cancellations"] = delta_lo
            validation_row["relative_delta_mode_changes"] = delta_mode
            validation_row["relative_score"] = delta_lo + args.relative_score_alpha * delta_mode
            validation_row["is_better_than_baseline"] = float(validation_row["relative_score"]) < 0.0
            validation_rows.append(validation_row)
            if _is_better_validation_row(
                candidate_row=validation_row,
                best_row=best_validation_row,
                save_best_by=args.save_best_by,
                relative_score_alpha=args.relative_score_alpha,
                require_better_than_baseline_for_best=args.require_better_than_baseline_for_best,
            ):
                best_validation_row = validation_row
                agent.save(model_best_path)
                best_model_saved = True
            print(
                {
                    "episode": episode + 1,
                    "validation_reward_mean": float(validation_row["reward_mean"]),
                    "train_loss_last": loss_last,
                }
            )

    train_log_path = output_dir / "train_log.csv"
    train_metrics_path = output_dir / "train_metrics.csv"
    action_hist_path = output_dir / "train_action_histogram.csv"
    validation_metrics_path = output_dir / "validation_metrics.csv"
    validation_unified_summary_path = output_dir / "validation_unified_summary.csv"
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
        "is_noop",
        "is_explicit_noop",
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
        "paper_reward",
        "noop_reward_bonus",
        "budget_change_norm",
        "budget_change_penalty_value",
        "budget_drift_mean",
        "budget_drift_penalty_value",
        "reward_after_regularization",
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
            "network_seed",
            "exploration_seed",
            "replay_seed",
            "total_util",
            "num_tasks",
            "taskset_fingerprint",
            "steps",
            "step_count",
            "selected_action_count",
            "total_reward",
            "epsilon",
            "loss_mean",
            "loss_last",
            "accepted_actions",
            "rejected_actions",
            "noop_actions",
            "explicit_noop_actions",
            "noop_action_rate",
            "explicit_noop_action_rate",
            "accepted_action_rate",
            "rejected_action_rate",
            "safety_checked_actions",
            "selected_invalid_mask_actions",
            "action_space_type",
            "action_count",
            "budget_increase_ratio",
            "budget_decrease_ratio",
            "budget_floor_ratio",
            "valid_action_count_mean",
            "masked_action_count_mean",
            "masked_decrease_hi_forbidden_count",
            "masked_decrease_hi_forbidden_rate",
            "masked_budget_floor_violation_count",
            "masked_budget_floor_violation_rate",
            "no_safe_action_steps",
            "selected_explicit_noop_actions",
            "selected_explicit_noop_rate",
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
            "reward_paper_sum",
            "reward_noop_bonus_sum",
            "reward_budget_change_penalty_sum",
            "reward_budget_change_norm_sum",
            "reward_budget_drift_penalty_sum",
            "reward_budget_drift_mean_sum",
            "noop_exploration_prob",
            "exploration_action_count",
            "exploration_noop_action_count",
            "exploration_noop_action_rate",
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
            "relative_score_alpha",
            "relative_delta_lo_cancellations",
            "relative_delta_mode_changes",
            "relative_score",
            "is_better_than_baseline",
        ]
        with validation_metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=validation_fieldnames)
            writer.writeheader()
            writer.writerows(validation_rows)
        # 阶段 0/1 统一 summary：保持与评估脚本一致的核心字段命名，便于直接横向对比。
        validation_unified_summary_rows = _build_validation_unified_summary_rows(validation_rows)
        validation_unified_summary_fields = [
            "episode",
            "validation_seed_count",
            "baseline_mode_changes_mean",
            "baseline_lo_cancellations_mean",
            "dqn_mode_changes_mean",
            "dqn_lo_cancellations_mean",
            "relative_score",
            "relative_score_alpha",
            "relative_delta_mode_changes",
            "relative_delta_lo_cancellations",
            "is_better_than_baseline",
            "mode_changes_delta_vs_baseline",
            "lo_cancellations_delta_vs_baseline",
            "mode_change_ratio",
            "lo_cancellation_ratio",
            "accepted_action_count_mean",
            "rejected_action_count_mean",
            "noop_action_count_mean",
            "noop_action_rate_mean",
            "explicit_noop_action_rate_mean",
            "accepted_action_rate_mean",
            "rejected_action_rate_mean",
            "masked_action_count_mean",
            "valid_action_count_mean",
        ]
        with validation_unified_summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=validation_unified_summary_fields)
            writer.writeheader()
            writer.writerows(validation_unified_summary_rows)

    agent.save(model_path)
    if validation_rows:
        if best_validation_row is None:
            best_validation_row = validation_rows[-1]
        best_relative_score = float(best_validation_row.get("relative_score", 0.0))
        best_delta_mode = float(best_validation_row.get("relative_delta_mode_changes", 0.0))
        best_delta_lo = float(best_validation_row.get("relative_delta_lo_cancellations", 0.0))
        best_model_metadata = {
            "save_best_by": args.save_best_by,
            "selection_metric": "relative_score" if args.save_best_by == "relative_score" else args.save_best_by,
            "relative_score_alpha": args.relative_score_alpha,
            "require_better_than_baseline_for_best": args.require_better_than_baseline_for_best,
            "best_validation_episode": int(best_validation_row["episode"]),
            "best_relative_score": best_relative_score,
            "dqn_lo_cancellations_mean": float(best_validation_row["lo_cancellations_mean"]),
            "baseline_lo_cancellations_mean": float(best_validation_row["baseline_lo_cancellations_mean"]),
            "dqn_mode_changes_mean": float(best_validation_row["mode_changes_mean"]),
            "baseline_mode_changes_mean": float(best_validation_row["baseline_mode_changes_mean"]),
            "delta_lo_cancellations_mean": best_delta_lo,
            "delta_mode_changes_mean": best_delta_mode,
            "best_model_is_better_than_baseline": best_relative_score < 0.0,
            "reward_mode": args.reward_mode,
            "reward_definition": reward_mode_config.describe(),
            "note": (
                "model_best.pt is the best available checkpoint on validation, but it may still be worse than baseline."
            ),
            "selection_reason": "Best checkpoint selected by configured criterion.",
        }
        with best_model_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(best_model_metadata, f, ensure_ascii=False, indent=2)
        if best_relative_score >= 0.0:
            # 文档要求：即使当前 best 仍劣于 baseline，也必须保留 best checkpoint。
            # 因此这里只打印提示，不中断、不回滚、不跳过保存。
            print("WARNING: Best available checkpoint is still worse than baseline on validation.")
            print("Saved anyway for trend analysis: model_best.pt")
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
        "fixed_taskset_seed": args.fixed_taskset_seed,
        "validation_seeds": validation_seeds,
        "validate_every": args.validate_every,
        "validation_end_time": args.validation_end_time,
        "save_best_by": args.save_best_by,
        "relative_score_alpha": args.relative_score_alpha,
        "require_better_than_baseline_for_best": args.require_better_than_baseline_for_best,
        "reward_mode": args.reward_mode,
        "reward_definition": reward_mode_config.describe(),
        "action_space": args.action_space,
        "budget_increase_ratio": args.budget_increase_ratio,
        "budget_decrease_ratio": args.budget_decrease_ratio,
        "budget_floor_ratio": args.budget_floor_ratio,
        "include_explicit_noop": args.include_explicit_noop,
        "forbid_decreasing_hi_budgets": args.forbid_decreasing_hi_budgets,
        "mask_detail_mode": args.mask_detail_mode,
        "log_train_metrics": args.log_train_metrics,
        "trace_every": args.trace_every,
        "seed_metadata": {
            "base_seed": args.seed,
            "network_seed": network_seed,
            "exploration_seed": exploration_seed,
            "replay_seed": replay_seed,
        },
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
