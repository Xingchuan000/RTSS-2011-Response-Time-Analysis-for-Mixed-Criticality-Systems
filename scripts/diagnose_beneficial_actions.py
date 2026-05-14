"""诊断 decision point 上相对 noop 的有益动作（不依赖 DQN）。"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

from amc_py.dqn import (
    build_automotive_experiment_config,
    build_env_from_experiment_config,
    build_mc_fairgen_experiment_config,
    build_rtss11_experiment_config,
    build_small_nominal_experiment_config,
    build_small_stress_experiment_config,
)
from amc_py.rl.feature_config import FeatureConfig
from amc_py.rl.reward_config import available_reward_modes
from amc_py.runtime_models import RuntimeSemantics


def _parse_seeds(raw_value: str) -> list[int]:
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["small", "rtss11", "automotive", "mc_fairgen"], default="small")
    parser.add_argument("--scenario", choices=["nominal", "stress"], default="stress")
    parser.add_argument("--eval-seeds", type=str, default="0")
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=1000)

    parser.add_argument("--total-util", type=float, default=0.65)
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--cf", type=float, default=2.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--scenario-seed-offset", type=int, default=100000)
    parser.add_argument("--fixed-taskset-seed", type=int, default=None)
    parser.add_argument("--require-schedulable", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--action-space",
        choices=["single", "constraint_guided_pair", "constraint_guided_transfer"],
        default="single",
    )
    parser.add_argument("--budget-increase-ratio", type=float, default=0.10)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.05)
    parser.add_argument("--include-explicit-noop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--budget-floor-ratio", type=float, default=0.0)
    parser.add_argument("--forbid-decreasing-hi-budgets", action="store_true")
    parser.add_argument("--mask-detail-mode", choices=["minimal", "full"], default="minimal")
    parser.add_argument(
        "--reward-mode",
        choices=list(available_reward_modes()),
        default="mendes",
    )

    parser.add_argument("--constraint-guided-pair-top-k-risk", type=int, default=3)
    parser.add_argument("--constraint-guided-pair-top-k-decrease", type=int, default=5)
    parser.add_argument("--constraint-guided-pair-prefer-lo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--constraint-guided-pair-include-hi-risk-boost", action="store_true")
    parser.add_argument(
        "--constraint-guided-pair-allow-increase-only-when-safe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--automotive-num-runnables", type=int, choices=[150, 250], default=150)
    parser.add_argument(
        "--automotive-mode",
        choices=["fast", "paper_like", "paper_exact", "paper_learnable_headroom"],
        default="paper_like",
    )
    parser.add_argument("--learnable-target-budget-util-min", type=float, default=0.62)
    parser.add_argument("--learnable-target-budget-util-max", type=float, default=0.78)
    parser.add_argument("--learnable-hi-budget-rho-min", type=float, default=0.45)
    parser.add_argument("--learnable-hi-budget-rho-max", type=float, default=0.65)
    parser.add_argument("--learnable-lo-budget-rho-min", type=float, default=0.35)
    parser.add_argument("--learnable-lo-budget-rho-max", type=float, default=0.60)

    parser.add_argument("--mc-fairgen-mode", type=str, default="paper_learnable_headroom")
    parser.add_argument("--mc-fairgen-num-tasks", type=int, default=16)
    parser.add_argument("--mc-fairgen-hi-ratio", type=float, default=0.5)
    parser.add_argument("--mc-fairgen-period-source", type=str, default="automotive")
    parser.add_argument("--mc-fairgen-period-scale", type=int, default=100)
    parser.add_argument("--mc-fairgen-u-hi-lo-min", type=float, default=0.20)
    parser.add_argument("--mc-fairgen-u-hi-lo-max", type=float, default=0.35)
    parser.add_argument("--mc-fairgen-u-hi-hi-min", type=float, default=0.45)
    parser.add_argument("--mc-fairgen-u-hi-hi-max", type=float, default=0.70)
    parser.add_argument("--mc-fairgen-u-lo-lo-min", type=float, default=0.35)
    parser.add_argument("--mc-fairgen-u-lo-lo-max", type=float, default=0.60)
    parser.add_argument("--mc-fairgen-hi-budget-rho-min", type=float, default=0.55)
    parser.add_argument("--mc-fairgen-hi-budget-rho-max", type=float, default=0.75)
    parser.add_argument("--mc-fairgen-lo-budget-rho-min", type=float, default=0.05)
    parser.add_argument("--mc-fairgen-lo-budget-rho-max", type=float, default=0.25)
    parser.add_argument("--mc-fairgen-hi-overrun-prob", type=float, default=0.08)
    parser.add_argument("--mc-fairgen-lo-overrun-prob", type=float, default=0.40)
    parser.add_argument("--mc-fairgen-hi-overrun-factor-min", type=float, default=1.02)
    parser.add_argument("--mc-fairgen-hi-overrun-factor-max", type=float, default=1.25)
    parser.add_argument("--mc-fairgen-lo-overrun-factor-min", type=float, default=1.05)
    parser.add_argument("--mc-fairgen-lo-overrun-factor-max", type=float, default=1.80)

    parser.add_argument("--observation-mode", choices=["v10_basic", "v11_full_10d"], default="v10_basic")
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--overrun-ema-alpha", type=float, default=0.1)
    parser.add_argument("--history-k", type=int, default=8)
    parser.add_argument("--event-window", type=int, default=10)
    parser.add_argument("--max-cost-weight", type=float, default=0.7)
    parser.add_argument("--risk-max-scale", type=float, default=3.0)
    parser.add_argument("--include-safety-margin", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--decision-csv", type=Path, default=Path("outputs/beneficial_actions/decision_level.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/beneficial_actions/summary.csv"))
    return parser


def _build_experiment_config(args: argparse.Namespace):
    if args.workload == "small":
        return build_small_nominal_experiment_config() if args.scenario == "nominal" else build_small_stress_experiment_config()
    if args.workload == "rtss11":
        return build_rtss11_experiment_config(
            total_util=args.total_util,
            num_tasks=args.num_tasks,
            cf=args.cf,
            cp=args.cp,
            require_schedulable=args.require_schedulable,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
        )
    if args.workload == "automotive":
        return build_automotive_experiment_config(
            num_runnables=args.automotive_num_runnables,
            mode=args.automotive_mode,
            require_schedulable=args.require_schedulable,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
            learnable_target_budget_util_min=args.learnable_target_budget_util_min,
            learnable_target_budget_util_max=args.learnable_target_budget_util_max,
            learnable_hi_budget_rho_min=args.learnable_hi_budget_rho_min,
            learnable_hi_budget_rho_max=args.learnable_hi_budget_rho_max,
            learnable_lo_budget_rho_min=args.learnable_lo_budget_rho_min,
            learnable_lo_budget_rho_max=args.learnable_lo_budget_rho_max,
            budget_floor_ratio=args.budget_floor_ratio,
        )
    if args.workload == "mc_fairgen":
        return build_mc_fairgen_experiment_config(
            mode=args.mc_fairgen_mode,
            num_tasks=args.mc_fairgen_num_tasks,
            hi_ratio=args.mc_fairgen_hi_ratio,
            period_source=args.mc_fairgen_period_source,
            period_scale=args.mc_fairgen_period_scale,
            require_schedulable=args.require_schedulable,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
            u_hi_lo_min=args.mc_fairgen_u_hi_lo_min,
            u_hi_lo_max=args.mc_fairgen_u_hi_lo_max,
            u_hi_hi_min=args.mc_fairgen_u_hi_hi_min,
            u_hi_hi_max=args.mc_fairgen_u_hi_hi_max,
            u_lo_lo_min=args.mc_fairgen_u_lo_lo_min,
            u_lo_lo_max=args.mc_fairgen_u_lo_lo_max,
            hi_budget_rho_min=args.mc_fairgen_hi_budget_rho_min,
            hi_budget_rho_max=args.mc_fairgen_hi_budget_rho_max,
            lo_budget_rho_min=args.mc_fairgen_lo_budget_rho_min,
            lo_budget_rho_max=args.mc_fairgen_lo_budget_rho_max,
            hi_overrun_prob=args.mc_fairgen_hi_overrun_prob,
            lo_overrun_prob=args.mc_fairgen_lo_overrun_prob,
            hi_overrun_factor_min=args.mc_fairgen_hi_overrun_factor_min,
            hi_overrun_factor_max=args.mc_fairgen_hi_overrun_factor_max,
            lo_overrun_factor_min=args.mc_fairgen_lo_overrun_factor_min,
            lo_overrun_factor_max=args.mc_fairgen_lo_overrun_factor_max,
        )
    raise ValueError(f"unsupported workload: {args.workload}")


def _build_feature_config(args: argparse.Namespace) -> FeatureConfig:
    return FeatureConfig(
        observation_mode=args.observation_mode,
        ema_alpha=args.ema_alpha,
        overrun_ema_alpha=args.overrun_ema_alpha,
        history_k=args.history_k,
        event_window=args.event_window,
        max_cost_weight=args.max_cost_weight,
        risk_max_scale=args.risk_max_scale,
        include_safety_margin=args.include_safety_margin,
    )



def _is_pareto_beneficial(candidate_info: dict, noop_info: dict) -> bool:
    cand_mc = float(candidate_info.get("delta_mode_changes", 0.0))
    cand_lc = float(candidate_info.get("delta_lo_cancellations", 0.0))
    cand_dm = float(candidate_info.get("delta_deadline_misses", 0.0))

    noop_mc = float(noop_info.get("delta_mode_changes", 0.0))
    noop_lc = float(noop_info.get("delta_lo_cancellations", 0.0))
    noop_dm = float(noop_info.get("delta_deadline_misses", 0.0))

    no_worse = cand_mc <= noop_mc and cand_lc <= noop_lc and cand_dm <= noop_dm
    strictly_better = cand_mc < noop_mc or cand_lc < noop_lc or cand_dm < noop_dm
    return bool(no_worse and strictly_better)


# --- inserted helpers ---

def _find_explicit_noop_action_id(env, valid_mask: list[bool]) -> int | None:
    """Return the explicit noop action id when available.

    In the current AMC action-space implementation, include_explicit_noop appends
    the noop action as the final action.  The diagnostic should compare real
    actions against the same explicit noop action that DQN can select, not
    against env.step(None).
    """
    if not valid_mask:
        return None
    last_idx = len(valid_mask) - 1
    return last_idx if bool(valid_mask[last_idx]) else None


def _best_row_by_reward(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("reward_delta_vs_noop", 0.0)))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    ordered = sorted(float(value) for value in values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _run_seed(args: argparse.Namespace, seed: int) -> list[dict[str, object]]:
    experiment_config = _build_experiment_config(args)
    feature_config = _build_feature_config(args)

    env = build_env_from_experiment_config(
        experiment_config,
        seed=seed,
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
        feature_config=feature_config,
        constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost=args.constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe=args.constraint_guided_pair_allow_increase_only_when_safe,
    )

    obs = env.reset(seed=seed)
    done = False
    decision_index = 0
    rows: list[dict[str, object]] = []

    while not done:
        decision_time = int(obs.time)
        valid_mask = list(env.valid_action_mask())
        noop_action_id = _find_explicit_noop_action_id(env, valid_mask)
        valid_actions = [
            idx
            for idx, is_valid in enumerate(valid_mask)
            if is_valid and (noop_action_id is None or idx != noop_action_id)
        ]

        snapshot = copy.deepcopy(env)

        noop_branch = copy.deepcopy(snapshot)
        noop_step = noop_branch.step(noop_action_id if noop_action_id is not None else None)
        noop_reward = float(noop_step.reward)

        rows.append(
            {
                "seed": seed,
                "decision_index": decision_index,
                "decision_time": decision_time,
                "action_id": noop_action_id if noop_action_id is not None else "noop(None)",
                "is_noop_branch": 1,
                "noop_action_id": noop_action_id if noop_action_id is not None else "None",
                "is_valid_action": 1,
                "reward": noop_reward,
                "delta_mode_changes": float(noop_step.info.get("delta_mode_changes", 0.0)),
                "delta_lo_cancellations": float(noop_step.info.get("delta_lo_cancellations", 0.0)),
                "delta_deadline_misses": float(noop_step.info.get("delta_deadline_misses", 0.0)),
                "reward_delta_vs_noop": 0.0,
                "reward_beneficial": 0,
                "pareto_beneficial": 0,
            }
        )

        for action_id in valid_actions:
            action_branch = copy.deepcopy(snapshot)
            action_step = action_branch.step(int(action_id))
            reward_delta = float(action_step.reward) - noop_reward
            reward_beneficial = int(reward_delta > 0.0)
            pareto_beneficial = int(_is_pareto_beneficial(action_step.info, noop_step.info))
            rows.append(
                {
                    "seed": seed,
                    "decision_index": decision_index,
                    "decision_time": decision_time,
                    "action_id": int(action_id),
                    "is_noop_branch": 0,
                    "noop_action_id": noop_action_id if noop_action_id is not None else "None",
                    "is_valid_action": 1,
                    "reward": float(action_step.reward),
                    "delta_mode_changes": float(action_step.info.get("delta_mode_changes", 0.0)),
                    "delta_lo_cancellations": float(action_step.info.get("delta_lo_cancellations", 0.0)),
                    "delta_deadline_misses": float(action_step.info.get("delta_deadline_misses", 0.0)),
                    "reward_delta_vs_noop": reward_delta,
                    "reward_beneficial": reward_beneficial,
                    "pareto_beneficial": pareto_beneficial,
                }
            )

        base_step = env.step(noop_action_id if noop_action_id is not None else None)
        obs = base_step.observation
        done = bool(base_step.done)
        decision_index += 1

    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _build_parser().parse_args()
    if args.action_space == "constraint_guided_pair":
        args.action_space = "constraint_guided_transfer"
    if args.budget_floor_ratio < 0.0 or args.budget_floor_ratio > 1.0:
        raise ValueError("--budget-floor-ratio must be in [0, 1]")

    seeds = _parse_seeds(args.eval_seeds)

    decision_rows: list[dict[str, object]] = []
    for seed in seeds:
        decision_rows.extend(_run_seed(args, seed))

    decision_fieldnames = [
        "seed",
        "decision_index",
        "decision_time",
        "action_id",
        "is_noop_branch",
        "noop_action_id",
        "is_valid_action",
        "reward",
        "delta_mode_changes",
        "delta_lo_cancellations",
        "delta_deadline_misses",
        "reward_delta_vs_noop",
        "reward_beneficial",
        "pareto_beneficial",
    ]
    _write_csv(args.decision_csv, decision_rows, decision_fieldnames)

    action_rows = [row for row in decision_rows if row["is_noop_branch"] == 0]
    action_rows_by_state: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in action_rows:
        key = (int(row["seed"]), int(row["decision_index"]))
        action_rows_by_state.setdefault(key, []).append(row)
    summary_rows: list[dict[str, object]] = []
    for seed in seeds:
        seed_rows = [row for row in action_rows if int(row["seed"]) == seed]
        decision_count = len({int(row["decision_index"]) for row in seed_rows})
        action_count = len(seed_rows)
        reward_beneficial_count = sum(int(row["reward_beneficial"]) for row in seed_rows)
        pareto_beneficial_count = sum(int(row["pareto_beneficial"]) for row in seed_rows)
        seed_state_rows = {
            key: rows_for_state
            for key, rows_for_state in action_rows_by_state.items()
            if key[0] == seed
        }
        reward_beneficial_state_count = sum(
            1 for rows_for_state in seed_state_rows.values() if any(int(row["reward_beneficial"]) for row in rows_for_state)
        )
        pareto_beneficial_state_count = sum(
            1 for rows_for_state in seed_state_rows.values() if any(int(row["pareto_beneficial"]) for row in rows_for_state)
        )
        best_rows = [
            best_row
            for rows_for_state in seed_state_rows.values()
            if (best_row := _best_row_by_reward(rows_for_state)) is not None
        ]
        best_reward_deltas = [float(row["reward_delta_vs_noop"]) for row in best_rows]
        best_mode_deltas = [float(row["delta_mode_changes"]) for row in best_rows]
        best_lo_deltas = [float(row["delta_lo_cancellations"]) for row in best_rows]
        best_deadline_deltas = [float(row["delta_deadline_misses"]) for row in best_rows]
        summary_rows.append(
            {
                "scope": "seed",
                "seed": seed,
                "decision_count": decision_count,
                "action_count": action_count,
                "reward_beneficial_count": reward_beneficial_count,
                "pareto_beneficial_count": pareto_beneficial_count,
                "reward_beneficial_rate": (reward_beneficial_count / action_count) if action_count > 0 else 0.0,
                "pareto_beneficial_rate": (pareto_beneficial_count / action_count) if action_count > 0 else 0.0,
                "reward_beneficial_state_count": reward_beneficial_state_count,
                "pareto_beneficial_state_count": pareto_beneficial_state_count,
                "reward_beneficial_state_ratio": (reward_beneficial_state_count / decision_count) if decision_count > 0 else 0.0,
                "pareto_beneficial_state_ratio": (pareto_beneficial_state_count / decision_count) if decision_count > 0 else 0.0,
                "best_reward_delta_vs_noop_mean": (sum(best_reward_deltas) / len(best_reward_deltas)) if best_reward_deltas else 0.0,
                "best_reward_delta_vs_noop_p50": _percentile(best_reward_deltas, 0.50),
                "best_reward_delta_vs_noop_p90": _percentile(best_reward_deltas, 0.90),
                "best_reward_delta_vs_noop_max": max(best_reward_deltas) if best_reward_deltas else 0.0,
                "best_mode_delta_for_best_reward_mean": (sum(best_mode_deltas) / len(best_mode_deltas)) if best_mode_deltas else 0.0,
                "best_lo_delta_for_best_reward_mean": (sum(best_lo_deltas) / len(best_lo_deltas)) if best_lo_deltas else 0.0,
                "best_deadline_delta_for_best_reward_sum": sum(best_deadline_deltas) if best_deadline_deltas else 0.0,
            }
        )

    total_decisions = len({(int(row["seed"]), int(row["decision_index"])) for row in action_rows})
    total_actions = len(action_rows)
    total_reward_beneficial = sum(int(row["reward_beneficial"]) for row in action_rows)
    total_pareto_beneficial = sum(int(row["pareto_beneficial"]) for row in action_rows)
    total_reward_beneficial_states = sum(
        1 for rows_for_state in action_rows_by_state.values() if any(int(row["reward_beneficial"]) for row in rows_for_state)
    )
    total_pareto_beneficial_states = sum(
        1 for rows_for_state in action_rows_by_state.values() if any(int(row["pareto_beneficial"]) for row in rows_for_state)
    )
    global_best_rows = [
        best_row
        for rows_for_state in action_rows_by_state.values()
        if (best_row := _best_row_by_reward(rows_for_state)) is not None
    ]
    global_best_reward_deltas = [float(row["reward_delta_vs_noop"]) for row in global_best_rows]
    global_best_mode_deltas = [float(row["delta_mode_changes"]) for row in global_best_rows]
    global_best_lo_deltas = [float(row["delta_lo_cancellations"]) for row in global_best_rows]
    global_best_deadline_deltas = [float(row["delta_deadline_misses"]) for row in global_best_rows]
    summary_rows.append(
        {
            "scope": "global",
            "seed": "all",
            "decision_count": total_decisions,
            "action_count": total_actions,
            "reward_beneficial_count": total_reward_beneficial,
            "pareto_beneficial_count": total_pareto_beneficial,
            "reward_beneficial_rate": (total_reward_beneficial / total_actions) if total_actions > 0 else 0.0,
            "pareto_beneficial_rate": (total_pareto_beneficial / total_actions) if total_actions > 0 else 0.0,
            "reward_beneficial_state_count": total_reward_beneficial_states,
            "pareto_beneficial_state_count": total_pareto_beneficial_states,
            "reward_beneficial_state_ratio": (total_reward_beneficial_states / total_decisions) if total_decisions > 0 else 0.0,
            "pareto_beneficial_state_ratio": (total_pareto_beneficial_states / total_decisions) if total_decisions > 0 else 0.0,
            "best_reward_delta_vs_noop_mean": (sum(global_best_reward_deltas) / len(global_best_reward_deltas)) if global_best_reward_deltas else 0.0,
            "best_reward_delta_vs_noop_p50": _percentile(global_best_reward_deltas, 0.50),
            "best_reward_delta_vs_noop_p90": _percentile(global_best_reward_deltas, 0.90),
            "best_reward_delta_vs_noop_max": max(global_best_reward_deltas) if global_best_reward_deltas else 0.0,
            "best_mode_delta_for_best_reward_mean": (sum(global_best_mode_deltas) / len(global_best_mode_deltas)) if global_best_mode_deltas else 0.0,
            "best_lo_delta_for_best_reward_mean": (sum(global_best_lo_deltas) / len(global_best_lo_deltas)) if global_best_lo_deltas else 0.0,
            "best_deadline_delta_for_best_reward_sum": sum(global_best_deadline_deltas) if global_best_deadline_deltas else 0.0,
        }
    )

    summary_fieldnames = [
        "scope",
        "seed",
        "decision_count",
        "action_count",
        "reward_beneficial_count",
        "pareto_beneficial_count",
        "reward_beneficial_rate",
        "pareto_beneficial_rate",
        "reward_beneficial_state_count",
        "pareto_beneficial_state_count",
        "reward_beneficial_state_ratio",
        "pareto_beneficial_state_ratio",
        "best_reward_delta_vs_noop_mean",
        "best_reward_delta_vs_noop_p50",
        "best_reward_delta_vs_noop_p90",
        "best_reward_delta_vs_noop_max",
        "best_mode_delta_for_best_reward_mean",
        "best_lo_delta_for_best_reward_mean",
        "best_deadline_delta_for_best_reward_sum",
    ]
    _write_csv(args.summary_csv, summary_rows, summary_fieldnames)


if __name__ == "__main__":
    main()
