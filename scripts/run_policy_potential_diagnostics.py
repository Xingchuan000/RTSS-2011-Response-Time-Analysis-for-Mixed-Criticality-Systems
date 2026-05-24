#!/usr/bin/env python3
"""Policy potential diagnostics for AMC DQN tasksets.

This script answers three questions for selected taskset seeds:
1) How much can an immediate greedy oracle improve over noop/baseline?
2) If the DQN top-1 action is banned, is there still improvement potential?
3) Is the useful opportunity concentrated in top-1, or distributed across other actions?

It intentionally reuses the project's existing environment, mask, reward and safety logic.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

from amc_py.dqn import build_env_from_experiment_config, build_mc_fairgen_experiment_config, resolve_experiment_bundle
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import compute_service_quality_metrics
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics


def parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for part in (x.strip() for x in raw.split(",")):
        if not part:
            continue
        if ":" in part:
            a, b = [int(x.strip()) for x in part.split(":", 1)]
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def load_top1_actions(path: Path | None) -> dict[int, int]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        result: dict[int, int] = {}
        for row in reader:
            if not row.get("candidate_seed") or not row.get("top1_action_id"):
                continue
            result[int(float(row["candidate_seed"]))] = int(float(row["top1_action_id"]))
        return result


def build_experiment_config(args: argparse.Namespace, taskset_seed: int):
    return build_mc_fairgen_experiment_config(
        mode=args.mc_fairgen_mode,
        num_tasks=args.mc_fairgen_num_tasks,
        hi_ratio=args.mc_fairgen_hi_ratio,
        period_source=args.mc_fairgen_period_source,
        period_scale=args.mc_fairgen_period_scale,
        require_schedulable=args.require_schedulable,
        scenario_seed_offset=args.scenario_seed_offset,
        fixed_taskset_seed=taskset_seed,
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


def build_feature_config(args: argparse.Namespace) -> FeatureConfig:
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


def build_env(args: argparse.Namespace, experiment_config: Any, eval_seed: int):
    return build_env_from_experiment_config(
        experiment_config,
        seed=eval_seed,
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
        feature_config=build_feature_config(args),
        constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost=args.constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe=args.constraint_guided_pair_allow_increase_only_when_safe,
    )


def action_name(env: Any, action_id: int | None) -> str:
    if action_id is None:
        return "noop(None)"
    action = env._actions[action_id]
    if getattr(action, "is_noop", False):
        return "explicit_noop"
    inc = getattr(action, "increase_task", None)
    dec = "+".join(getattr(action, "decrease_tasks", ()) or ())
    if inc and dec:
        return f"inc:{inc}|dec:{dec}"
    if inc:
        return f"inc:{inc}"
    if dec:
        return f"dec:{dec}"
    return f"action_{action_id}"


def explicit_noop_id(env: Any, valid_mask: list[bool]) -> int | None:
    for idx, action in enumerate(env._actions):
        if getattr(action, "is_noop", False) and idx < len(valid_mask) and valid_mask[idx]:
            return idx
    return None


def choose_greedy_action(
    env: Any,
    *,
    exclude: set[int],
    only: set[int] | None,
) -> tuple[int | None, dict[str, Any]]:
    valid_mask = list(env.valid_action_mask())
    noop_id = explicit_noop_id(env, valid_mask)
    noop_branch = copy.deepcopy(env)
    noop_step = noop_branch.step(noop_id if noop_id is not None else None)
    noop_reward = float(noop_step.reward)

    best_action: int | None = None
    best_step = noop_step
    best_reward = noop_reward
    best_delta = 0.0
    valid_nonnoop = 0
    candidate_count = 0
    positive_count = 0

    for action_id, is_valid in enumerate(valid_mask):
        if not is_valid:
            continue
        if noop_id is not None and action_id == noop_id:
            continue
        if action_id in exclude:
            continue
        if only is not None and action_id not in only:
            continue
        valid_nonnoop += 1
        branch = copy.deepcopy(env)
        step = branch.step(action_id)
        delta = float(step.reward) - noop_reward
        candidate_count += 1
        if delta > 0:
            positive_count += 1
        if delta > best_delta:
            best_action = action_id
            best_step = step
            best_reward = float(step.reward)
            best_delta = delta

    return best_action, {
        "noop_action_id": noop_id if noop_id is not None else "None",
        "valid_nonnoop_count": valid_nonnoop,
        "candidate_count": candidate_count,
        "positive_action_count": positive_count,
        "chosen_reward": best_reward,
        "chosen_reward_delta_vs_noop": best_delta,
        "chosen_delta_mode_changes": float(best_step.info.get("delta_mode_changes", 0.0)),
        "chosen_delta_lo_cancellations": float(best_step.info.get("delta_lo_cancellations", 0.0)),
        "chosen_delta_deadline_misses": float(best_step.info.get("delta_deadline_misses", 0.0)),
    }


def run_policy(args: argparse.Namespace, taskset_seed: int, eval_seed: int, policy_name: str, top1_action_id: int | None) -> dict[str, Any]:
    experiment_config = build_experiment_config(args, taskset_seed)
    env = build_env(args, experiment_config, eval_seed)
    obs = env.reset(seed=eval_seed)
    done = False
    decisions = 0
    selected_nonnoop = 0
    selected_top1 = 0
    positive_decisions = 0
    valid_nonnoop_counts: list[float] = []
    positive_action_counts: list[float] = []
    reward_delta_sum = 0.0
    action_hist: dict[int, int] = {}

    if policy_name == "greedy_all":
        exclude: set[int] = set()
        only: set[int] | None = None
    elif policy_name == "greedy_ban_top1":
        exclude = {top1_action_id} if top1_action_id is not None else set()
        only = None
    elif policy_name == "greedy_only_top1":
        exclude = set()
        only = {top1_action_id} if top1_action_id is not None else set()
    else:
        raise ValueError(f"unknown policy_name={policy_name}")

    while not done:
        chosen, info = choose_greedy_action(env, exclude=exclude, only=only)
        decisions += 1
        valid_nonnoop_counts.append(float(info["valid_nonnoop_count"]))
        positive_action_counts.append(float(info["positive_action_count"]))
        reward_delta_sum += float(info["chosen_reward_delta_vs_noop"])
        if float(info["chosen_reward_delta_vs_noop"]) > 0:
            positive_decisions += 1
        if chosen is not None:
            selected_nonnoop += 1
            action_hist[chosen] = action_hist.get(chosen, 0) + 1
            if top1_action_id is not None and chosen == top1_action_id:
                selected_top1 += 1
        step = env.step(chosen)
        obs = step.observation
        done = bool(step.done)

    runtime_result = env._engine.finish()
    service = compute_service_quality_metrics(runtime_result)
    top_action_id = max(action_hist, key=action_hist.get) if action_hist else None
    top_action_share = (action_hist[top_action_id] / selected_nonnoop) if top_action_id is not None and selected_nonnoop else 0.0
    return {
        "candidate_seed": taskset_seed,
        "eval_seed": eval_seed,
        "policy": policy_name,
        "top1_action_id_from_dqn": top1_action_id if top1_action_id is not None else "",
        "decisions": decisions,
        "selected_nonnoop": selected_nonnoop,
        "selected_nonnoop_rate": selected_nonnoop / decisions if decisions else 0.0,
        "selected_top1": selected_top1,
        "selected_top1_rate": selected_top1 / decisions if decisions else 0.0,
        "positive_decisions": positive_decisions,
        "positive_decision_rate": positive_decisions / decisions if decisions else 0.0,
        "valid_nonnoop_count_mean": mean(valid_nonnoop_counts) if valid_nonnoop_counts else 0.0,
        "positive_action_count_mean": mean(positive_action_counts) if positive_action_counts else 0.0,
        "reward_delta_vs_noop_sum": reward_delta_sum,
        "mode_changes": runtime_result.mode_change_count(),
        "lo_cancellations": runtime_result.lo_job_cancellation_count(),
        "deadline_misses": len(runtime_result.deadline_misses),
        "released_lo_jobs": service.released_lo_jobs,
        "cancelled_lo_jobs": service.cancelled_lo_jobs,
        "lc_service_loss": service.lc_service_loss,
        "lc_qos": service.lc_qos,
        "top_selected_action_id": top_action_id if top_action_id is not None else "",
        "top_selected_action_name": action_name(env, top_action_id) if top_action_id is not None else "",
        "top_selected_action_share": top_action_share,
    }


def run_baseline(args: argparse.Namespace, taskset_seed: int, eval_seed: int) -> dict[str, Any]:
    experiment_config = build_experiment_config(args, taskset_seed)
    bundle = resolve_experiment_bundle(experiment_config, eval_seed)
    result = simulate_ordered_taskset_event_driven(
        ordered_tasks=bundle.ordered_tasks,
        scenario=bundle.scenario,
        config=RuntimeConfig(end_time=args.end_time, semantics=RuntimeSemantics.AMC_PLUS),
    )
    service = compute_service_quality_metrics(result)
    return {
        "candidate_seed": taskset_seed,
        "eval_seed": eval_seed,
        "policy": "amc_plus_baseline",
        "top1_action_id_from_dqn": "",
        "decisions": "",
        "selected_nonnoop": "",
        "selected_nonnoop_rate": "",
        "selected_top1": "",
        "selected_top1_rate": "",
        "positive_decisions": "",
        "positive_decision_rate": "",
        "valid_nonnoop_count_mean": "",
        "positive_action_count_mean": "",
        "reward_delta_vs_noop_sum": "",
        "mode_changes": result.mode_change_count(),
        "lo_cancellations": result.lo_job_cancellation_count(),
        "deadline_misses": len(result.deadline_misses),
        "released_lo_jobs": service.released_lo_jobs,
        "cancelled_lo_jobs": service.cancelled_lo_jobs,
        "lc_service_loss": service.lc_service_loss,
        "lc_qos": service.lc_qos,
        "top_selected_action_id": "",
        "top_selected_action_name": "",
        "top_selected_action_share": "",
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["candidate_seed"]), str(row["policy"])), []).append(row)
    out: list[dict[str, Any]] = []
    baseline_by_seed: dict[int, float] = {}
    for (seed, policy), rs in groups.items():
        lc_values = [float(r["lo_cancellations"]) for r in rs]
        if policy == "amc_plus_baseline":
            baseline_by_seed[seed] = mean(lc_values)
    for (seed, policy), rs in sorted(groups.items()):
        lo_mean = mean(float(r["lo_cancellations"]) for r in rs)
        base = baseline_by_seed.get(seed, math.nan)
        rel_red = ((base - lo_mean) / base) if base and not math.isnan(base) else math.nan
        out.append({
            "candidate_seed": seed,
            "policy": policy,
            "eval_seed_count": len(rs),
            "lo_cancellations_mean": lo_mean,
            "baseline_lo_cancellations_mean": base,
            "relative_lc_reduction": rel_red,
            "relative_lc_reduction_pct": rel_red * 100 if not math.isnan(rel_red) else math.nan,
            "mode_changes_mean": mean(float(r["mode_changes"]) for r in rs),
            "deadline_misses_sum": sum(int(float(r["deadline_misses"])) for r in rs),
            "lc_qos_mean": mean(float(r["lc_qos"]) for r in rs),
            "positive_decision_rate_mean": mean(float(r["positive_decision_rate"] or 0.0) for r in rs),
            "selected_nonnoop_rate_mean": mean(float(r["selected_nonnoop_rate"] or 0.0) for r in rs),
            "top_selected_action_share_mean": mean(float(r["top_selected_action_share"] or 0.0) for r in rs),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--taskset-seeds", required=True, help="Comma list/ranges of fixed taskset seeds, e.g. 1896,962,185")
    p.add_argument("--eval-seeds", default="200:229")
    p.add_argument("--top1-action-csv", type=Path, default=None, help="CSV with candidate_seed and top1_action_id columns")
    p.add_argument("--policies", default="greedy_all,greedy_ban_top1,greedy_only_top1")
    p.add_argument("--output-detail", type=Path, required=True)
    p.add_argument("--output-summary", type=Path, required=True)

    p.add_argument("--end-time", type=int, default=10_000_000)
    p.add_argument("--agent-period", type=int, default=25_000)
    p.add_argument("--reward-mode", default="interval_qos_v2")
    p.add_argument("--action-space", default="single", choices=["single", "pair", "triple", "constraint_guided_pair", "constraint_guided_transfer"])
    p.add_argument("--budget-increase-ratio", type=float, default=0.025)
    p.add_argument("--budget-decrease-ratio", type=float, default=0.015)
    p.add_argument("--include-explicit-noop", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--budget-floor-ratio", type=float, default=0.9)
    p.add_argument("--forbid-decreasing-hi-budgets", action="store_true")
    p.add_argument("--mask-detail-mode", choices=["minimal", "full"], default="minimal")

    p.add_argument("--mc-fairgen-mode", default="paper_learnable_headroom")
    p.add_argument("--mc-fairgen-num-tasks", type=int, default=12)
    p.add_argument("--mc-fairgen-hi-ratio", type=float, default=0.5)
    p.add_argument("--mc-fairgen-period-source", default="controlled_medium")
    p.add_argument("--mc-fairgen-period-scale", type=int, default=500)
    p.add_argument("--mc-fairgen-u-hi-lo-min", type=float, default=0.20)
    p.add_argument("--mc-fairgen-u-hi-lo-max", type=float, default=0.35)
    p.add_argument("--mc-fairgen-u-hi-hi-min", type=float, default=0.45)
    p.add_argument("--mc-fairgen-u-hi-hi-max", type=float, default=0.70)
    p.add_argument("--mc-fairgen-u-lo-lo-min", type=float, default=0.25)
    p.add_argument("--mc-fairgen-u-lo-lo-max", type=float, default=0.45)
    p.add_argument("--mc-fairgen-hi-budget-rho-min", type=float, default=0.55)
    p.add_argument("--mc-fairgen-hi-budget-rho-max", type=float, default=0.75)
    p.add_argument("--mc-fairgen-lo-budget-rho-min", type=float, default=0.20)
    p.add_argument("--mc-fairgen-lo-budget-rho-max", type=float, default=0.40)
    p.add_argument("--mc-fairgen-hi-overrun-prob", type=float, default=0.08)
    p.add_argument("--mc-fairgen-lo-overrun-prob", type=float, default=0.12)
    p.add_argument("--mc-fairgen-hi-overrun-factor-min", type=float, default=1.02)
    p.add_argument("--mc-fairgen-hi-overrun-factor-max", type=float, default=1.25)
    p.add_argument("--mc-fairgen-lo-overrun-factor-min", type=float, default=1.02)
    p.add_argument("--mc-fairgen-lo-overrun-factor-max", type=float, default=1.25)
    p.add_argument("--require-schedulable", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scenario-seed-offset", type=int, default=100000)

    p.add_argument("--observation-mode", choices=["v10_basic", "v11_full_10d", "v12_full_14d"], default="v11_full_10d")
    p.add_argument("--ema-alpha", type=float, default=0.2)
    p.add_argument("--overrun-ema-alpha", type=float, default=0.1)
    p.add_argument("--history-k", type=int, default=8)
    p.add_argument("--event-window", type=int, default=10)
    p.add_argument("--max-cost-weight", type=float, default=0.7)
    p.add_argument("--risk-max-scale", type=float, default=3.0)
    p.add_argument("--include-safety-margin", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--constraint-guided-pair-top-k-risk", type=int, default=3)
    p.add_argument("--constraint-guided-pair-top-k-decrease", type=int, default=5)
    p.add_argument("--constraint-guided-pair-prefer-lo", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--constraint-guided-pair-include-hi-risk-boost", action="store_true")
    p.add_argument("--constraint-guided-pair-allow-increase-only-when-safe", action=argparse.BooleanOptionalAction, default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.action_space == "constraint_guided_pair":
        args.action_space = "constraint_guided_transfer"
    taskset_seeds = parse_seeds(args.taskset_seeds)
    eval_seeds = parse_seeds(args.eval_seeds)
    top1_map = load_top1_actions(args.top1_action_csv)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    for seed in taskset_seeds:
        top1 = top1_map.get(seed)
        for eval_seed in eval_seeds:
            rows.append(run_baseline(args, seed, eval_seed))
            for policy in policies:
                rows.append(run_policy(args, seed, eval_seed, policy, top1))
    write_csv(args.output_detail, rows)
    write_csv(args.output_summary, aggregate(rows))


if __name__ == "__main__":
    main()
