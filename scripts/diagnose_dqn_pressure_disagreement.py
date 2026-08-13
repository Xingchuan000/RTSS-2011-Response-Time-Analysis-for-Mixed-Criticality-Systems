"""Same-state DQN-vs-Pressure disagreement/action-effectiveness diagnostic.

This script runs the DQN trajectory exactly as normal evaluation does.  At each
DQN decision state it *queries* (but never executes) a PressureThresholdValidSelector
on the same observation/mask, records the DQN Q-vector and both decisions, then
executes only the DQN action.  Therefore the diagnostic does not perturb the DQN
trajectory and can be used to explain DQN-vs-Pressure performance gaps.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.dqn import (
    DqnBudgetAgent,
    build_env_from_experiment_config,
    resolve_experiment_bundle,
    validate_observation_binding,
)
from amc_py.evaluation.csem_dynamic_baselines import load_pressure_heuristic_selection
from amc_py.models import Criticality
from amc_py.rl.baseline_policies import PressureThresholdValidSelector
from amc_py.rl.feature_config import FeatureConfig, is_qamc_observation_mode
from amc_py.runtime_models import RuntimeSemantics
from scripts.common_mc_stratified_dynamic_cli import build_mc_stratified_dynamic_config_from_args
from scripts.evaluate_dqn_amc import build_parser as build_eval_parser, _parse_seeds


def _action_desc(action_id, actions):
    if action_id is None:
        return {
            "action_id": None,
            "kind": "hold",
            "increase_task": None,
            "decrease_tasks": [],
            "is_noop": True,
        }
    action = actions[int(action_id)]
    increase_task = getattr(action, "increase_task", None)
    decrease_tasks = list(getattr(action, "decrease_tasks", ()) or ())
    is_noop = bool(getattr(action, "is_noop", False))
    if is_noop:
        kind = "hold"
    elif increase_task is not None and not decrease_tasks:
        kind = "increase"
    elif increase_task is None and len(decrease_tasks) == 1:
        kind = "decrease"
    else:
        kind = "transfer"
    return {
        "action_id": int(action_id),
        "kind": kind,
        "increase_task": increase_task,
        "decrease_tasks": decrease_tasks,
        "is_noop": is_noop,
    }


def _disagreement_category(dqn, pressure):
    if dqn["action_id"] == pressure["action_id"]:
        return "exact_match"
    if dqn["kind"] == "hold" and pressure["kind"] != "hold":
        return "dqn_hold_pressure_action"
    if dqn["kind"] != "hold" and pressure["kind"] == "hold":
        return "dqn_action_pressure_hold"
    if dqn["kind"] == pressure["kind"] and dqn["kind"] in {"increase", "decrease"}:
        return "same_direction_wrong_task"
    if {dqn["kind"], pressure["kind"]} == {"increase", "decrease"}:
        return "opposite_direction"
    return "other_disagreement"


def _pressure_context(observation, ordered_tasks):
    util = sum(float(observation.raw_budgets[t.name]) / float(t.period) for t in ordered_tasks)
    pressures = {}
    for task in ordered_tasks:
        if task.criticality is not Criticality.LO:
            continue
        budget = max(1.0, float(observation.raw_budgets[task.name]))
        recent = float(observation.raw_recent_costs.get(task.name, 0))
        pressures[task.name] = recent / budget
    desc = [name for name, _ in sorted(pressures.items(), key=lambda kv: (-kv[1], kv[0]))]
    asc = [name for name, _ in sorted(pressures.items(), key=lambda kv: (kv[1], kv[0]))]
    return util, pressures, desc, asc


def _target_pressure_rank(action, desc_rank, asc_rank):
    if action["kind"] == "increase" and action["increase_task"] in desc_rank:
        return desc_rank.index(action["increase_task"]) + 1
    if action["kind"] == "decrease" and len(action["decrease_tasks"]) == 1 and action["decrease_tasks"][0] in asc_rank:
        return asc_rank.index(action["decrease_tasks"][0]) + 1
    return None


def _budget_effect(info):
    before = {str(k): float(v) for k, v in dict(info.get("budget_before", {})).items()}
    after = {str(k): float(v) for k, v in dict(info.get("budget_after", {})).items()}
    changed = {}
    for name in sorted(set(before) | set(after)):
        delta = after.get(name, 0.0) - before.get(name, 0.0)
        if abs(delta) > 0.0:
            changed[name] = delta
    return before, after, changed, float(sum(abs(v) for v in changed.values()))


def _mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(vals) if vals else None


def _summarize(rows, *, seed, u_low, u_high):
    cats = Counter(str(r["disagreement_category"]) for r in rows)
    dqn_kinds = Counter(str(r["dqn_action"]["kind"]) for r in rows)
    pressure_kinds = Counter(str(r["pressure_action"]["kind"]) for r in rows)
    nonnoop = [r for r in rows if r["dqn_action"]["kind"] != "hold"]
    effective = [r for r in nonnoop if r["dqn_action_effective"]]
    zero_effect = [r for r in nonnoop if not r["dqn_action_effective"]]
    hi_inc = [
        r for r in nonnoop
        if r["dqn_action"]["kind"] == "increase"
        and str(r["dqn_action"]["increase_task"] or "").startswith("mc_sd_hi_")
    ]
    lo_inc_ranks = Counter(
        int(r["dqn_target_pressure_rank"])
        for r in rows
        if r["dqn_action"]["kind"] == "increase" and r["dqn_target_pressure_rank"] is not None
    )
    agree = [r for r in rows if r["disagreement_category"] == "exact_match"]
    disagree = [r for r in rows if r["disagreement_category"] != "exact_match"]
    return {
        "evaluation_seed": int(seed),
        "pressure_u_low": float(u_low),
        "pressure_u_high": float(u_high),
        "step_count": len(rows),
        "exact_agreement_rate": cats["exact_match"] / len(rows) if rows else None,
        "disagreement_category_counts": dict(cats),
        "dqn_action_kind_counts": dict(dqn_kinds),
        "pressure_action_kind_counts": dict(pressure_kinds),
        "dqn_nonhold_rate": len(nonnoop) / len(rows) if rows else None,
        "pressure_nonhold_rate": sum(k != "hold" and v for k, v in pressure_kinds.items()) / len(rows) if rows else None,
        "dqn_effective_nonhold_rate": len(effective) / len(nonnoop) if nonnoop else None,
        "dqn_zero_effect_nonhold_count": len(zero_effect),
        "dqn_zero_effect_nonhold_rate": len(zero_effect) / len(nonnoop) if nonnoop else None,
        "dqn_hi_increase_count": len(hi_inc),
        "dqn_hi_increase_rate": len(hi_inc) / len(rows) if rows else None,
        "dqn_lo_increase_pressure_rank_counts": {str(k): v for k, v in sorted(lo_inc_ranks.items())},
        "q_margin_mean_agreement": _mean(r["q_margin_second"] for r in agree),
        "q_margin_mean_disagreement": _mean(r["q_margin_second"] for r in disagree),
        "q_margin_normalized_mean_agreement": _mean(r["q_margin_normalized"] for r in agree),
        "q_margin_normalized_mean_disagreement": _mean(r["q_margin_normalized"] for r in disagree),
        "pressure_action_q_rank_mean_on_disagreement": _mean(r["pressure_action_q_rank"] for r in disagree),
        "q_advantage_dqn_over_pressure_mean_on_disagreement": _mean(r["q_advantage_dqn_over_pressure"] for r in disagree),
        "pressure_effective_action_rate": (
            sum(bool(r["pressure_action_effective"]) for r in rows if r["pressure_action"]["kind"] != "hold")
            / max(1, sum(r["pressure_action"]["kind"] != "hold" for r in rows))
        ),
        "dqn_increase_at_upper_bound_count": sum(bool(r["dqn_increase_at_upper_bound_before"]) for r in rows),
        "future_jne_4_mean_agreement": _mean(r["future_equiv_jne_4"] for r in agree),
        "future_jne_4_mean_disagreement": _mean(r["future_equiv_jne_4"] for r in disagree),
        "future_jne_8_mean_agreement": _mean(r["future_equiv_jne_8"] for r in agree),
        "future_jne_8_mean_disagreement": _mean(r["future_equiv_jne_8"] for r in disagree),
    }


def _run_one_seed(args, experiment_config, feature_config, seed, u_low, u_high):
    bundle = resolve_experiment_bundle(experiment_config, seed)
    env = build_env_from_experiment_config(
        experiment_config,
        seed=seed,
        end_time=args.end_time,
        agent_period=args.agent_period,
        semantics=RuntimeSemantics(args.dqn_runtime_semantics),
        reward_mode=args.reward_mode,
        action_space=args.action_space,
        budget_increase_ratio=args.budget_increase_ratio,
        budget_decrease_ratio=args.budget_decrease_ratio,
        include_explicit_noop=args.include_explicit_noop,
        budget_floor_ratio=args.budget_floor_ratio,
        forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
        mask_detail_mode=args.mask_detail_mode,
        enable_deploy_cap_mask=args.enable_deploy_cap_mask,
        deploy_cap_mask_ratio=args.deploy_cap_mask_ratio,
        deploy_cap_mask_criticality=args.deploy_cap_mask_criticality,
        capture_trace=False,
        capture_debug_events=False,
        record_dropped_lo_releases=True,
        c_amc_sem_xf=args.c_amc_sem_xf,
        feature_config=feature_config,
    )
    agent = DqnBudgetAgent.load(args.model, device="cpu")
    agent.double_dqn = bool(args.double_dqn)
    if agent.action_dim != env.action_space_size:
        raise ValueError(f"MODEL_ACTION_DIM_MISMATCH: {agent.action_dim} != {env.action_space_size}")
    obs = env.reset(seed=seed)
    validate_observation_binding(
        agent,
        expected_mode=feature_config.observation_mode,
        expected_feature_names=env.get_observation_feature_names(),
        require_schema_binding=is_qamc_observation_mode(feature_config.observation_mode),
    )
    if getattr(agent, "q_network_type", "mlp") == "action_aware":
        agent.set_action_features(
            env.get_action_feature_matrix(agent.action_feature_mode),
            env.get_action_feature_names(agent.action_feature_mode),
        )
    pressure_selector = PressureThresholdValidSelector(
        ordered_tasks=tuple(bundle.ordered_tasks), u_low=float(u_low), u_high=float(u_high)
    )
    feature_names = list(env.get_observation_feature_names())
    rows = []
    done = False
    step = 0
    while not done:
        mask = tuple(bool(v) for v in env.valid_action_mask())
        if getattr(agent, "q_network_type", "mlp") == "action_aware" and agent.action_feature_mode == "dynamic_v1":
            agent.set_action_features(
                env.get_action_feature_matrix(agent.action_feature_mode),
                env.get_action_feature_names(agent.action_feature_mode),
            )
        qd = agent.compute_q_diagnostics(obs.state_vector, valid_action_mask=mask)
        dqn_id = qd["best_action_id"]
        pressure_id = pressure_selector.select_action_id(
            observation=obs, valid_action_mask=mask, actions=env.actions
        )
        dqn_desc = _action_desc(dqn_id, env.actions)
        pressure_desc = _action_desc(pressure_id, env.actions)
        budget_util, pressures, desc_rank, asc_rank = _pressure_context(obs, tuple(bundle.ordered_tasks))

        valid_ranked = sorted(
            ((int(aid), float(qd["raw_q_values"][int(aid)])) for aid in qd["valid_action_ids"]),
            key=lambda item: (-item[1], item[0]),
        )
        q_rank_by_action = {aid: rank + 1 for rank, (aid, _) in enumerate(valid_ranked)}
        pressure_q_value = (
            float(qd["raw_q_values"][int(pressure_id)]) if pressure_id is not None else None
        )
        q_advantage_over_pressure = (
            float(qd["q_best"]) - pressure_q_value
            if qd["q_best"] is not None and pressure_q_value is not None
            else None
        )
        q_range = (
            float(qd["q_best"]) - float(qd["q_worst"])
            if qd["q_best"] is not None and qd["q_worst"] is not None
            else None
        )
        q_margin_normalized = (
            float(qd["q_margin_second"]) / q_range
            if qd["q_margin_second"] is not None and q_range is not None and q_range > 0.0
            else None
        )

        pressure_candidate_updates = {}
        pressure_candidate_effective = False
        pressure_candidate_abs_delta = 0.0
        if pressure_id is not None:
            peval = env.evaluate_budget_candidate(
                action=env.actions[int(pressure_id)],
                budget_before=dict(obs.raw_budgets),
            )
            pressure_candidate_updates = {str(k): int(v) for k, v in peval.updates.items()}
            pressure_candidate_abs_delta = float(
                sum(abs(float(v) - float(obs.raw_budgets[k])) for k, v in peval.updates.items())
            )
            pressure_candidate_effective = bool(peval.accepted and pressure_candidate_abs_delta > 0.0)

        task_by_name = {t.name: t for t in bundle.ordered_tasks}
        dqn_upper_bound = None
        dqn_at_upper_bound_before = False
        if dqn_desc["kind"] == "increase" and dqn_desc["increase_task"] is not None:
            target_task = task_by_name[dqn_desc["increase_task"]]
            dqn_upper_bound = int(target_task.c_hi if target_task.criticality is Criticality.HI else target_task.deadline)
            dqn_at_upper_bound_before = int(obs.raw_budgets[target_task.name]) >= dqn_upper_bound

        result = env.step(dqn_id)
        before, after, changed, abs_delta = _budget_effect(result.info)
        rows.append({
            "evaluation_seed": int(seed),
            "step": int(step),
            "sim_time_before": int(result.info.get("action_time", result.info.get("time", 0))),
            "pressure_u_low": float(u_low),
            "pressure_u_high": float(u_high),
            "budget_util": float(budget_util),
            "lo_pressure_by_task": pressures,
            "lo_pressure_rank_desc": desc_rank,
            "state_feature_names": feature_names,
            "state_vector": [float(v) for v in obs.state_vector],
            "valid_action_mask": list(mask),
            "valid_action_ids": list(qd["valid_action_ids"]),
            "raw_q_values": list(qd["raw_q_values"]),
            "masked_q_values": [None if not math.isfinite(float(v)) else float(v) for v in qd["masked_q_values"]],
            "q_best": qd["q_best"],
            "q_second_best": qd["q_second_best"],
            "q_worst": qd["q_worst"],
            "q_margin_second": qd["q_margin_second"],
            "q_margin_normalized": q_margin_normalized,
            "pressure_action_q_value": pressure_q_value,
            "pressure_action_q_rank": q_rank_by_action.get(int(pressure_id)) if pressure_id is not None else None,
            "q_advantage_dqn_over_pressure": q_advantage_over_pressure,
            "dqn_action": dqn_desc,
            "pressure_action": pressure_desc,
            "pressure_candidate_updates": pressure_candidate_updates,
            "pressure_candidate_abs_delta": pressure_candidate_abs_delta,
            "pressure_action_effective": pressure_candidate_effective,
            "dqn_increase_upper_bound": dqn_upper_bound,
            "dqn_increase_at_upper_bound_before": dqn_at_upper_bound_before,
            "disagreement_category": _disagreement_category(dqn_desc, pressure_desc),
            "dqn_target_pressure_rank": _target_pressure_rank(dqn_desc, desc_rank, asc_rank),
            "pressure_target_pressure_rank": _target_pressure_rank(pressure_desc, desc_rank, asc_rank),
            "budget_before": before,
            "budget_after": after,
            "budget_changed_tasks": changed,
            "budget_abs_delta": abs_delta,
            "dqn_action_effective": bool(abs_delta > 0.0),
            "accepted": bool(result.info.get("accepted", False)),
            "is_noop": bool(result.info.get("is_noop", False)),
            "delta_lo_equiv_jne": float(result.info.get("delta_lo_equiv_jne", 0.0)),
            "delta_lo_service_quality_sum": float(result.info.get("delta_lo_service_quality_sum", 0.0)),
            "delta_lo_finalized_jobs": float(result.info.get("delta_lo_finalized_jobs", 0.0)),
            "lo_pressure_mean_after": float(result.info.get("lo_pressure_mean", 0.0)),
            "lo_near_cancel_rate_after": float(result.info.get("lo_near_cancel_rate", 0.0)),
            "pingpong_action": float(result.info.get("pingpong_action", 0.0)),
        })
        obs = result.observation
        done = bool(result.done)
        step += 1

    # Attach short-horizon realized outcomes from the unmodified DQN trajectory.
    jne = [float(r["delta_lo_equiv_jne"]) for r in rows]
    for i, row in enumerate(rows):
        row["future_equiv_jne_4"] = float(sum(jne[i:min(len(jne), i + 4)]))
        row["future_equiv_jne_8"] = float(sum(jne[i:min(len(jne), i + 8)]))
    return rows, _summarize(rows, seed=seed, u_low=u_low, u_high=u_high)


def main():
    parser = build_eval_parser()
    parser.description = "Same-state DQN-vs-Pressure diagnostic for mc_stratified_dynamic C-AMC-sem"
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-pressure-u-low", type=float, default=1.0)
    parser.add_argument("--diagnostic-pressure-u-high", type=float, default=1.2)
    parser.add_argument("--diagnostic-pressure-config", type=Path, default=None)
    args = parser.parse_args()

    if args.workload != "mc_stratified_dynamic":
        raise ValueError("This diagnostic currently supports only mc_stratified_dynamic")
    if args.dqn_runtime_semantics != "C_AMC_SEM":
        raise ValueError("This diagnostic currently supports only C_AMC_SEM")
    if args.action_space != "single":
        raise ValueError("This diagnostic currently requires action_space=single")
    if args.evaluation_workers != 1:
        raise ValueError("Use --evaluation-workers 1 for deterministic diagnostic output")

    u_low = float(args.diagnostic_pressure_u_low)
    u_high = float(args.diagnostic_pressure_u_high)
    if args.diagnostic_pressure_config is not None:
        cfg = load_pressure_heuristic_selection(args.diagnostic_pressure_config)
        u_low = float(cfg["u_low"])
        u_high = float(cfg["u_high"])
    if not u_low < u_high:
        raise ValueError("diagnostic pressure thresholds require u_low < u_high")

    feature_config = FeatureConfig(
        observation_mode=args.observation_mode,
        ema_alpha=args.ema_alpha,
        overrun_ema_alpha=args.overrun_ema_alpha,
        history_k=args.history_k,
        event_window=args.event_window,
        max_cost_weight=args.max_cost_weight,
        risk_max_scale=args.risk_max_scale,
        include_safety_margin=args.include_safety_margin,
    )
    experiment_config = build_mc_stratified_dynamic_config_from_args(args)
    args.diagnostic_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    detail_path = args.diagnostic_dir / "dqn_pressure_same_state.jsonl.gz"
    with gzip.open(detail_path, "wt", encoding="utf-8") as out:
        for seed in _parse_seeds(args.seeds):
            rows, summary = _run_one_seed(args, experiment_config, feature_config, seed, u_low, u_high)
            all_summaries.append(summary)
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            (args.diagnostic_dir / f"seed{seed}_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(summary, ensure_ascii=False))

    aggregate = {
        "model": str(args.model),
        "taskset_seed": args.fixed_taskset_seed,
        "seeds": _parse_seeds(args.seeds),
        "end_time": int(args.end_time),
        "pressure_u_low": u_low,
        "pressure_u_high": u_high,
        "per_seed": all_summaries,
    }
    (args.diagnostic_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"detail={detail_path}")


if __name__ == "__main__":
    main()
