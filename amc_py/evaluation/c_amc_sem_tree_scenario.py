"""Ordinary C-AMC-sem tree HOUT scenario evaluator.

This is intentionally a thin research adapter around the existing VIPER
``evaluate_tree_policy_once`` routine.  It accepts only ordinary runtime inputs;
mutation selection remains outside ``amc_py`` in ``nonvacuity_lab``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from amc_py.dqn.experiment import resolve_experiment_bundle
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics
from amc_py.viper.artifacts import load_tree_policy_artifact
from amc_py.viper.metrics import evaluate_tree_policy_once


def _load_callable(spec: str) -> Callable[..., Any]:
    module, sep, name = str(spec).partition(":")
    if not sep:
        raise ValueError("experiment_factory must be module:function")
    value = getattr(importlib.import_module(module), name, None)
    if not callable(value):
        raise TypeError(f"experiment_factory is not callable: {spec}")
    return value


def _factory_from_config(config: dict[str, Any]):
    spec = config.get("experiment_factory")
    if isinstance(spec, str):
        return _load_callable(spec)
    name = str(config.get("experiment_name", "")).lower()
    aliases = {
        "small_stress": "amc_py.dqn.experiment:build_small_stress_experiment_config",
        "small_nominal": "amc_py.dqn.experiment:build_small_nominal_experiment_config",
        "rtss11": "amc_py.dqn.experiment:build_rtss11_experiment_config",
        "automotive": "amc_py.dqn.experiment:build_automotive_experiment_config",
        "mc_fairgen": "amc_py.dqn.experiment:build_mc_fairgen_experiment_config",
    }
    if name not in aliases:
        raise ValueError("runtime config requires experiment_factory or known experiment_name")
    return _load_callable(aliases[name])


def _read_metadata(tree_path: Path) -> dict[str, Any]:
    path = tree_path.parent / "metadata.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()




def _criticality_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).upper()


def _canonical_task_signature(taskset_path: Path) -> tuple[tuple[Any, ...], ...]:
    raw = json.loads(taskset_path.read_text(encoding="utf-8"))
    rows = raw.get("ordered_tasks", raw.get("tasks", [])) if isinstance(raw, dict) else []
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"taskset artifact has no task rows: {taskset_path}")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("taskset row must be an object")
        result.append((
            str(row.get("name", row.get("task_id"))),
            int(row["period"]),
            int(row.get("deadline", row["period"])),
            int(row.get("code_c_lo", row.get("c_lo"))),
            int(row.get("code_c_hi", row.get("c_hi", row.get("code_c_lo", row.get("c_lo"))))),
            _criticality_text(row["criticality"]),
        ))
    return tuple(result)


def _runtime_task_signature(tasks) -> tuple[tuple[Any, ...], ...]:
    return tuple((
        str(task.name), int(task.period), int(task.deadline),
        int(task.c_lo), int(task.c_hi), _criticality_text(task.criticality),
    ) for task in tasks)


def _verify_runtime_taskset(experiment_config, *, seed: int, taskset_path: Path | None) -> str | None:
    if taskset_path is None:
        return None
    expected = _canonical_task_signature(taskset_path)
    actual = _runtime_task_signature(resolve_experiment_bundle(experiment_config, seed).ordered_tasks)
    if actual != expected:
        raise ValueError(
            "HOUT_TASKSET_MISMATCH: generated runtime taskset does not match "
            f"the resolved proof taskset: expected={expected!r}:actual={actual!r}"
        )
    payload = json.dumps(actual, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _infer_taskset_seed(seed_dir: Path, config: dict[str, Any]) -> int:
    if config.get("taskset_seed") is not None:
        return int(config["taskset_seed"])
    for part in reversed(seed_dir.parts):
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:])
    return int(config.get("default_taskset_seed", 0))


def run_c_amc_sem_tree_scenario(
    *, seed_dir: Path, tree_path: Path, scenario_seed: int,
    runtime_config: dict[str, Any],
):
    seed_dir = Path(seed_dir).resolve()
    tree_path = Path(tree_path).resolve()
    metadata = _read_metadata(tree_path)
    factory = _factory_from_config(runtime_config)
    kwargs = dict(runtime_config.get("experiment_factory_kwargs", {}))
    experiment_config = factory(**kwargs)
    tree_policy = load_tree_policy_artifact(tree_path.parent, require_integer_tree=True)

    observation_mode = str(
        runtime_config.get(
            "observation_mode",
            metadata.get("observation_mode", metadata.get("student_observation_mode", "v11_full_10d")),
        )
    )
    feature_config = FeatureConfig(observation_mode=observation_mode)
    semantics = RuntimeSemantics(str(runtime_config.get("runtime_semantics", "C_AMC_SEM")))
    taskset_seed = _infer_taskset_seed(seed_dir, runtime_config)
    horizon = int(runtime_config.get("horizon", runtime_config.get("end_time", 0)))
    if horizon <= 0:
        raise ValueError("HOUT runtime config requires positive horizon/end_time")
    evaluation_seed = int(
        scenario_seed if runtime_config.get("scenario_seed_drives_bundle", True) else taskset_seed
    )
    taskset_path_raw = runtime_config.get("taskset_path")
    taskset_path = Path(str(taskset_path_raw)).resolve() if taskset_path_raw else None
    runtime_taskset_fingerprint = _verify_runtime_taskset(
        experiment_config, seed=evaluation_seed, taskset_path=taskset_path
    )
    row, runtime_result, action_log = evaluate_tree_policy_once(
        tree_policy=tree_policy,
        experiment_config=experiment_config,
        seed=evaluation_seed,
        end_time=horizon,
        agent_period=int(runtime_config.get("agent_period", 100)),
        runtime_semantics=semantics,
        reward_mode=str(runtime_config.get("reward_mode", "mendes")),
        action_space=str(runtime_config.get("action_space", "single")),
        budget_increase_ratio=float(runtime_config.get("budget_increase_ratio", 0.02)),
        budget_decrease_ratio=float(runtime_config.get("budget_decrease_ratio", 0.02)),
        include_explicit_noop=bool(runtime_config.get("include_explicit_noop", False)),
        budget_floor_ratio=float(runtime_config.get("budget_floor_ratio", 0.0)),
        forbid_decreasing_hi_budgets=bool(runtime_config.get("forbid_decreasing_hi_budgets", True)),
        mask_detail_mode=str(runtime_config.get("mask_detail_mode", "full")),
        enable_deploy_cap_mask=bool(runtime_config.get("enable_deploy_cap_mask", True)),
        deploy_cap_mask_ratio=float(runtime_config.get("deploy_cap_mask_ratio", 4.0)),
        deploy_cap_mask_criticality=str(runtime_config.get("deploy_cap_mask_criticality", "lo")).lower(),
        feature_config=feature_config,
        c_amc_sem_xf=float(runtime_config.get("c_amc_sem_xf", 0.5)),
        teacher=None,
        leaf_audit_enabled=True,
        leaf_audit_state_mode=str(runtime_config.get("leaf_audit_state_mode", "split")),
        leaf_audit_top_k_actions=int(runtime_config.get("leaf_audit_top_k_actions", 5)),
    )
    demand_path_raw = runtime_config.get("demand_trace_path")
    demand_path = Path(str(demand_path_raw)).resolve() if demand_path_raw else None
    variant = tree_path.parent.name
    events = []
    lo_qos = row.get("lo_quality_qos", row.get("lc_qos"))
    zero_service = row.get("lo_zero_service_ratio")
    cancellation_count = row.get("lo_cancellations", row.get("cancelled_lo_jobs"))
    hi_miss_count = int(row.get("hi_deadline_misses", 0))
    lo_miss_count = int(row.get("lo_deadline_misses", 0))
    for decision_index, item in enumerate(action_log):
        event = dict(item)
        raw_invalid = bool(event.get("tree_raw_top1_invalid", False))
        raw_action = event.get("tree_raw_top1_action_id")
        selected_action = event.get("tree_selected_action_id", event.get("action_id"))
        candidate = event.get("candidate_budgets", event.get("candidate_budget", {}))
        raw_reject_reason = (
            event.get("tree_raw_top1_reject_reason")
            if raw_invalid
            else event.get("reject_reason")
        )
        event.update({
            "seed": taskset_seed,
            "tree_variant": variant,
            "scenario_id": int(scenario_seed),
            "scenario_seed": int(scenario_seed),
            "controller_decision_index": int(event.get("tree_audit_step_index", decision_index)),
            "leaf_id": event.get("tree_leaf_id", event.get("leaf_id")),
            "selected_rank": event.get("tree_selected_rank", event.get("selected_rank")),
            "raw_top1_invalid": raw_invalid,
            "raw_top1_valid": not raw_invalid,
            "implicit_noop": bool(event.get("tree_no_valid_action", event.get("noop", False))),
            "all_invalid": bool(event.get("tree_no_valid_action", False)),
            "selected_action_id": selected_action,
            "raw_top1_action_id": raw_action,
            "rejected_action_id": event.get("rejected_action_id", raw_action if raw_invalid else None),
            "reject_reason": raw_reject_reason,
            "candidate_budget": candidate,
            "budget_before": event.get("budget_before", {}),
            "budget_after": event.get("budget_after", {}),
            "hi_miss": False,
            "lo_miss": False,
            "lo_qos": lo_qos,
            "zero_service": (None if zero_service is None else float(zero_service) > 0.0),
            "cancellation_count": cancellation_count,
        })
        events.append(event)
    if events:
        events[-1]["hi_miss"] = hi_miss_count > 0
        events[-1]["lo_miss"] = lo_miss_count > 0
    summary = {
        **row,
        "scenario_seed": int(scenario_seed),
        "taskset_seed": taskset_seed,
        "scenario_list": [int(scenario_seed)],
        "horizon": horizon,
        "taskset_hash": _sha256(taskset_path),
        "runtime_taskset_fingerprint": runtime_taskset_fingerprint,
        "demand_trace_fingerprint": _sha256(demand_path) or f"scenario_seed:{scenario_seed}",
        "tree_variant": variant,
        "hi_deadline_miss_count": int(row.get("hi_deadline_misses", 0)),
    }
    return summary, events
