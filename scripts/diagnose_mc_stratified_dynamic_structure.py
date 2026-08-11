#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent structure diagnostics for ``mc_stratified_dynamic``.

This file deliberately owns the new workload family's diagnostics.  In
particular, it does not import the legacy workload or CLI helpers.  The only
workload dependency is the Plan-1 provider contract, loaded lazily so that the
pure metric helpers remain testable while the provider is developed in
parallel.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, is_dataclass
from hashlib import sha256
import importlib
import inspect
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence

from amc_py.experiments import evaluate_taskset, resolve_ordering
from amc_py.metrics import compute_lo_quality_weighted_metrics
from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SimulationResult


MANIFEST_SCHEMA_VERSION = "mc_stratified_dynamic_manifest_v1"
DIAGNOSTICS_SCHEMA_VERSION = "mc_stratified_dynamic_diagnostics_v1"
SELECTION_SCHEMA_VERSION = "mc_stratified_dynamic_selection_v1"

SCENARIO_SEEDS = tuple(range(200, 206))
DEFAULT_D1_SCENARIO_SEEDS = tuple(range(200, 202))
DEFAULT_END_TIME = 3_000_000
DEFAULT_D1_END_TIME = 1_000_000
AGENT_PERIOD = 25_000
INCREASE_RATIO = 0.02
DECREASE_RATIO = 0.02
BUDGET_FLOOR_RATIO = 0.90
XF = 0.50
LO_DEPLOY_CAP_RATIO = 4.0

# This is an explicit audit boundary.  A field with one of these prefixes may
# be copied into a diagnostics/audit CSV, but it can never be read by the
# selector's feature construction.
FORBIDDEN_SELECTION_PREFIXES = (
    "dqn_",
    "viper_",
    "heuristic_",
    "pressure_",
    "retention",
    "reward",
    "train_",
    "hout_",
)

SELECTION_FEATURES = (
    "load_p",
    "tightness_p",
    "autocorr_p",
    "stress_p",
    "leader_turnover_p",
    "mask_turnover_p",
    "competition_p",
    "mode_pressure_p",
)

# Kept as a literal, shared contract with the selector.  Hashing this object
# makes a diagnostics file auditable even when it is moved between machines.
SELECTION_CONFIG = {
    "schema_version": SELECTION_SCHEMA_VERSION,
    "features": SELECTION_FEATURES,
    "prototype_weights": {
        "load_p": 1.0,
        "tightness_p": 1.0,
        "autocorr_p": 1.0,
        "stress_p": 1.0,
        "leader_turnover_p": 1.0,
        "mask_turnover_p": 1.0,
        "competition_p": 1.0,
        "mode_pressure_p": 1.0,
    },
}


def canonical_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible configuration data."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def selection_config_hash() -> str:
    return canonical_hash(SELECTION_CONFIG)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "pass", "accepted"}:
        return True
    if text in {"0", "false", "no", "n", "off", "fail", "rejected"}:
        return False
    return default


def normalize_period_family(value: Any) -> str:
    """Normalize the two declared period families without guessing others."""

    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"semi-harmonic", "semiharmonic", "semi-harmonic-period"}:
        return "semi-harmonic"
    if text in {"log-uniform", "loguniform", "log-uniform-period"}:
        return "log-uniform"
    return text


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed unless every CSV row declares the new manifest schema."""

    if not rows:
        raise ValueError("manifest is empty")
    for index, row in enumerate(rows, start=2):
        if str(row.get("schema_version", "")).strip() != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest row {index} has unsupported schema_version="
                f"{row.get('schema_version')!r}; expected {MANIFEST_SCHEMA_VERSION!r}"
            )
        if str(row.get("candidate_seed", "")).strip() == "":
            raise ValueError(f"manifest row {index} is missing candidate_seed")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_manifest_rows(rows)
    return [dict(row) for row in rows]


def parse_seed_spec(raw: str) -> tuple[int, ...]:
    """Parse ``200:206`` as 200..205; comma-separated values are inclusive."""

    values: list[int] = []
    for part in raw.replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            start_text, stop_text = token.split(":", 1)
            start, stop = int(start_text), int(stop_text)
            step = 1 if stop >= start else -1
            values.extend(range(start, stop, step))
        else:
            values.append(int(token))
    result = tuple(dict.fromkeys(values))
    if not result:
        raise ValueError("scenario seed specification is empty")
    return result


def _json_object(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("configuration JSON must be an object")
    return parsed


def _provider_config_dict(base_config: Mapping[str, Any], row: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Merge config JSON and row-local generator fields without importing old code."""

    result: dict[str, Any] = {}
    for key in ("workload_config", "provider_config", "generator_config"):
        result.update(_json_object(base_config.get(key)))
    for key in ("workload_config_json", "provider_config_json", "generator_config_json"):
        result.update(_json_object(row.get(key)))
    # A manifest can carry explicit generator parameters as generator_* fields.
    for key, value in row.items():
        if key.startswith("generator_"):
            result[key[len("generator_") :]] = value
    result.setdefault("seed", seed)
    result.setdefault("candidate_seed", seed)
    if row.get("period_family") not in (None, ""):
        # The provider dataclass uses underscores; CSV/audit artifacts use
        # the human-readable hyphenated spelling.
        result.setdefault("period_family", normalize_period_family(row["period_family"]).replace("-", "_"))
    return result


def _accepted_kwargs(callable_object: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    """Filter kwargs for dataclass or ordinary constructor contracts."""

    if is_dataclass(callable_object):
        names = {item.name for item in fields(callable_object) if item.init}
        return {key: value for key, value in values.items() if key in names}
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return dict(values)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(values)
    return {key: value for key, value in values.items() if key in signature.parameters}


def load_stratified_dynamic_bundle(
    row: Mapping[str, Any],
    base_config: Mapping[str, Any],
    *,
    scenario_seed: int | None = None,
) -> Any:
    """Build one candidate through the Plan-1 provider contract.

    Import is intentionally lazy.  This function is the only place that knows
    how the parallel provider is loaded, and it fails loudly if that contract
    is not available.
    """

    module = importlib.import_module("amc_py.workloads.mc_stratified_dynamic")
    config_class = getattr(module, "MCStratifiedDynamicWorkloadConfig")
    provider_class = getattr(module, "MCStratifiedDynamicWorkloadProvider")
    seed = int(row["candidate_seed"])
    config_values = _provider_config_dict(base_config, row, seed)
    config_kwargs = _accepted_kwargs(config_class, config_values)
    try:
        workload_config = config_class(**config_kwargs)
    except TypeError:
        # Some provider implementations use candidate_seed rather than seed.
        config_values.pop("seed", None)
        config_kwargs = _accepted_kwargs(config_class, config_values)
        workload_config = config_class(**config_kwargs)

    provider_values: dict[str, Any] = {"config": workload_config}
    # The new provider is allowed to expose fixed_taskset_seed, matching the
    # workload-layer contract used by the formal experiments.  Prefer the
    # manifest seed for the taskset and the requested scenario seed for build().
    provider_values["fixed_taskset_seed"] = seed
    provider_values["scenario_seed_offset"] = 0
    provider_kwargs = _accepted_kwargs(provider_class, provider_values)
    try:
        provider = provider_class(**provider_kwargs)
    except TypeError:
        provider = provider_class(workload_config)
    bundle = provider.build(seed=seed if scenario_seed is None else int(scenario_seed))
    if not getattr(bundle, "tasks", None) or not getattr(bundle, "scenario", None):
        raise ValueError("mc_stratified_dynamic provider returned an incomplete WorkloadBundle")
    return bundle


def _metadata_value(metadata: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    return None


def initial_budgets_for_bundle(tasks: Sequence[Task], metadata: Mapping[str, Any] | None) -> dict[str, int]:
    """Read provider-declared initial budgets, falling back to Task.c_lo."""

    metadata = metadata or {}
    result: dict[str, int] = {task.name: int(task.c_lo) for task in tasks}
    by_name = _metadata_value(
        metadata,
        ("initial_budget_by_task", "initial_budgets", "budget_by_task"),
    )
    if isinstance(by_name, Mapping):
        for task in tasks:
            if task.name in by_name:
                result[task.name] = max(1, int(float(by_name[task.name])))
    task_meta = _metadata_value(metadata, ("task_meta", "task_metadata"))
    if isinstance(task_meta, Mapping):
        task_meta = list(task_meta.values())
    if isinstance(task_meta, Sequence) and not isinstance(task_meta, (str, bytes)):
        for item in task_meta:
            if isinstance(item, Mapping):
                name = item.get("name")
                budget = item.get("initial_budget", item.get("budget"))
            else:
                name = getattr(item, "name", None)
                budget = getattr(item, "initial_budget", getattr(item, "budget", None))
            if name in result and budget is not None:
                result[str(name)] = max(1, int(float(budget)))
    return result


def static_characterization(bundle: Any, *, priority_policy: str = "dm") -> dict[str, Any]:
    """Compute D0 metrics from tasks and AMC-rtb only."""

    tasks = tuple(bundle.tasks)
    metadata = getattr(bundle, "metadata", None) or {}
    budgets = initial_budgets_for_bundle(tasks, metadata)
    total_lo = sum(float(task.c_lo) / task.period for task in tasks)
    hi_lo = sum(float(task.c_lo) / task.period for task in tasks if task.criticality is Criticality.HI)
    lo_lo = sum(float(task.c_lo) / task.period for task in tasks if task.criticality is Criticality.LO)
    hi_hi = sum(float(task.c_hi) / task.period for task in tasks if task.criticality is Criticality.HI)
    criticality_factors = [float(task.c_hi) / max(float(task.c_lo), 1.0) for task in tasks]
    initial_total = sum(float(budgets[task.name]) / task.period for task in tasks)
    initial_hi = sum(
        float(budgets[task.name]) / task.period
        for task in tasks
        if task.criticality is Criticality.HI
    )
    initial_lo = sum(
        float(budgets[task.name]) / task.period
        for task in tasks
        if task.criticality is Criticality.LO
    )

    schedulability = evaluate_taskset(tasks, method="amc_rtb", priority_policy=priority_policy)
    response_slacks = [
        float(task.deadline - schedulability.response_times[task.name])
        for task in tasks
        if task.name in schedulability.response_times
    ]
    normalized_slacks = [
        float(task.deadline - schedulability.response_times[task.name]) / float(task.deadline)
        for task in tasks
        if task.name in schedulability.response_times
    ]
    return {
        "num_tasks": len(tasks),
        "num_hi_tasks": sum(task.criticality is Criticality.HI for task in tasks),
        "num_lo_tasks": sum(task.criticality is Criticality.LO for task in tasks),
        "total_util_lo_mode": total_lo,
        "hi_util_lo_mode": hi_lo,
        "lo_util_lo_mode": lo_lo,
        "hi_util_hi_mode": hi_hi,
        "criticality_factor_mean": statistics.fmean(criticality_factors) if criticality_factors else 0.0,
        "criticality_factor_max": max(criticality_factors, default=0.0),
        "initial_budget_util_total": initial_total,
        "initial_budget_util_hi": initial_hi,
        "initial_budget_util_lo": initial_lo,
        "amc_rtb_min_slack": min(response_slacks, default=0.0),
        "amc_rtb_normalized_slack": min(normalized_slacks, default=0.0),
        "amc_rtb_schedulable": bool(schedulability.schedulable),
        "amc_rtb_response_details": schedulability.details,
        "_initial_budgets": budgets,
        "_ordered_tasks": tuple(resolve_ordering(tasks, priority_policy, "amc_rtb")),
    }


def lag1_autocorrelation(values: Sequence[float]) -> float:
    """Pearson lag-1 autocorrelation; constant/short sequences map to 0."""

    if len(values) < 3:
        return 0.0
    left = [float(value) for value in values[:-1]]
    right = [float(value) for value in values[1:]]
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator > 1e-12 else 0.0


def normalized_hamming_distance(previous: Sequence[bool], current: Sequence[bool]) -> float:
    """Normalized Hamming distance, including length mismatch as changed bits."""

    width = max(len(previous), len(current))
    if width == 0:
        return 0.0
    changed = sum(
        1
        for index in range(width)
        if (previous[index] if index < len(previous) else False)
        != (current[index] if index < len(current) else False)
    )
    return float(changed) / float(width)


def _consecutive_run_mean(flags: Sequence[bool]) -> float:
    runs: list[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return statistics.fmean(runs) if runs else 0.0


def _leader_turnover(winners_by_window: Mapping[int, str | None]) -> float:
    winners = [winner for _, winner in sorted(winners_by_window.items()) if winner is not None]
    if len(winners) < 2:
        return 0.0
    return sum(a != b for a, b in zip(winners, winners[1:])) / float(len(winners) - 1)


def _fraction_time_hi(result: SimulationResult) -> float:
    horizon = max(1, int(result.end_time))
    events: list[tuple[int, int, str]] = [(0, 0, "LO")]
    events.extend((int(event.switch_time), 0, "HI") for event in result.mode_switches)
    events.extend((int(event.recovery_time), 1, "LO") for event in result.mode_recoveries)
    events.sort(key=lambda item: (item[0], item[1]))
    hi_time = 0
    for index, (time, _, mode) in enumerate(events):
        next_time = events[index + 1][0] if index + 1 < len(events) else horizon
        start = max(0, min(horizon, time))
        stop = max(start, min(horizon, next_time))
        if mode == "HI":
            hi_time += stop - start
    return float(hi_time) / float(horizon)


def baseline_result_metrics(
    results: Sequence[SimulationResult],
    *,
    initial_budgets: Mapping[str, int],
    ordered_tasks: Sequence[Task],
    observation_window: int = AGENT_PERIOD,
) -> dict[str, float]:
    """Aggregate baseline service, temporal, mode and pressure metrics."""

    if not results:
        return {
            "baseline_lo_quality_qos": 0.0,
            "baseline_lo_zero_service_ratio": 1.0,
            "baseline_lo_cancellations_per_1m": 0.0,
            "baseline_mode_changes_per_1m": 0.0,
            "baseline_deadline_misses_sum": 0.0,
            "lo_cost_lag1_autocorr_mean": 0.0,
            "hi_cost_lag1_autocorr_mean": 0.0,
            "stress_duty_empirical_mean": 0.0,
            "stress_dwell_empirical_mean": 0.0,
            "stress_leader_turnover_rate": 0.0,
            "lo_pressure_leader_turnover_rate": 0.0,
            "mode_change_rate": 0.0,
            "hi_overrun_event_rate": 0.0,
            "fraction_time_hi_mode": 0.0,
        }

    qos_values: list[float] = []
    zero_values: list[float] = []
    autocorr_lo: list[float] = []
    autocorr_hi: list[float] = []
    stress_duty: list[float] = []
    stress_dwell: list[float] = []
    stress_leader_values: list[float] = []
    lo_leader_values: list[float] = []
    mode_changes = 0
    cancellations = 0
    deadline_misses = 0
    hi_overruns = 0
    mode_rate_values: list[float] = []
    hi_fraction_values: list[float] = []

    for result in results:
        quality = compute_lo_quality_weighted_metrics(result)
        qos_values.append(float(quality.lo_quality_qos))
        zero_values.append(float(quality.lo_zero_service_ratio))
        horizon = max(1, int(result.end_time))
        mode_changes += result.mode_change_count()
        cancellations += result.lo_job_cancellation_count()
        deadline_misses += len(result.deadline_misses)
        mode_rate_values.append(result.mode_change_count() * 1_000_000.0 / horizon)
        hi_fraction_values.append(_fraction_time_hi(result))

        by_task: dict[str, list[Any]] = {task.name: [] for task in ordered_tasks}
        for job in result.jobs:
            by_task.setdefault(job.task.name, []).append(job)
            original_cost = int(job.original_actual_cost or job.actual_cost)
            if job.task.criticality is Criticality.HI and original_cost > job.task.c_lo:
                hi_overruns += 1
        for jobs in by_task.values():
            jobs.sort(key=lambda job: (job.release_index, job.release_time))
            costs = [float(job.original_actual_cost or job.actual_cost) for job in jobs]
            if jobs:
                budget = float(initial_budgets.get(jobs[0].task.name, jobs[0].task.c_lo))
                flags = [cost > budget for cost in costs]
                stress_duty.append(sum(flags) / float(len(flags)))
                stress_dwell.append(_consecutive_run_mean(flags))
                if jobs[0].task.criticality is Criticality.LO:
                    autocorr_lo.append(lag1_autocorrelation(costs))
                else:
                    autocorr_hi.append(lag1_autocorrelation(costs))

        stress_winners: dict[int, tuple[float, str]] = {}
        lo_winners: dict[int, tuple[float, str]] = {}
        for job in result.jobs:
            budget = float(initial_budgets.get(job.task.name, job.task.c_lo))
            pressure = float(job.original_actual_cost or job.actual_cost) / max(budget, 1.0)
            window = int(job.release_time) // max(1, observation_window)
            current = stress_winners.get(window)
            if current is None or (pressure, job.task.name) > current:
                stress_winners[window] = (pressure, job.task.name)
            if job.task.criticality is Criticality.LO:
                current_lo = lo_winners.get(window)
                if current_lo is None or (pressure, job.task.name) > current_lo:
                    lo_winners[window] = (pressure, job.task.name)
        stress_leader_values.append(_leader_turnover({key: value[1] for key, value in stress_winners.items()}))
        lo_leader_values.append(_leader_turnover({key: value[1] for key, value in lo_winners.items()}))

    total_horizon = sum(max(1, int(result.end_time)) for result in results)
    return {
        "baseline_lo_quality_qos": statistics.fmean(qos_values) if qos_values else 0.0,
        "baseline_lo_zero_service_ratio": statistics.fmean(zero_values) if zero_values else 1.0,
        "baseline_lo_cancellations_per_1m": cancellations * 1_000_000.0 / total_horizon,
        "baseline_mode_changes_per_1m": mode_changes * 1_000_000.0 / total_horizon,
        "baseline_deadline_misses_sum": float(deadline_misses),
        "lo_cost_lag1_autocorr_mean": statistics.fmean(autocorr_lo) if autocorr_lo else 0.0,
        "hi_cost_lag1_autocorr_mean": statistics.fmean(autocorr_hi) if autocorr_hi else 0.0,
        "stress_duty_empirical_mean": statistics.fmean(stress_duty) if stress_duty else 0.0,
        "stress_dwell_empirical_mean": statistics.fmean(stress_dwell) if stress_dwell else 0.0,
        "stress_leader_turnover_rate": statistics.fmean(stress_leader_values) if stress_leader_values else 0.0,
        "lo_pressure_leader_turnover_rate": statistics.fmean(lo_leader_values) if lo_leader_values else 0.0,
        "mode_change_rate": statistics.fmean(mode_rate_values) if mode_rate_values else 0.0,
        "hi_overrun_event_rate": hi_overruns * 1_000_000.0 / total_horizon,
        "fraction_time_hi_mode": statistics.fmean(hi_fraction_values) if hi_fraction_values else 0.0,
    }


def _mask_shape(mask: Sequence[bool], actions: Sequence[Any], tasks: Sequence[Task]) -> tuple[int, int, int]:
    valid = 0
    valid_lo_increase = 0
    valid_lo_decrease = 0
    for is_valid, action in zip(mask, actions, strict=True):
        if not is_valid:
            continue
        valid += 1
        if action.increase_task is not None and not action.decrease_tasks:
            task = tasks[action.increase_idx] if action.increase_idx is not None else None
            if task is not None and task.criticality is Criticality.LO:
                valid_lo_increase += 1
        if action.increase_task is None and len(action.decrease_tasks) == 1:
            task = tasks[action.decrease_indices[0]] if action.decrease_indices else None
            if task is not None and task.criticality is Criticality.LO:
                valid_lo_decrease += 1
    return valid, valid_lo_increase, valid_lo_decrease


def one_step_competition_probe(
    *,
    mask_before: Sequence[bool],
    actions: Sequence[Any],
    budget_before: Mapping[str, int],
    mask_for_budget: Callable[[Mapping[str, int]], Sequence[bool]],
    tasks_by_name: Mapping[str, Task],
    increase_ratio: float = INCREASE_RATIO,
) -> list[float]:
    """Probe every legal LO increase on a copied budget vector.

    ``mask_for_budget`` must be a side-effect-free/call-and-restore adapter.
    The helper itself never mutates ``budget_before`` and never applies an
    update to the live runtime.
    """

    before_count = sum(
        bool(is_valid)
        and action.increase_task is not None
        and not action.decrease_tasks
        and tasks_by_name[action.increase_task].criticality is Criticality.LO
        for is_valid, action in zip(mask_before, actions, strict=True)
    )
    if before_count <= 0:
        return []
    values: list[float] = []
    for is_valid, action in zip(mask_before, actions, strict=True):
        if not is_valid or action.increase_task is None or action.decrease_tasks:
            continue
        task = tasks_by_name[action.increase_task]
        if task.criticality is not Criticality.LO:
            continue
        candidate = dict(budget_before)
        current = int(candidate[action.increase_task])
        candidate[action.increase_task] = min(
            int(task.deadline),
            max(current + 1, math.ceil(current * (1.0 + increase_ratio))),
        )
        mask_after = mask_for_budget(candidate)
        after_count = sum(
            bool(after_valid)
            and after_action.increase_task is not None
            and not after_action.decrease_tasks
            and tasks_by_name[after_action.increase_task].criticality is Criticality.LO
            for after_valid, after_action in zip(mask_after, actions, strict=True)
        )
        values.append(max(0.0, before_count - after_count) / max(float(before_count), 1.0))
    return values


def collect_runtime_characterization(
    bundle: Any,
    static: Mapping[str, Any],
    *,
    scenario_seeds: Sequence[int] = SCENARIO_SEEDS,
    end_time: int = DEFAULT_END_TIME,
    priority_policy: str = "dm",
    scenario_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Run no-action C-AMC-sem baseline plus read-only masks and probes."""

    tasks = tuple(bundle.tasks)
    ordered_tasks = tuple(static["_ordered_tasks"])
    runtime_config = RuntimeConfig(
        end_time=end_time,
        semantics=RuntimeSemantics.C_AMC_SEM,
        capture_trace=False,
        capture_debug_events=False,
        record_dropped_lo_releases=True,
        drop_lo_jobs_on_hi_switch=False,
        c_amc_sem_lo_degradation_ratio=XF,
        c_amc_sem_primary_on_switch_time=True,
    )
    all_results: list[SimulationResult] = []
    all_masks: list[tuple[bool, ...]] = []
    mask_sequences: list[list[tuple[bool, ...]]] = []
    valid_counts: list[int] = []
    valid_lo_increases: list[int] = []
    valid_lo_decreases: list[int] = []
    competition_values: list[float] = []
    budget_mutation_violations = 0

    for scenario_seed in scenario_seeds:
        scenario = bundle.scenario
        # The provider bundle may expose a scenario factory; the CLI also
        # supplies one when the provider supports fixed taskset seeds.  Without
        # an explicit factory there is no safe way to rebuild the scenario, so
        # retaining the bundle scenario is deterministic and fail-closed.
        bundle_scenario_factory = getattr(bundle, "scenario_for_seed", None)
        if callable(scenario_factory):
            scenario = scenario_factory(int(scenario_seed))
        elif callable(bundle_scenario_factory):
            scenario = bundle_scenario_factory(int(scenario_seed))
        elif int(scenario_seed) != int(getattr(bundle, "scenario_seed", scenario_seed)):
            # A provider may encode a scenario seed in metadata.  Without an
            # explicit factory there is no safe way to rebuild it, so retaining
            # the bundle scenario is deterministic and fail-closed.
            scenario = bundle.scenario

        env = AmcBudgetEnv(
            ordered_tasks=ordered_tasks,
            scenario=scenario,
            runtime_config=runtime_config,
            agent_period=AGENT_PERIOD,
            check_safety=True,
            normalization_bounds=getattr(bundle, "normalization_bounds", None),
            reward_mode="mendes",
            action_space="single",
            mask_detail_mode="full",
            budget_increase_ratio=INCREASE_RATIO,
            budget_decrease_ratio=DECREASE_RATIO,
            include_explicit_noop=False,
            budget_floor_ratio=BUDGET_FLOOR_RATIO,
            forbid_decreasing_hi_budgets=True,
            enable_deploy_cap_mask=True,
            deploy_cap_mask_ratio=LO_DEPLOY_CAP_RATIO,
            deploy_cap_mask_criticality="lo",
        )
        env.reset(seed=int(scenario_seed))
        actions = tuple(env._actions)  # fixed, public action semantics are audited by the env itself
        tasks_by_name = {task.name: task for task in ordered_tasks}
        scenario_masks: list[tuple[bool, ...]] = []
        while True:
            mask = tuple(bool(value) for value in env.valid_action_mask())
            all_masks.append(mask)
            scenario_masks.append(mask)
            shape = _mask_shape(mask, actions, ordered_tasks)
            valid_counts.append(shape[0])
            valid_lo_increases.append(shape[1])
            valid_lo_decreases.append(shape[2])
            before_budgets = dict(env._engine.runtime_budgets.budgets)  # type: ignore[union-attr]

            def mask_for_budget(candidate: Mapping[str, int]) -> Sequence[bool]:
                nonlocal budget_mutation_violations
                engine = env._engine
                if engine is None:
                    raise RuntimeError("environment engine missing during counterfactual probe")
                saved = dict(engine.runtime_budgets.budgets)
                try:
                    engine.runtime_budgets.budgets.clear()
                    engine.runtime_budgets.budgets.update({name: int(value) for name, value in candidate.items()})
                    return tuple(bool(value) for value in env.valid_action_mask())
                finally:
                    restored = dict(engine.runtime_budgets.budgets)
                    engine.runtime_budgets.budgets.clear()
                    engine.runtime_budgets.budgets.update(saved)
                    if restored == saved:
                        # This branch is intentionally empty; the comparison
                        # below is made after restoration as a stronger guard.
                        pass
                    if dict(engine.runtime_budgets.budgets) != saved:
                        budget_mutation_violations += 1

            competition_values.extend(
                one_step_competition_probe(
                    mask_before=mask,
                    actions=actions,
                    budget_before=before_budgets,
                    mask_for_budget=mask_for_budget,
                    tasks_by_name=tasks_by_name,
                )
            )
            step_result = env.step(None)  # explicit no-action path; never mutates budget
            if step_result.info.get("updates"):
                budget_mutation_violations += 1
            if step_result.done:
                break
        result = env.runtime_result
        if result.budget_update_events:
            budget_mutation_violations += len(result.budget_update_events)
        all_results.append(result)
        mask_sequences.append(scenario_masks)

    metrics = baseline_result_metrics(
        all_results,
        initial_budgets=static["_initial_budgets"],
        ordered_tasks=ordered_tasks,
        observation_window=AGENT_PERIOD,
    )
    mask_turnover = [
        normalized_hamming_distance(previous, current)
        for sequence in mask_sequences
        for previous, current in zip(sequence, sequence[1:])
    ]
    competition_summary = {
        "budget_competition_index": statistics.fmean(competition_values) if competition_values else 0.0,
        "budget_competition_p50": statistics.median(competition_values) if competition_values else 0.0,
        "budget_competition_p90": (
            _quantile(competition_values, 0.90) if competition_values else 0.0
        ),
        "budget_competition_max": max(competition_values, default=0.0),
    }
    metrics.update(competition_summary)
    metrics.update(
        {
            "valid_action_count_mean": statistics.fmean(valid_counts) if valid_counts else 0.0,
            "valid_action_count_std": statistics.pstdev(valid_counts) if len(valid_counts) > 1 else 0.0,
            "valid_lo_increase_count_mean": statistics.fmean(valid_lo_increases) if valid_lo_increases else 0.0,
            "valid_lo_increase_count_std": (
                statistics.pstdev(valid_lo_increases) if len(valid_lo_increases) > 1 else 0.0
            ),
            "valid_lo_decrease_count_mean": statistics.fmean(valid_lo_decreases) if valid_lo_decreases else 0.0,
            "valid_lo_decrease_count_std": (
                statistics.pstdev(valid_lo_decreases) if len(valid_lo_decreases) > 1 else 0.0
            ),
            "mask_turnover_rate": statistics.fmean(mask_turnover) if mask_turnover else 0.0,
            "lo_increase_mask_turnover_rate": _masked_action_turnover(
                mask_sequences, actions, ordered_tasks, kind="increase"
            ),
            "lo_decrease_mask_turnover_rate": _masked_action_turnover(
                mask_sequences, actions, ordered_tasks, kind="decrease"
            ),
            "mask_observation_count": len(all_masks),
            "characterization_scenario_count": len(scenario_seeds),
            "characterization_horizon": end_time,
            "budget_probe_mutation_violations": budget_mutation_violations,
            "baseline_budget_update_count": sum(len(result.budget_update_events) for result in all_results),
            "baseline_mode_change_count": sum(result.mode_change_count() for result in all_results),
        }
    )
    return metrics


def _masked_action_turnover(
    mask_sequences: Sequence[Sequence[Sequence[bool]]],
    actions: Sequence[Any],
    tasks: Sequence[Task],
    *,
    kind: str,
) -> float:
    indices: list[int] = []
    for index, action in enumerate(actions):
        if kind == "increase" and action.increase_task is not None and not action.decrease_tasks:
            if tasks[action.increase_idx].criticality is Criticality.LO:
                indices.append(index)
        elif kind == "decrease" and action.increase_task is None and len(action.decrease_tasks) == 1:
            if tasks[action.decrease_indices[0]].criticality is Criticality.LO:
                indices.append(index)
    values = [
        normalized_hamming_distance(
            tuple(previous[index] for index in indices),
            tuple(current[index] for index in indices),
        )
        for masks in mask_sequences
        for previous, current in zip(masks, masks[1:])
    ]
    return statistics.fmean(values) if values else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def deterministic_static_reservoir(rows: Sequence[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    """Select a quantile-spread reservoir using static fields and seed only."""

    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    ordered = sorted(
        rows,
        key=lambda row: (
            _as_float(row.get("total_util_lo_mode")),
            _as_float(row.get("amc_rtb_normalized_slack")),
            _as_float(row.get("hi_util_hi_mode")),
            int(row["candidate_seed"]),
        ),
    )
    selected: list[Mapping[str, Any]] = []
    for position in range(limit):
        index = round(position * (len(ordered) - 1) / max(limit - 1, 1))
        selected.append(ordered[index])
    unique: dict[int, Mapping[str, Any]] = {int(row["candidate_seed"]): row for row in selected}
    if len(unique) < limit:
        for row in ordered:
            unique.setdefault(int(row["candidate_seed"]), row)
            if len(unique) == limit:
                break
    return [unique[key] for key in sorted(unique)]


def _runtime_metric_defaults() -> dict[str, float | int]:
    names = (
        "baseline_lo_quality_qos",
        "baseline_lo_zero_service_ratio",
        "baseline_lo_cancellations_per_1m",
        "baseline_mode_changes_per_1m",
        "baseline_deadline_misses_sum",
        "lo_cost_lag1_autocorr_mean",
        "hi_cost_lag1_autocorr_mean",
        "stress_duty_empirical_mean",
        "stress_dwell_empirical_mean",
        "stress_leader_turnover_rate",
        "lo_pressure_leader_turnover_rate",
        "valid_action_count_mean",
        "valid_action_count_std",
        "valid_lo_increase_count_mean",
        "valid_lo_increase_count_std",
        "valid_lo_decrease_count_mean",
        "valid_lo_decrease_count_std",
        "mask_turnover_rate",
        "lo_increase_mask_turnover_rate",
        "lo_decrease_mask_turnover_rate",
        "budget_competition_index",
        "budget_competition_p50",
        "budget_competition_p90",
        "budget_competition_max",
        "mode_change_rate",
        "hi_overrun_event_rate",
        "fraction_time_hi_mode",
    )
    result: dict[str, float | int] = {name: 0.0 for name in names}
    result.update(
        {
            "mask_observation_count": 0,
            "characterization_scenario_count": 0,
            "characterization_horizon": 0,
            "budget_probe_mutation_violations": 0,
            "baseline_budget_update_count": 0,
            "baseline_mode_change_count": 0,
        }
    )
    return result


def build_diagnostic_row(
    manifest_row: Mapping[str, Any],
    static: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any],
    diagnostics_hash: str,
    runtime_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(manifest_row)
    row.update({key: value for key, value in static.items() if not key.startswith("_")})
    row["schema_version"] = DIAGNOSTICS_SCHEMA_VERSION
    row["input_schema_version"] = MANIFEST_SCHEMA_VERSION
    row["candidate_seed"] = int(manifest_row["candidate_seed"])
    row["period_family"] = normalize_period_family(
        manifest_row.get("period_family", base_config.get("period_family", ""))
    )
    generator_config = base_config.get("workload_config", base_config)
    row["generator_config_hash"] = canonical_hash(generator_config)
    row["diagnostics_config_hash"] = diagnostics_hash
    row["selection_config_hash"] = selection_config_hash()
    row["selection_feature_list"] = json.dumps(SELECTION_FEATURES, ensure_ascii=False)
    row["selection_forbidden_prefixes"] = json.dumps(FORBIDDEN_SELECTION_PREFIXES, ensure_ascii=False)
    row["runtime_characterized"] = bool(runtime_metrics is not None)
    if runtime_metrics is None:
        row.update(_runtime_metric_defaults())
    else:
        row.update(runtime_metrics)
    return row


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty diagnostics CSV")
    preferred = [
        "schema_version",
        "input_schema_version",
        "candidate_seed",
        "period_family",
        "runtime_characterized",
        "total_util_lo_mode",
        "hi_util_lo_mode",
        "lo_util_lo_mode",
        "hi_util_hi_mode",
        "criticality_factor_mean",
        "criticality_factor_max",
        "initial_budget_util_total",
        "initial_budget_util_hi",
        "initial_budget_util_lo",
        "amc_rtb_min_slack",
        "amc_rtb_normalized_slack",
        "amc_rtb_schedulable",
        "baseline_lo_quality_qos",
        "baseline_lo_zero_service_ratio",
        "baseline_lo_cancellations_per_1m",
        "baseline_mode_changes_per_1m",
        "baseline_deadline_misses_sum",
        "lo_cost_lag1_autocorr_mean",
        "hi_cost_lag1_autocorr_mean",
        "stress_duty_empirical_mean",
        "stress_dwell_empirical_mean",
        "stress_leader_turnover_rate",
        "lo_pressure_leader_turnover_rate",
        "valid_action_count_mean",
        "valid_action_count_std",
        "valid_lo_increase_count_mean",
        "valid_lo_increase_count_std",
        "valid_lo_decrease_count_mean",
        "valid_lo_decrease_count_std",
        "mask_turnover_rate",
        "lo_increase_mask_turnover_rate",
        "lo_decrease_mask_turnover_rate",
        "budget_competition_index",
        "budget_competition_p50",
        "budget_competition_p90",
        "budget_competition_max",
        "mode_change_rate",
        "hi_overrun_event_rate",
        "fraction_time_hi_mode",
        "generator_config_hash",
        "diagnostics_config_hash",
        "selection_config_hash",
        "selection_feature_list",
        "selection_forbidden_prefixes",
    ]
    columns = list(dict.fromkeys(preferred + [key for row in rows for key in row]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items()} for row in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-seed-column", default="candidate_seed")
    parser.add_argument("--config", type=Path, default=Path("configs/tasksets/mc_stratified_dynamic_v1_selector.json"))
    parser.add_argument("--stage", choices=("D0", "D1", "D2"), default="D2")
    parser.add_argument("--scenario-seeds", default="200:206")
    parser.add_argument("--end-time", type=int, default=DEFAULT_END_TIME)
    parser.add_argument("--d1-end-time", type=int, default=DEFAULT_D1_END_TIME)
    parser.add_argument("--d1-max-candidates", type=int, default=800)
    parser.add_argument("--d2-max-candidates", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run_cli(args: argparse.Namespace) -> None:
    rows = read_manifest(args.manifest)
    if args.manifest_seed_column != "candidate_seed":
        if args.manifest_seed_column not in rows[0]:
            raise ValueError(f"manifest seed column not found: {args.manifest_seed_column}")
        rows = [dict(row, candidate_seed=row[args.manifest_seed_column]) for row in rows]
        validate_manifest_rows(rows)
    config = _json_object(json.loads(args.config.read_text(encoding="utf-8")))
    scenario_seeds = parse_seed_spec(args.scenario_seeds)
    if args.stage == "D1":
        scenario_seeds = DEFAULT_D1_SCENARIO_SEEDS
        horizon = args.d1_end_time
    else:
        horizon = args.end_time
    diagnostics_hash = canonical_hash(
        {
            "runtime_semantics": "C_AMC_SEM",
            "xf": XF,
            "agent_period": AGENT_PERIOD,
            "increase_ratio": INCREASE_RATIO,
            "decrease_ratio": DECREASE_RATIO,
            "budget_floor_ratio": BUDGET_FLOOR_RATIO,
            "forbid_decreasing_hi_budgets": True,
            "mask_detail": "full",
            "lo_deploy_cap_ratio": LO_DEPLOY_CAP_RATIO,
            "scenario_seeds": scenario_seeds,
            "end_time": horizon,
            "stage": args.stage,
        }
    )

    static_entries: list[tuple[dict[str, str], Any, dict[str, Any]]] = []
    for manifest_row in rows:
        bundle = load_stratified_dynamic_bundle(manifest_row, config)
        static = static_characterization(bundle)
        static_entries.append((manifest_row, bundle, static))
    eligible_entries = static_entries
    if args.stage == "D1":
        eligible_entries = [
            item
            for item in static_entries
            if int(item[2]["num_tasks"]) >= 2
        ]
        eligible_entries = deterministic_static_reservoir(
            [dict(item[0], **{key: value for key, value in item[2].items() if not key.startswith("_")}) for item in eligible_entries],
            args.d1_max_candidates,
        )
        selected_seeds = {int(item["candidate_seed"]) for item in eligible_entries}
        eligible_entries = [item for item in static_entries if int(item[0]["candidate_seed"]) in selected_seeds]
    elif args.stage == "D2" and len(static_entries) > args.d2_max_candidates:
        reservoir = deterministic_static_reservoir(
            [dict(item[0], **{key: value for key, value in item[2].items() if not key.startswith("_")}) for item in static_entries],
            args.d2_max_candidates,
        )
        selected_seeds = {int(item["candidate_seed"]) for item in reservoir}
        eligible_entries = [item for item in static_entries if int(item[0]["candidate_seed"]) in selected_seeds]

    output_rows: list[dict[str, Any]] = []
    for manifest_row, bundle, static in eligible_entries:
        metrics = None
        if args.stage != "D0":
            metrics = collect_runtime_characterization(
                bundle,
                static,
                scenario_seeds=scenario_seeds,
                end_time=horizon,
                scenario_factory=lambda scenario_seed, row=manifest_row: load_stratified_dynamic_bundle(
                    row,
                    config,
                    scenario_seed=scenario_seed,
                ).scenario,
            )
        output_rows.append(
            build_diagnostic_row(
                manifest_row,
                static,
                base_config=config,
                diagnostics_hash=diagnostics_hash,
                runtime_metrics=metrics,
            )
        )
    write_csv(args.output, output_rows)


def main() -> None:
    run_cli(build_parser().parse_args())


if __name__ == "__main__":
    main()
