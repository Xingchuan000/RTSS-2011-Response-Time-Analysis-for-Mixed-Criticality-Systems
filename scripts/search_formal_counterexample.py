#!/usr/bin/env python3
"""Search concrete runtime counterexamples for formally UNRESOLVED seeds.

The search reuses the production C-AMC-sem runtime and deployed integer VIPER
policy from one formal proof workspace.  It is deliberately *not* a proof
procedure: a found trace is a concrete unsafe witness, while failure to find a
trace says nothing about safety.

The current search is targeted at the residual V10.17 blocker seen in s715 and
s2942:

    THETA_0__LO_SWITCH[0,0]__TARGET_ABNORMAL

For each theta-compatible target release the tool searches reachable LO-mode
prehistories.  Before the trigger, HI jobs are constrained to A <= C_LO so no
premature C-AMC-sem switch is manufactured; LO jobs and normal HI jobs are
varied over their formal actual-demand interval to steer both carry-in and the
deployed tree's budget/history state.  At the trigger the target is ABNORMAL,
and from that boundary through its deadline higher-priority demand is driven to
its formal upper envelope.  Every candidate is replayed by the real event
runtime and real deployed policy.

A row is a concrete counterexample only when all of the following hold:
  * no HI deadline miss occurs before the chosen trigger;
  * no LO->HI switch occurs before the trigger;
  * the first LO->HI switch occurs exactly at the trigger boundary; and
  * the selected target job misses its deadline.

Default workspace convention (Windows):
    D:\\AMC\\build\\formal\\s715_best_overall_v9_1_e2e

Examples:
    python scripts/search_formal_counterexample.py --seed 715 --trials 200
    python scripts/search_formal_counterexample.py --seed 2942 --trials 500
    python scripts/search_formal_counterexample.py --seed 715 --mode both --native-count 100000 --native-replay-top 64
    python scripts/search_formal_counterexample.py --seed 715 --trials 0 \
        --max-trigger-candidates 0
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import heapq
import json
from math import gcd
from pathlib import Path
import random
import sys
import time
from typing import Any


FORMAL_ROOT_DEFAULT = Path(r"D:\AMC\build\formal")
LEVEL_COUNT = 5


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _source_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _workspace_path(seed: int, formal_root: Path, formal_dir: Path | None) -> Path:
    workspace = formal_dir if formal_dir is not None else formal_root / f"s{seed}_best_overall_v9_1_e2e"
    workspace = workspace.resolve()
    required = workspace / "request" / "proof_request.json"
    if not required.is_file():
        raise FileNotFoundError(f"formal workspace not found: {workspace}")
    return workspace


def _load_formal_target(source_root: Path, workspace: Path) -> Any:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from formal_toolchain.adapters.target_factory import build_target

    request = _read_json(workspace / "request" / "proof_request.json")
    recipe = request["target_recipe"]
    return build_target(str(recipe["factory"]), dict(recipe["kwargs"]))


def _load_integer_policy(workspace: Path) -> Any:
    from amc_py.viper.fixed_point import fixed_point_config_from_dict
    from amc_py.viper.integer_tree import load_integer_tree_json
    from amc_py.viper.tree_policy import IntegerTreeBudgetPolicy

    tree_dir = workspace / "request" / "inputs" / "tree_artifact"
    metadata = _read_json(tree_dir / "metadata.json")
    feature_names = tuple(json.loads((tree_dir / "feature_names.json").read_text(encoding="utf-8")))
    action_definitions = list(json.loads((tree_dir / "action_definitions.json").read_text(encoding="utf-8")))
    config = fixed_point_config_from_dict(_read_json(tree_dir / "fixed_point_config.json"))
    model = load_integer_tree_json(tree_dir / "integer_tree.json")
    return IntegerTreeBudgetPolicy(model, metadata, feature_names, action_definitions, config)


def _formal_demand_bounds(target: Any) -> dict[str, tuple[int, int]]:
    raw = target.provenance.get("actual_demand_bounds_by_task")
    if not isinstance(raw, dict):
        raise ValueError("target provenance lacks actual_demand_bounds_by_task")
    return {str(name): (int(row["min"]), int(row["max"])) for name, row in raw.items()}


def _unresolved_cases(workspace: Path) -> list[dict[str, Any]]:
    receipts = _read_json(workspace / "verified" / "proof_receipts.json")
    rows: list[dict[str, Any]] = []
    for target_row in receipts.get("pcssc_targets", []):
        if not isinstance(target_row, dict) or target_row.get("status") == "PASS":
            continue
        target_name = str(target_row.get("target"))
        for case in target_row.get("tested_horizons", []):
            if not isinstance(case, dict) or case.get("status") == "PASS":
                continue
            if case.get("terminal") != "CASE_CONSISTENT":
                continue
            rows.append({"target": target_name, **case})
    return rows


def _select_case(workspace: Path, target_name: str | None) -> dict[str, Any]:
    rows = _unresolved_cases(workspace)
    if target_name is not None:
        rows = [row for row in rows if row["target"] == target_name]
    if not rows:
        raise ValueError("no unresolved CASE_CONSISTENT target found in proof receipts")

    preferred = [
        row
        for row in rows
        if int(row.get("theta", -1)) == 0
        and row.get("switch_kind") == "LO_SWITCH"
        and int(row.get("switch_lower", -1)) == 0
        and int(row.get("switch_upper", -1)) == 0
        and row.get("target_classification") == "ABNORMAL"
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(rows) == 1:
        return rows[0]
    details = [f"{row['target']}::{row.get('case_id')}" for row in rows]
    raise ValueError(f"multiple unresolved cases; pass --target. candidates={details}")


def _require_supported_case(case: dict[str, Any]) -> None:
    supported = (
        case.get("switch_kind") == "LO_SWITCH"
        and int(case.get("switch_lower", -1)) == 0
        and int(case.get("switch_upper", -1)) == 0
        and case.get("target_classification") == "ABNORMAL"
    )
    if not supported:
        raise ValueError(
            "counterexample search currently targets LO_SWITCH[0,0] + TARGET_ABNORMAL; "
            f"got {case.get('case_id')}"
        )


def _mode_value(mode: Any) -> str:
    return str(getattr(mode, "value", mode))


def _task_is_hi(task: Any) -> bool:
    return str(getattr(task.criticality, "value", task.criticality)) == "HI"


def _level_value(lower: int, upper: int, level: int) -> int:
    if not 0 <= int(level) < LEVEL_COUNT:
        raise ValueError(f"invalid demand level {level}")
    if upper <= lower:
        return int(lower)
    numerator = int(level)
    return int(lower + ((upper - lower) * numerator + (LEVEL_COUNT - 2)) // (LEVEL_COUNT - 1))


@dataclass(frozen=True)
class SearchGenome:
    early_levels: tuple[int, ...]
    recent_levels: tuple[int, ...]
    simultaneous_hi_abnormal: bool = True

    def key(self) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
        return self.early_levels, self.recent_levels, bool(self.simultaneous_hi_abnormal)

    def as_dict(self, task_names: tuple[str, ...]) -> dict[str, Any]:
        return {
            "early_levels": {name: int(level) for name, level in zip(task_names, self.early_levels)},
            "recent_levels": {name: int(level) for name, level in zip(task_names, self.recent_levels)},
            "simultaneous_hi_abnormal": bool(self.simultaneous_hi_abnormal),
        }


def _baseline_genomes(task_count: int) -> list[tuple[str, SearchGenome]]:
    def uniform(early: int, recent: int, simultaneous_hi_abnormal: bool = True) -> SearchGenome:
        return SearchGenome((early,) * task_count, (recent,) * task_count, simultaneous_hi_abnormal)

    return [
        ("target_only_abnormal_at_trigger", uniform(4, 4, False)),
        ("max_pressure", uniform(4, 4, True)),
        ("recent_spike", uniform(0, 4, True)),
        ("sustained_high", uniform(3, 4, True)),
        ("mid_pressure", uniform(2, 3, True)),
        ("minimum_history", uniform(0, 0, True)),
    ]


def _mutate_genome(rng: random.Random, genome: SearchGenome, target_priority: int) -> SearchGenome:
    early = list(genome.early_levels)
    recent = list(genome.recent_levels)
    # Most mutations focus on tasks that can affect target interference.  A
    # smaller share touches lower-priority jobs because they can still steer the
    # deployed policy's feature/history state before the trigger.
    mutations = rng.randint(1, 4)
    for _ in range(mutations):
        if rng.random() < 0.82:
            index = rng.randrange(target_priority + 1)
        else:
            index = rng.randrange(len(early))
        levels = recent if rng.random() < 0.7 else early
        current = levels[index]
        if rng.random() < 0.65:
            step = rng.choice((-1, 1))
            levels[index] = max(0, min(LEVEL_COUNT - 1, current + step))
        else:
            levels[index] = rng.randrange(LEVEL_COUNT)
    simultaneous = genome.simultaneous_hi_abnormal
    if rng.random() < 0.08:
        simultaneous = not simultaneous
    return SearchGenome(tuple(early), tuple(recent), simultaneous)


def _trigger_indices(task: Any, agent_period: int, theta: int, end_time: int) -> list[int]:
    period = int(task.period)
    g = gcd(period, int(agent_period))
    if int(theta) % g != 0:
        return []
    cycle = int(agent_period) // g
    residues = [n for n in range(cycle) if (n * period) % int(agent_period) == int(theta)]
    rows: list[int] = []
    for residue in residues:
        n = int(residue)
        while n * period + int(task.deadline) <= int(end_time):
            rows.append(n)
            n += cycle
    return sorted(set(rows))


def _stratified(values: list[int], maximum: int) -> list[int]:
    if maximum <= 0 or len(values) <= maximum:
        return list(values)
    if maximum == 1:
        return [values[0]]
    positions = {round(i * (len(values) - 1) / (maximum - 1)) for i in range(maximum)}
    positions.update({0, len(values) - 1})
    return [values[i] for i in sorted(positions)]


def _build_scenario(
    target: Any,
    target_name: str,
    trigger_index: int,
    bounds: dict[str, tuple[int, int]],
    genome: SearchGenome,
    recent_window: int,
):
    from amc_py.runtime_scenarios import ExecutionScenario

    tasks = tuple(target.ordered_tasks)
    names = tuple(str(task.name) for task in tasks)
    priority = {name: index for index, name in enumerate(names)}
    target_priority = priority[target_name]
    target_task = tasks[target_priority]
    trigger_time = int(trigger_index) * int(target_task.period)

    def pretrigger_range(task: Any) -> tuple[int, int]:
        lower, formal_upper = bounds[str(task.name)]
        if _task_is_hi(task):
            return int(lower), int(min(formal_upper, int(task.c_lo)))
        return int(lower), int(formal_upper)

    def resolver(task: Any, release_index: int) -> int:
        name = str(task.name)
        release_time = int(release_index) * int(task.period)
        p = priority[name]
        lower, formal_upper = bounds[name]

        # The selected target release is the ABNORMAL job from the unresolved
        # formal case.  Use the formal upper demand, not a synthetic overrun.
        if name == target_name and int(release_index) == int(trigger_index):
            if int(formal_upper) <= int(task.c_lo):
                raise ValueError(f"target {name} has no ABNORMAL actual-demand support")
            return int(formal_upper)

        if release_time < trigger_time:
            pre_lower, pre_upper = pretrigger_range(task)
            last_before = (trigger_time - 1) // int(task.period)
            recent = (int(last_before) - int(release_index)) < int(recent_window)
            level = genome.recent_levels[p] if recent else genome.early_levels[p]
            return _level_value(pre_lower, pre_upper, level)

        # Optionally keep every *other* HI arrival at the trigger NORMAL so the
        # target itself is the unique semi-clairvoyant switch trigger.  This is
        # useful for producing a cleaner counterexample than a simultaneous
        # multi-HI stress arrival.
        if (
            release_time == trigger_time
            and name != target_name
            and _task_is_hi(task)
            and not genome.simultaneous_hi_abnormal
        ):
            return int(min(formal_upper, int(task.c_lo)))

        # From the trigger through the target deadline, maximize only demand that
        # can directly interfere with the target.  Lower-priority work is kept at
        # its formal minimum; it cannot preempt the target and otherwise only
        # adds search noise.
        if p <= target_priority:
            return int(formal_upper)
        return int(lower)

    return ExecutionScenario(
        name=f"formal_cex_{target_name}_release_{trigger_index}",
        resolver=resolver,
    )


def _remaining_carry_jobs(env: Any, target_priority: int, trigger_time: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total = 0
    priority_by_name = {str(task.name): index for index, task in enumerate(env.ordered_tasks)}
    for job in env._engine.state.active_jobs:
        if priority_by_name[str(job.task.name)] >= target_priority or int(job.release_time) >= trigger_time:
            continue
        if job.dropped or job.completion_time is not None:
            continue
        raw = int(job.original_actual_cost if job.original_actual_cost is not None else job.actual_cost)
        if _task_is_hi(job.task):
            cap = int(job.actual_cost)
        else:
            budget = int(job.runtime_budget_at_release if job.runtime_budget_at_release is not None else job.task.c_lo)
            cap = min(raw, budget + 1)
        remaining = max(0, cap - int(job.executed_time))
        if remaining <= 0:
            continue
        rows.append(
            {
                "task": str(job.task.name),
                "release_index": int(job.release_index),
                "release_time": int(job.release_time),
                "actual_cost": int(job.actual_cost),
                "runtime_budget_at_release": job.runtime_budget_at_release,
                "executed_time": int(job.executed_time),
                "effective_remaining": int(remaining),
            }
        )
        total += remaining
    return rows, int(total)


def _run_candidate(
    base_target: Any,
    policy: Any,
    target_name: str,
    trigger_index: int,
    bounds: dict[str, tuple[int, int]],
    genome: SearchGenome,
    recent_window: int,
    action_trace_radius: int,
) -> dict[str, Any]:
    env = copy.deepcopy(base_target.runtime_adapter.environment)
    env.runtime_config = replace(
        env.runtime_config,
        capture_trace=False,
        capture_debug_events=False,
        stop_at_first_miss=False,
    )
    tasks = tuple(env.ordered_tasks)
    names = tuple(str(task.name) for task in tasks)
    target_priority = names.index(target_name)
    target_task = tasks[target_priority]
    trigger_time = int(trigger_index) * int(target_task.period)
    stop_time = trigger_time + int(target_task.deadline)
    env.scenario = _build_scenario(base_target, target_name, trigger_index, bounds, genome, recent_window)

    obs = env.reset()
    trigger_snapshot: dict[str, Any] | None = None
    action_rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()

    while int(env._engine.current_time) < stop_time and not env._done:
        now = int(env._engine.current_time)
        mask = tuple(bool(value) for value in env.formal_valid_action_mask())
        action_id, trace = policy.select_action_id(tuple(obs.state_vector), mask, include_decision_trace=True)
        if abs(now - trigger_time) <= int(action_trace_radius) * int(env.agent_period):
            action_rows.append(
                {
                    "time": now,
                    "action_id": int(action_id),
                    "tree_leaf_id": trace.get("tree_leaf_id"),
                    "tree_selected_rank": trace.get("tree_selected_rank"),
                    "valid_action_count": sum(mask),
                }
            )
        step = env.step(action_id)
        obs = step.observation

        if trigger_snapshot is None and int(env._engine.current_time) >= trigger_time:
            jobs, carry = _remaining_carry_jobs(env, target_priority, trigger_time)
            snapshot = env._engine.finish()
            switches_before = [event for event in snapshot.mode_switches if int(event.switch_time) < trigger_time]
            switches_at = [event for event in snapshot.mode_switches if int(event.switch_time) == trigger_time]
            hi_misses_before = [
                miss
                for miss in snapshot.deadline_misses
                if int(miss.absolute_deadline) <= trigger_time and str(miss.task).startswith("mc_sd_hi_")
            ]
            trigger_snapshot = {
                "time": int(env._engine.current_time),
                "mode_after_boundary": _mode_value(env._engine.state.mode),
                "concrete_hp_carry": int(carry),
                "carry_jobs": jobs,
                "switches_before_trigger": len(switches_before),
                "switches_at_trigger": len(switches_at),
                "hi_misses_before_trigger": len(hi_misses_before),
                "runtime_budgets": {str(k): int(v) for k, v in env._engine.runtime_budgets.budgets.items()},
            }

    result = env._engine.finish()
    matches = [
        job
        for job in result.jobs
        if str(job.task.name) == target_name and int(job.release_index) == int(trigger_index)
    ]
    if len(matches) != 1:
        target_outcome: dict[str, Any] = {"found": False}
    else:
        job = matches[0]
        deadline = int(job.absolute_deadline)
        completion = None if job.completion_time is None else int(job.completion_time)
        miss = completion is None or completion > deadline
        response = None if completion is None else completion - int(job.release_time)
        matching_misses = [
            miss
            for miss in result.deadline_misses
            if str(miss.task) == target_name and int(miss.release_index) == int(trigger_index)
        ]
        first_matching_miss = matching_misses[0] if matching_misses else None
        target_outcome = {
            "found": True,
            "release_time": int(job.release_time),
            "absolute_deadline": deadline,
            "actual_cost": int(job.actual_cost),
            "c_lo": int(job.task.c_lo),
            "c_hi": int(job.task.c_hi),
            "target_is_abnormal": int(job.actual_cost) > int(job.task.c_lo),
            "executed_time": int(job.executed_time),
            "completion_time": completion,
            "response_time": response,
            "deadline_slack": None if completion is None else deadline - completion,
            "remaining_at_end": max(0, int(job.actual_cost) - int(job.executed_time)),
            "remaining_at_miss": (
                None
                if first_matching_miss is None
                else max(0, int(job.actual_cost) - int(first_matching_miss.executed_at_miss))
            ),
            "incomplete_or_late": bool(miss),
            "deadline_miss_recorded": bool(matching_misses),
            "deadline_miss_records": [
                {
                    "task": str(item.task),
                    "release_index": int(item.release_index),
                    "release_time": int(item.release_time),
                    "absolute_deadline": int(item.absolute_deadline),
                    "mode_at_miss": _mode_value(item.mode_at_miss),
                    "executed_at_miss": int(item.executed_at_miss),
                }
                for item in matching_misses
            ],
            "miss": bool(matching_misses),
        }

    first_switch = min(result.mode_switches, key=lambda event: int(event.switch_time), default=None)
    first_switch_time = None if first_switch is None else int(first_switch.switch_time)
    pre = trigger_snapshot or {}
    valid_prefix = bool(pre) and int(pre.get("switches_before_trigger", 1)) == 0 and int(pre.get("hi_misses_before_trigger", 1)) == 0
    switch_at_trigger = first_switch_time == trigger_time
    concrete_counterexample = bool(
        valid_prefix
        and switch_at_trigger
        and target_outcome.get("target_is_abnormal")
        and target_outcome.get("deadline_miss_recorded")
    )

    relevant_jobs = []
    for job in result.jobs:
        task_priority = names.index(str(job.task.name))
        if task_priority > target_priority:
            continue
        if int(job.absolute_deadline) < trigger_time or int(job.release_time) > stop_time:
            continue
        relevant_jobs.append({
            "task": str(job.task.name),
            "release_index": int(job.release_index),
            "release_time": int(job.release_time),
            "absolute_deadline": int(job.absolute_deadline),
            "actual_cost": int(job.actual_cost),
            "original_actual_cost": None if job.original_actual_cost is None else int(job.original_actual_cost),
            "runtime_budget_at_release": None if job.runtime_budget_at_release is None else int(job.runtime_budget_at_release),
            "executed_time": int(job.executed_time),
            "completion_time": None if job.completion_time is None else int(job.completion_time),
            "dropped": bool(job.dropped),
        })

    return {
        "trigger_index": int(trigger_index),
        "trigger_time": int(trigger_time),
        "valid_reachable_prefix": bool(valid_prefix),
        "first_switch_time": first_switch_time,
        "first_switch_at_trigger": bool(switch_at_trigger),
        "first_switch_triggering_task": None if first_switch is None else str(first_switch.triggering_task),
        "trigger_snapshot": trigger_snapshot,
        "target_outcome": target_outcome,
        "concrete_counterexample": concrete_counterexample,
        "actions_near_trigger": action_rows,
        "mode_switches": [
            {
                "switch_time": int(event.switch_time),
                "triggering_task": str(event.triggering_task),
                "triggering_release_index": int(event.triggering_release_index),
                "reason": str(event.reason),
            }
            for event in result.mode_switches
        ],
        "hi_deadline_misses": [
            {
                "task": str(miss.task),
                "release_index": int(miss.release_index),
                "release_time": int(miss.release_time),
                "absolute_deadline": int(miss.absolute_deadline),
                "mode_at_miss": _mode_value(miss.mode_at_miss),
                "executed_at_miss": int(miss.executed_at_miss),
            }
            for miss in result.deadline_misses
            if str(miss.task).startswith("mc_sd_hi_")
        ],
        "relevant_jobs": relevant_jobs,
        "simulation_wall_seconds": time.perf_counter() - wall_start,
    }


def _score(row: dict[str, Any]) -> float:
    out = row.get("target_outcome") or {}
    snap = row.get("trigger_snapshot") or {}
    score = 0.0
    if row.get("valid_reachable_prefix"):
        score += 1_000_000_000.0
    else:
        score -= 1_000_000_000.0
    if row.get("first_switch_at_trigger"):
        score += 100_000_000.0
    elif row.get("first_switch_time") is not None:
        score -= abs(int(row["first_switch_time"]) - int(row["trigger_time"]))
    if row.get("concrete_counterexample"):
        score += 1_000_000_000_000.0
    if out.get("found"):
        if out.get("completion_time") is None:
            score += 10_000_000.0 + 1000.0 * int(out.get("remaining_at_end", 0))
        else:
            score += float(out.get("response_time") or 0)
        score += 1000.0 * int(out.get("remaining_at_end", 0))
    score += float(snap.get("concrete_hp_carry", 0))
    return score


def _trial_record(
    name: str,
    trigger_index: int,
    genome: SearchGenome,
    task_names: tuple[str, ...],
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trial_name": name,
        "score": _score(row),
        "trigger_index": int(trigger_index),
        "genome": genome.as_dict(task_names),
        **row,
    }


def _compact_trial(row: dict[str, Any]) -> dict[str, Any]:
    outcome = row.get("target_outcome") or {}
    snapshot = row.get("trigger_snapshot") or {}
    return {
        "trial_name": row.get("trial_name"),
        "score": row.get("score"),
        "trigger_index": row.get("trigger_index"),
        "trigger_time": row.get("trigger_time"),
        "valid_reachable_prefix": row.get("valid_reachable_prefix"),
        "first_switch_time": row.get("first_switch_time"),
        "first_switch_triggering_task": row.get("first_switch_triggering_task"),
        "simultaneous_hi_abnormal": (row.get("genome") or {}).get("simultaneous_hi_abnormal"),
        "target_is_abnormal": outcome.get("target_is_abnormal"),
        "target_miss": outcome.get("miss"),
        "target_response_time": outcome.get("response_time"),
        "target_deadline_slack": outcome.get("deadline_slack"),
        "target_remaining_at_end": outcome.get("remaining_at_end"),
        "target_remaining_at_miss": outcome.get("remaining_at_miss"),
        "concrete_hp_carry": snapshot.get("concrete_hp_carry"),
        "simulation_wall_seconds": row.get("simulation_wall_seconds"),
    }



def _workspace_seed(workspace: Path) -> int:
    return int(_read_json(workspace / "proof_result.json")["taskset_seed"])


def _build_native_workload(source_root: Path, workspace: Path, seed: int) -> Any:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from amc_py.dqn.experiment import build_mc_stratified_dynamic_experiment_config
    from amc_py.workloads.mc_stratified_dynamic import build_mc_stratified_dynamic_workload

    request = _read_json(workspace / "request" / "proof_request.json")
    recipe = request["target_recipe"]
    if "mc_stratified_dynamic_target:build_target" not in str(recipe["factory"]):
        raise ValueError("native search is implemented only for mc_stratified_dynamic targets")
    config = build_mc_stratified_dynamic_experiment_config(**dict(recipe["kwargs"]["workload_args"]))
    provider = config.workload_provider
    if provider is None:
        raise ValueError("mc_stratified_dynamic experiment has no workload provider")
    return build_mc_stratified_dynamic_workload(replace(provider.config, seed=int(seed)))


def _serialize_native_miss(miss: Any) -> dict[str, Any]:
    return {
        "task": str(miss.task),
        "release_index": int(miss.release_index),
        "release_time": int(miss.release_time),
        "absolute_deadline": int(miss.absolute_deadline),
        "mode_at_miss": _mode_value(miss.mode_at_miss),
        "executed_at_miss": int(miss.executed_at_miss),
    }


def _native_release0_pressure(
    scenario: Any,
    tasks: tuple[Any, ...],
    target_priority: int,
    bounds: dict[str, tuple[int, int]],
) -> tuple[float, dict[str, int]]:
    values: dict[str, int] = {}
    score = 0.0
    for task in tasks[: target_priority + 1]:
        name = str(task.name)
        actual = int(scenario.actual_cost_for(task, 0))
        values[name] = actual
        lower, upper = bounds[name]
        score += (actual - lower) / max(1, upper - lower)
        if _task_is_hi(task) and actual > int(task.c_lo):
            score += 0.5
    return score, values


def _run_native_release0_candidate(
    base_target: Any,
    policy: Any,
    scenario: Any,
    target_name: str,
) -> dict[str, Any]:
    env = copy.deepcopy(base_target.runtime_adapter.environment)
    env.runtime_config = replace(
        env.runtime_config,
        end_time=int(next(task for task in env.ordered_tasks if str(task.name) == target_name).deadline) + 1,
        capture_trace=False,
        capture_debug_events=False,
        stop_at_first_miss=False,
    )
    env.scenario = scenario
    obs = env.reset()
    actions: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    while not env._done:
        now = int(env._engine.current_time)
        mask = tuple(bool(value) for value in env.formal_valid_action_mask())
        action_id, trace = policy.select_action_id(tuple(obs.state_vector), mask, include_decision_trace=True)
        actions.append(
            {
                "time": now,
                "action_id": int(action_id),
                "tree_leaf_id": trace.get("tree_leaf_id"),
                "tree_selected_rank": trace.get("tree_selected_rank"),
            }
        )
        obs = env.step(int(action_id)).observation

    result = env._engine.finish()
    target_jobs = [
        job for job in result.jobs
        if str(job.task.name) == target_name and int(job.release_index) == 0
    ]
    target_outcome: dict[str, Any] = {"found": False}
    if len(target_jobs) == 1:
        job = target_jobs[0]
        miss_rows = [
            miss for miss in result.deadline_misses
            if str(miss.task) == target_name and int(miss.release_index) == 0
        ]
        target_outcome = {
            "found": True,
            "actual_cost": int(job.actual_cost),
            "c_lo": int(job.task.c_lo),
            "c_hi": int(job.task.c_hi),
            "target_is_abnormal": int(job.actual_cost) > int(job.task.c_lo),
            "absolute_deadline": int(job.absolute_deadline),
            "executed_time": int(job.executed_time),
            "completion_time": None if job.completion_time is None else int(job.completion_time),
            "deadline_miss_recorded": bool(miss_rows),
            "remaining_at_miss": None if not miss_rows else max(0, int(job.actual_cost) - int(miss_rows[0].executed_at_miss)),
        }
    first_switch = min(result.mode_switches, key=lambda event: int(event.switch_time), default=None)
    case_match = bool(
        target_outcome.get("target_is_abnormal")
        and first_switch is not None
        and int(first_switch.switch_time) == 0
    )
    concrete = bool(case_match and target_outcome.get("deadline_miss_recorded"))
    hi_misses = [
        _serialize_native_miss(miss)
        for miss in result.deadline_misses
        if str(miss.task).startswith("mc_sd_hi_")
    ]
    jobs = [
        {
            "task": str(job.task.name),
            "criticality": str(getattr(job.task.criticality, "value", job.task.criticality)),
            "release_index": int(job.release_index),
            "release_time": int(job.release_time),
            "absolute_deadline": int(job.absolute_deadline),
            "actual_cost": int(job.actual_cost),
            "c_lo": int(job.task.c_lo),
            "c_hi": int(job.task.c_hi),
            "runtime_budget_at_release": None if job.runtime_budget_at_release is None else int(job.runtime_budget_at_release),
            "executed_time": int(job.executed_time),
            "completion_time": None if job.completion_time is None else int(job.completion_time),
            "dropped": bool(job.dropped),
        }
        for job in result.jobs
        if int(job.release_time) <= int(target_outcome.get("absolute_deadline", 0))
    ]
    return {
        "case_match": case_match,
        "concrete_counterexample": concrete,
        "target_outcome": target_outcome,
        "first_mode_switch": None if first_switch is None else {
            "switch_time": int(first_switch.switch_time),
            "triggering_task": str(first_switch.triggering_task),
            "triggering_release_index": int(first_switch.triggering_release_index),
            "reason": str(first_switch.reason),
        },
        "hi_deadline_misses": hi_misses,
        "actions": actions,
        "jobs_through_target_deadline": jobs,
        "simulation_wall_seconds": time.perf_counter() - wall_start,
    }


def _native_search(
    source_root: Path,
    workspace: Path,
    base_target: Any,
    policy: Any,
    target_name: str,
    bounds: dict[str, tuple[int, int]],
    count: int,
    replay_top: int,
    seed_start: int | None,
    progress: bool,
) -> dict[str, Any]:
    if count <= 0:
        return {"status": "SKIPPED", "requested_count": int(count), "scan_count": 0, "replay_count": 0, "counterexample": None, "runs": []}
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from amc_py.workloads.mc_stratified_dynamic import build_mc_stratified_dynamic_execution_scenario

    taskset_seed = _workspace_seed(workspace)
    workload = _build_native_workload(source_root, workspace, taskset_seed)
    default_start = int(base_target.provenance.get("scenario_seed", taskset_seed + 100000))
    seed0 = default_start if seed_start is None else int(seed_start)
    tasks = tuple(base_target.ordered_tasks)
    names = tuple(str(task.name) for task in tasks)
    target_priority = names.index(target_name)
    target_task = tasks[target_priority]

    # Native search is intentionally targeted at release 0.  The unresolved
    # cases in the current 10-seed experiment all have theta=0, so release 0 is
    # a legitimate instance of the exact formal case and lets us scan very many
    # Markov scenario seeds without simulating 8,000,000 ticks each.
    heap: list[tuple[float, int, dict[str, int]]] = []
    abnormal_candidates = 0
    scan_started = time.perf_counter()
    keep = max(1, int(replay_top))
    for offset in range(int(count)):
        scenario_seed = int(seed0 + offset)
        scenario = build_mc_stratified_dynamic_execution_scenario(workload, scenario_seed=scenario_seed)
        target_actual = int(scenario.actual_cost_for(target_task, 0))
        if target_actual <= int(target_task.c_lo):
            continue
        abnormal_candidates += 1
        score, values = _native_release0_pressure(scenario, tasks, target_priority, bounds)
        item = (float(score), scenario_seed, values)
        if len(heap) < keep:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)
    ranked = sorted(heap, reverse=True)

    runs: list[dict[str, Any]] = []
    witness: dict[str, Any] | None = None
    for rank, (pressure, scenario_seed, release0_values) in enumerate(ranked, 1):
        scenario = build_mc_stratified_dynamic_execution_scenario(workload, scenario_seed=scenario_seed)
        row = _run_native_release0_candidate(base_target, policy, scenario, target_name)
        compact = {
            "rank": rank,
            "scenario_seed": int(scenario_seed),
            "release0_pressure": float(pressure),
            "release0_target_actual": int(release0_values[target_name]),
            "case_match": bool(row["case_match"]),
            "target_miss": bool(row["target_outcome"].get("deadline_miss_recorded")),
            "remaining_at_miss": row["target_outcome"].get("remaining_at_miss"),
            "first_mode_switch": row["first_mode_switch"],
            "simulation_wall_seconds": row["simulation_wall_seconds"],
        }
        runs.append(compact)
        if progress:
            print(json.dumps({"native_replay": compact}, ensure_ascii=False), flush=True)
        if row["concrete_counterexample"]:
            witness = {
                "kind": "NATIVE_MARKOV_COUNTEREXAMPLE",
                "taskset_seed": taskset_seed,
                "scenario_seed": int(scenario_seed),
                "release0_pressure": float(pressure),
                "release0_actual_costs": release0_values,
                **row,
            }
            break

    return {
        "status": "COUNTEREXAMPLE_FOUND" if witness is not None else "NO_COUNTEREXAMPLE_FOUND_IN_FINITE_SEARCH",
        "strategy": "SCAN_NATIVE_RELEASE0_PRESSURE_THEN_REPLAY_TOP",
        "scenario_seed_start": int(seed0),
        "requested_count": int(count),
        "scan_count": int(count),
        "abnormal_target_candidate_count": int(abnormal_candidates),
        "replay_top": int(keep),
        "replay_count": len(runs),
        "scan_wall_seconds": time.perf_counter() - scan_started,
        "counterexample": witness,
        "runs": runs,
        "warning": "absence of a native counterexample is empirical evidence only, not a safety proof",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Formal counterexample search",
        "",
        f"Seed: `{report['seed']}`",
        f"Target: `{report['target']}`",
        f"Formal blocker: `{report['formal_case']['case_id']}`",
        f"Overall verdict: **{report['verdict']}**",
        "",
        "## Formal-envelope targeted search",
        "",
        f"Status: **{report['envelope_search']['status']}**",
        "",
        "| rank | trigger | simultaneous HI abnormal | valid prefix | switch@trigger | target miss | response | slack | carry |",
        "|---:|---:|:---:|:---:|:---:|:---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report.get("top_trials", []), 1):
        lines.append(
            f"| {rank} | {row.get('trigger_time')} | {row.get('simultaneous_hi_abnormal')} | "
            f"{row.get('valid_reachable_prefix')} | {row.get('first_switch_time') == row.get('trigger_time')} | "
            f"{row.get('target_miss')} | {row.get('target_response_time')} | "
            f"{row.get('target_deadline_slack')} | {row.get('concrete_hp_carry')} |"
        )
    native = report.get("native_search") or {}
    lines += [
        "",
        "## Native Markov scenario search",
        "",
        f"Status: **{native.get('status', 'SKIPPED')}**, scanned={native.get('scan_count', 0)}, replayed={native.get('replay_count', 0)}, "
        f"start_seed={native.get('scenario_seed_start')}",
        "",
        "## Interpretation",
        "",
        "- `FORMAL_ENVELOPE_COUNTEREXAMPLE_FOUND`: a real deadline miss occurred in the production runtime using job demands inside the formal P0 integer envelope. This is sufficient to refute a universal P0 safety claim, but the exact joint demand sequence may be rare or absent under the native Markov generator.",
        "- `NATIVE_MARKOV_COUNTEREXAMPLE_FOUND`: the original mc_stratified_dynamic generator itself produced a deadline-miss trace; this is stronger distribution-realizable evidence.",
        "- No counterexample found by either finite search is **not** a safety proof.",
        "",
    ]
    return "\n".join(lines) + "\n"

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT_DEFAULT)
    parser.add_argument("--formal-dir", type=Path, default=None, help="explicit s<seed>_best_overall_v9_1_e2e directory")
    parser.add_argument("--target", default=None, help="override unresolved target auto-detection")
    parser.add_argument("--trials", type=int, default=200, help="mutated search trials after deterministic baselines")
    parser.add_argument("--recent-window", type=int, default=8, help="per-task releases controlled by recent_levels before trigger")
    parser.add_argument("--max-trigger-candidates", type=int, default=32, help="0 means all theta-compatible releases")
    parser.add_argument("--exclude-zero-trigger", action="store_true", help="search only non-empty prehistories")
    parser.add_argument("--elite-count", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--action-trace-radius", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("envelope", "native", "both"), default="envelope")
    parser.add_argument("--native-count", type=int, default=100000, help="native Markov scenario seeds to scan cheaply at release 0")
    parser.add_argument("--native-replay-top", type=int, default=64, help="highest-pressure native seeds to replay in the runtime")
    parser.add_argument("--native-seed-start", type=int, default=None)
    parser.add_argument("--quiet-native", action="store_true")
    args = parser.parse_args()

    if args.trials < 0 or args.recent_window <= 0 or args.elite_count <= 0 or args.native_count < 0 or args.native_replay_top <= 0:
        raise ValueError("invalid search size")

    source_root = _source_root()
    workspace = _workspace_path(args.seed, args.formal_root, args.formal_dir)
    formal_case = _select_case(workspace, args.target)
    _require_supported_case(formal_case)
    target_name = str(formal_case["target"])

    base_target = _load_formal_target(source_root, workspace)
    policy = _load_integer_policy(workspace)
    bounds = _formal_demand_bounds(base_target)
    tasks = tuple(base_target.ordered_tasks)
    task_names = tuple(str(task.name) for task in tasks)
    target_priority = task_names.index(target_name)
    target_task = tasks[target_priority]
    theta = int(formal_case["theta"])
    agent_period = int(base_target.runtime_config.agent_period)
    end_time = int(base_target.runtime_config.end_time)

    all_triggers = _trigger_indices(target_task, agent_period, theta, end_time)
    if args.exclude_zero_trigger:
        all_triggers = [index for index in all_triggers if index > 0]
    triggers = _stratified(all_triggers, int(args.max_trigger_candidates))
    if not triggers:
        raise ValueError("no runtime target release satisfies the unresolved theta and end_time")

    output_dir = args.output_dir or workspace / "counterexample_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "search_trials.jsonl"
    trials_path.write_text("", encoding="utf-8")

    rng = random.Random(int(args.random_seed))
    evaluated: list[tuple[float, int, SearchGenome, dict[str, Any]]] = []
    seen: set[tuple[int, tuple[tuple[int, ...], tuple[int, ...], bool]]] = set()
    counterexample: dict[str, Any] | None = None

    def evaluate(name: str, trigger_index: int, genome: SearchGenome) -> dict[str, Any] | None:
        nonlocal counterexample
        identity = (int(trigger_index), genome.key())
        if identity in seen:
            return None
        seen.add(identity)
        runtime_row = _run_candidate(
            base_target,
            policy,
            target_name,
            int(trigger_index),
            bounds,
            genome,
            int(args.recent_window),
            int(args.action_trace_radius),
        )
        row = _trial_record(name, trigger_index, genome, task_names, runtime_row)
        evaluated.append((float(row["score"]), int(trigger_index), genome, row))
        with trials_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        compact = _compact_trial(row)
        print(json.dumps(compact, ensure_ascii=False), flush=True)
        if row.get("concrete_counterexample"):
            counterexample = row
        return row

    if args.mode in {"envelope", "both"}:
        # Deterministic profiles first.  Target-only abnormal is deliberately
        # tried before simultaneous HI stress to prefer the simplest witness.
        baselines = _baseline_genomes(len(tasks))
        for trigger_index in triggers:
            for profile_name, genome in baselines:
                evaluate(f"baseline:{profile_name}", trigger_index, genome)
                if counterexample is not None:
                    break
            if counterexample is not None:
                break

        # Best-first randomized local search over reachable prehistory demand levels.
        for trial_index in range(int(args.trials)):
            if counterexample is not None:
                break
            ranked = sorted(evaluated, key=lambda item: item[0], reverse=True)
            elites = ranked[: min(len(ranked), int(args.elite_count))]
            if not elites:
                break
            _, parent_trigger, parent_genome, _ = rng.choice(elites)
            genome = _mutate_genome(rng, parent_genome, target_priority)
            trigger_index = parent_trigger
            if len(triggers) > 1 and rng.random() < 0.25:
                trigger_index = rng.choice(triggers)
            evaluate(f"mutation:{trial_index}", trigger_index, genome)

    ranked_rows = [item[3] for item in sorted(evaluated, key=lambda item: item[0], reverse=True)]
    top = [_compact_trial(row) for row in ranked_rows[:20]]
    native = (
        _native_search(
            source_root,
            workspace,
            base_target,
            policy,
            target_name,
            bounds,
            int(args.native_count),
            int(args.native_replay_top),
            args.native_seed_start,
            not bool(args.quiet_native),
        )
        if args.mode in {"native", "both"}
        else {"status": "SKIPPED", "requested_count": 0, "scan_count": 0, "replay_count": 0, "counterexample": None, "runs": []}
    )
    native_counterexample = native.get("counterexample")
    if native_counterexample is not None:
        verdict = "NATIVE_MARKOV_COUNTEREXAMPLE_FOUND"
    elif counterexample is not None:
        verdict = "FORMAL_ENVELOPE_COUNTEREXAMPLE_FOUND"
    else:
        verdict = "NO_COUNTEREXAMPLE_FOUND_IN_FINITE_SEARCH"

    report = {
        "schema_version": "formal_counterexample_search_v2",
        "diagnostic_only": True,
        "seed": int(args.seed),
        "workspace": str(workspace),
        "framework_revision": _read_json(workspace / "verified" / "proof_receipts.json").get("framework_revision"),
        "target": target_name,
        "target_priority": int(target_priority),
        "target_period": int(target_task.period),
        "target_deadline": int(target_task.deadline),
        "formal_case": {
            "case_id": formal_case.get("case_id"),
            "theta": theta,
            "switch_kind": formal_case.get("switch_kind"),
            "switch_lower": formal_case.get("switch_lower"),
            "switch_upper": formal_case.get("switch_upper"),
            "target_classification": formal_case.get("target_classification"),
            "failure": formal_case.get("failure"),
            "candidate_path": formal_case.get("candidate_path"),
            "refinement_candidate_path": formal_case.get("refinement_candidate_path"),
        },
        "formal_demand_bounds": {
            name: {"min": int(low), "max": int(high)} for name, (low, high) in bounds.items()
        },
        "search": {
            "mode": args.mode,
            "formal_theta_compatible_trigger_count": len(all_triggers),
            "searched_trigger_indices": triggers if args.mode in {"envelope", "both"} else [],
            "recent_window": int(args.recent_window),
            "requested_mutation_trials": int(args.trials),
            "executed_envelope_trials": len(evaluated),
            "random_seed": int(args.random_seed),
        },
        "verdict": verdict,
        "envelope_search": {
            "status": "COUNTEREXAMPLE_FOUND" if counterexample is not None else (
                "NO_COUNTEREXAMPLE_FOUND_IN_FINITE_SEARCH" if args.mode in {"envelope", "both"} else "SKIPPED"
            ),
            "counterexample": counterexample,
            "executed_trials": len(evaluated),
        },
        "native_search": native,
        "best_trial": None if not ranked_rows else ranked_rows[0],
        "top_trials": top,
        "nonproof_warning": "absence of a counterexample in either finite search is not a safety proof",
    }
    (output_dir / "counterexample_search_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "counterexample_search_summary.md").write_text(_markdown(report), encoding="utf-8")
    if counterexample is not None:
        (output_dir / "envelope_counterexample.json").write_text(
            json.dumps(counterexample, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if native_counterexample is not None:
        (output_dir / "native_counterexample.json").write_text(
            json.dumps(native_counterexample, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    strongest = native_counterexample if native_counterexample is not None else counterexample
    if strongest is not None:
        (output_dir / "counterexample.json").write_text(
            json.dumps(strongest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps({
        "verdict": verdict,
        "executed_envelope_trials": len(evaluated),
        "native_seeds_scanned": native.get("scan_count", 0),
        "native_candidates_replayed": native.get("replay_count", 0),
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    return 0 if strongest is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
