#!/usr/bin/env python3
"""Bounded conditioned reachable-carry search for s603 / mc_sd_hi_0.

Diagnostic only.  The search targets the V10.12 failure class

    theta=0, LO_SWITCH[0,0], TARGET_ABNORMAL,
    no earlier LO->HI switch, no earlier HI deadline miss.

The tool runs the real deployed integer VIPER policy and formal mask/FirstValid
runtime under several admissible LO-mode demand histories.  Each history keeps
all HI jobs NORMAL before the trigger and is run once across all eligible
theta=0 target-release times, so the search observes concrete carry and budget
states without independently maximizing carry and switch history.

The best concrete states are then replayed from boot with the target made
ABNORMAL at the selected release and higher-priority future demand set to the
formal upper endpoint.  A replay counts as a counterexample only if the prefix
has no prior HI miss/switch, the first switch is exactly at the target release,
and the target misses its deadline.

The search is bounded.  Its maximum carry is NOT a formal upper bound.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from math import gcd
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _find_workspace(root: Path, seed: str) -> Path:
    direct = root / seed
    if (direct / "request/proof_request.json").is_file():
        return direct
    rows = sorted({p.parent.parent.resolve() for p in root.rglob("request/proof_request.json") if seed in str(p.parent.parent)})
    if len(rows) != 1:
        raise ValueError(f"expected one workspace matching {seed!r}, found {len(rows)}")
    return rows[0]


class WorkspaceInput:
    def __init__(self, path: Path, seed: str) -> None:
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        if path.is_dir():
            root = path.resolve()
        elif path.is_file() and path.suffix.lower() == ".zip":
            self._tmp = tempfile.TemporaryDirectory(prefix="s603_condcarry_")
            root = Path(self._tmp.name)
            with zipfile.ZipFile(path) as archive:
                archive.extractall(root)
        else:
            raise ValueError("--formal must be a proof-result directory or ZIP")
        self.workspace = _find_workspace(root, seed)

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


def _load_target(source_root: Path, workspace: Path) -> Any:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from formal_toolchain.adapters.target_factory import build_target

    request = _read_json(workspace / "request/proof_request.json")
    recipe = request["target_recipe"]
    return build_target(str(recipe["factory"]), dict(recipe["kwargs"]))


def _load_integer_policy(workspace: Path) -> Any:
    from amc_py.viper.fixed_point import fixed_point_config_from_dict
    from amc_py.viper.integer_tree import load_integer_tree_json
    from amc_py.viper.tree_policy import IntegerTreeBudgetPolicy

    tree_dir = workspace / "request/inputs/tree_artifact"
    metadata = _read_json(tree_dir / "metadata.json")
    feature_names = tuple(json.loads((tree_dir / "feature_names.json").read_text(encoding="utf-8")))
    action_definitions = list(json.loads((tree_dir / "action_definitions.json").read_text(encoding="utf-8")))
    config = fixed_point_config_from_dict(_read_json(tree_dir / "fixed_point_config.json"))
    model = load_integer_tree_json(tree_dir / "integer_tree.json")
    return IntegerTreeBudgetPolicy(model, metadata, feature_names, action_definitions, config)


def _demand_bounds(target: Any) -> dict[str, tuple[int, int]]:
    raw = target.provenance.get("actual_demand_bounds_by_task")
    if not isinstance(raw, dict):
        raise ValueError("target provenance lacks actual_demand_bounds_by_task")
    return {str(name): (int(row["min"]), int(row["max"])) for name, row in raw.items()}


def _proof_carry_bound(workspace: Path, target_name: str, deadline: int) -> int | None:
    receipts = _read_json(workspace / "verified/proof_receipts.json")
    rows = [row for row in receipts.get("pcssc_targets", []) if isinstance(row, dict) and row.get("target") == target_name]
    if len(rows) != 1:
        return None
    for tested in reversed(rows[0].get("tested_horizons", [])):
        if not isinstance(tested, dict) or int(tested.get("R", -1)) != int(deadline):
            continue
        candidate = tested.get("maximizing_case")
        if isinstance(candidate, dict) and isinstance(candidate.get("carry_in_model"), dict):
            return int(candidate["carry_in_model"].get("carry_in", 0))
    return None


def _stable_unit(*parts: object) -> float:
    raw = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") / float(2**64 - 1)


def _normal_interval(task: Any, bounds: dict[str, tuple[int, int]]) -> tuple[int, int]:
    lo, hi = bounds[str(task.name)]
    if str(task.criticality.value) == "HI":
        hi = min(hi, int(task.c_lo))
    return int(lo), int(max(lo, hi))


def _level(lo: int, hi: int, mode: str, u: float) -> int:
    if mode == "min":
        return lo
    if mode == "max":
        return hi
    if mode == "mid":
        return (lo + hi) // 2
    if u < 0.20:
        return lo
    if u < 0.45:
        return (lo + hi) // 2
    return hi


def _profile_demand(task: Any, release_index: int, bounds: dict[str, tuple[int, int]], profile: str, profile_seed: int) -> int:
    lo, hi = _normal_interval(task, bounds)
    crit = str(task.criticality.value)
    if profile == "pressure":
        mode = "max"
    elif profile == "light":
        mode = "min"
    elif profile == "mid":
        mode = "mid"
    elif profile == "lo_pressure":
        mode = "max" if crit == "LO" else "mid"
    elif profile == "hi_pressure":
        mode = "max" if crit == "HI" else "mid"
    elif profile.startswith("random"):
        mode = "random"
    else:
        raise ValueError(f"unknown profile: {profile}")
    return _level(lo, hi, mode, _stable_unit(profile_seed, profile, task.name, release_index))


def _background_scenario(target: Any, bounds: dict[str, tuple[int, int]], profile: str, profile_seed: int) -> Any:
    from amc_py.runtime_scenarios import ExecutionScenario

    def resolver(task: Any, release_index: int) -> int:
        return _profile_demand(task, release_index, bounds, profile, profile_seed)

    return ExecutionScenario(name=f"s603_profile_{profile}_{profile_seed}", resolver=resolver)


def _trigger_scenario(
    target: Any,
    target_name: str,
    bounds: dict[str, tuple[int, int]],
    profile: str,
    profile_seed: int,
    trigger_time: int,
) -> Any:
    from amc_py.runtime_scenarios import ExecutionScenario

    tasks = list(target.ordered_tasks)
    names = [str(task.name) for task in tasks]
    target_index = names.index(target_name)

    def resolver(task: Any, release_index: int) -> int:
        name = str(task.name)
        release_time = int(release_index) * int(task.period)
        if release_time < trigger_time:
            return _profile_demand(task, release_index, bounds, profile, profile_seed)
        lo, hi = bounds[name]
        return int(hi if names.index(name) <= target_index else lo)

    return ExecutionScenario(name=f"s603_trigger_{profile}_{profile_seed}_{trigger_time}", resolver=resolver)


def _step(env: Any, obs: Any, policy: Any) -> Any:
    mask = tuple(bool(value) for value in env.formal_valid_action_mask())
    action_id, _trace = policy.select_action_id(tuple(obs.state_vector), mask, include_decision_trace=True)
    return env.step(action_id).observation


def _remaining_carry_jobs(env: Any, target_priority: int, trigger_time: int) -> tuple[list[dict[str, Any]], int]:
    priority_by_name = {str(task.name): index for index, task in enumerate(env.ordered_tasks)}
    rows: list[dict[str, Any]] = []
    total = 0
    for job in env._engine.state.active_jobs:
        if priority_by_name[str(job.task.name)] >= target_priority or int(job.release_time) >= trigger_time:
            continue
        if job.dropped or job.completion_time is not None:
            continue
        raw = int(job.original_actual_cost if job.original_actual_cost is not None else job.actual_cost)
        if str(job.task.criticality.value) == "LO":
            budget = int(job.runtime_budget_at_release if job.runtime_budget_at_release is not None else job.task.c_lo)
            cap = min(raw, budget + 1)
        else:
            cap = int(job.actual_cost)
        remaining = max(0, cap - int(job.executed_time))
        if remaining:
            rows.append({
                "task": str(job.task.name),
                "release_index": int(job.release_index),
                "release_time": int(job.release_time),
                "actual_cost": int(job.actual_cost),
                "runtime_budget_at_release": job.runtime_budget_at_release,
                "executed_time": int(job.executed_time),
                "effective_remaining": int(remaining),
            })
            total += remaining
    return rows, int(total)


def _scan_profile(
    target: Any,
    policy: Any,
    target_name: str,
    bounds: dict[str, tuple[int, int]],
    profile: str,
    profile_seed: int,
    trigger_indices: list[int],
) -> list[dict[str, Any]]:
    env = copy.deepcopy(target.runtime_adapter.environment)
    env.scenario = _background_scenario(target, bounds, profile, profile_seed)
    obs = env.reset()
    target_task = next(task for task in env.ordered_tasks if task.name == target_name)
    target_priority = list(env.ordered_tasks).index(target_task)
    trigger_by_time = {int(index) * int(target_task.period): int(index) for index in trigger_indices}
    last_time = max(trigger_by_time)
    rows: list[dict[str, Any]] = []

    while int(env._engine.current_time) < last_time and not env._done:
        obs = _step(env, obs, policy)
        now = int(env._engine.current_time)
        if now not in trigger_by_time:
            continue
        result = env._engine.finish()
        prior_switches = [event for event in result.mode_switches if int(event.switch_time) < now]
        prior_hi_misses = [miss for miss in result.deadline_misses if int(miss.absolute_deadline) < now and str(miss.task).startswith("mc_sd_hi_")]
        carry_jobs, carry = _remaining_carry_jobs(env, target_priority, now)
        rows.append({
            "profile": profile,
            "profile_seed": int(profile_seed),
            "trigger_index": trigger_by_time[now],
            "trigger_time": now,
            "valid_lo_prefix": not prior_switches and not prior_hi_misses,
            "concrete_hp_carry": int(carry),
            "carry_jobs": carry_jobs,
            "runtime_budgets": dict(env._engine.runtime_budgets.budgets),
            "prior_switch_count": len(prior_switches),
            "prior_hi_miss_count": len(prior_hi_misses),
        })
    return rows


def _replay_trigger(
    target: Any,
    policy: Any,
    target_name: str,
    bounds: dict[str, tuple[int, int]],
    profile: str,
    profile_seed: int,
    trigger_index: int,
) -> dict[str, Any]:
    env = copy.deepcopy(target.runtime_adapter.environment)
    target_task = next(task for task in env.ordered_tasks if task.name == target_name)
    trigger_time = int(trigger_index) * int(target_task.period)
    stop_time = trigger_time + int(target_task.deadline)
    env.scenario = _trigger_scenario(target, target_name, bounds, profile, profile_seed, trigger_time)
    obs = env.reset()
    while int(env._engine.current_time) < stop_time and not env._done:
        obs = _step(env, obs, policy)
    result = env._engine.finish()
    first_switch = min((int(event.switch_time) for event in result.mode_switches), default=None)
    prior_hi_misses = [miss for miss in result.deadline_misses if int(miss.absolute_deadline) < trigger_time and str(miss.task).startswith("mc_sd_hi_")]
    jobs = [job for job in result.jobs if job.task.name == target_name and int(job.release_index) == trigger_index]
    if len(jobs) != 1:
        outcome = {"found": False}
    else:
        job = jobs[0]
        misses = [miss for miss in result.deadline_misses if str(miss.task) == target_name and int(miss.release_index) == trigger_index]
        miss = bool(misses)
        executed = int(misses[0].executed_at_miss) if miss else int(job.actual_cost)
        outcome = {
            "found": True,
            "actual_cost": int(job.actual_cost),
            "deadline": int(job.absolute_deadline),
            "executed_at_deadline": executed,
            "remaining_at_deadline": max(0, int(job.actual_cost) - executed),
            "completion_time": None if job.completion_time is None else int(job.completion_time),
            "miss": miss,
        }
    valid = not prior_hi_misses and first_switch == trigger_time
    return {
        "profile": profile,
        "profile_seed": int(profile_seed),
        "trigger_index": int(trigger_index),
        "trigger_time": int(trigger_time),
        "valid_conditioned_prefix": bool(valid),
        "first_switch_time": first_switch,
        "first_switch_at_trigger": first_switch == trigger_time,
        "prior_hi_miss_count": len(prior_hi_misses),
        "target_outcome": outcome,
        "concrete_counterexample": bool(valid and outcome.get("miss")),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# s603 conditioned reachable-carry search",
        "",
        f"Signal: **{report['diagnostic_signal']}**",
        "",
        f"Proof carry bound: `{report['proof_carry_bound']}`",
        f"Best concrete carry in conditioned LO prefixes: `{report['max_conditioned_concrete_carry']}`",
        "",
        "| carry | profile | seed | q | replay miss |",
        "|---:|---|---:|---:|:---:|",
    ]
    replay_by_key = {(row["profile"], row["profile_seed"], row["trigger_index"]): row for row in report["replays"]}
    for row in report["top_states"]:
        replay = replay_by_key.get((row["profile"], row["profile_seed"], row["trigger_index"]), {})
        lines.append(
            f"| {row['concrete_hp_carry']} | `{row['profile']}` | {row['profile_seed']} | "
            f"{row['q']} | `{(replay.get('target_outcome') or {}).get('miss')}` |"
        )
    lines += ["", "Bounded concrete search only; the observed maximum is not a formal upper bound."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--seed-workspace", default="s603_best_overall_v9_1_e2e")
    parser.add_argument("--target", default="mc_sd_hi_0")
    parser.add_argument("--q-start", type=int, default=1)
    parser.add_argument("--q-count", type=int, default=6)
    parser.add_argument("--profiles", default="pressure,mid,lo_pressure,hi_pressure,light")
    parser.add_argument("--random-profiles", type=int, default=4)
    parser.add_argument("--search-seed", type=int, default=603013)
    parser.add_argument("--replay-top", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/s603_conditioned_carry"))
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    wi = WorkspaceInput(args.formal, args.seed_workspace)
    try:
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        target = _load_target(source_root, wi.workspace)
        policy = _load_integer_policy(wi.workspace)
        bounds = _demand_bounds(target)
        task = next(row for row in target.ordered_tasks if row.name == args.target)
        step = int(target.runtime_config.agent_period) // gcd(int(task.period), int(target.runtime_config.agent_period))
        trigger_indices = [(args.q_start + offset) * step for offset in range(args.q_count)]
        proof_carry = _proof_carry_bound(wi.workspace, args.target, int(task.deadline))

        profile_specs: list[tuple[str, int]] = []
        for index, name in enumerate(item.strip() for item in args.profiles.split(",") if item.strip()):
            profile_specs.append((name, args.search_seed + index))
        for index in range(args.random_profiles):
            profile_specs.append((f"random{index}", args.search_seed + 1000 + index))

        states: list[dict[str, Any]] = []
        for profile, profile_seed in profile_specs:
            for row in _scan_profile(target, policy, args.target, bounds, profile, profile_seed, trigger_indices):
                if row.get("valid_lo_prefix"):
                    row["q"] = int(row["trigger_index"]) // step
                    states.append(row)
        states.sort(key=lambda row: int(row["concrete_hp_carry"]), reverse=True)
        top = states[: max(1, args.replay_top)]

        replays = [
            _replay_trigger(
                target,
                policy,
                args.target,
                bounds,
                row["profile"],
                int(row["profile_seed"]),
                int(row["trigger_index"]),
            )
            for row in top
        ]
        counterexamples = [row for row in replays if row.get("concrete_counterexample")]
        max_carry = int(top[0]["concrete_hp_carry"]) if top else None
        if counterexamples:
            signal = "CONCRETE_REACHABLE_COUNTEREXAMPLE_FOUND"
        elif max_carry is None:
            signal = "NO_VALID_CONDITIONED_PREFIX_FOUND"
        elif proof_carry is not None and max_carry < proof_carry:
            signal = "BOUNDED_CONDITIONED_SEARCH_BELOW_PROOF_CARRY"
        else:
            signal = "BOUNDED_SEARCH_REACHES_PROOF_CARRY_SCALE"

        report = {
            "schema_version": "s603_conditioned_reachable_carry_search_v2",
            "diagnostic_only": True,
            "seed_workspace": args.seed_workspace,
            "target": args.target,
            "condition": "theta=0 AND immediate LO_SWITCH[0,0] after a no-switch/no-HI-miss LO prefix",
            "proof_carry_bound": proof_carry,
            "max_conditioned_concrete_carry": max_carry,
            "profile_count": len(profile_specs),
            "valid_state_count": len(states),
            "profile_specs": [{"profile": p, "seed": s} for p, s in profile_specs],
            "top_states": top,
            "replays": replays,
            "counterexample_count": len(counterexamples),
            "diagnostic_signal": signal,
            "warning": "bounded concrete search; max_conditioned_concrete_carry is not a proof upper bound",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "s603_conditioned_carry_search.json"
        md_path = args.output_dir / "s603_conditioned_carry_search.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_markdown(report), encoding="utf-8")
        print(json.dumps({
            "signal": signal,
            "max_conditioned_carry": max_carry,
            "proof_carry": proof_carry,
            "counterexamples": len(counterexamples),
            "valid_states": len(states),
        }, ensure_ascii=False))
        return 0
    finally:
        wi.close()


if __name__ == "__main__":
    raise SystemExit(main())
