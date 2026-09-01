#!/usr/bin/env python3
"""Diagnostic-only reachable-prehistory validation for s603 / mc_sd_hi_0.

This tool does not modify the V10.12 proof DAG.  It replays the deployed integer
VIPER policy with the production C-AMC-sem environment and searches target
releases compatible with theta=0.  Before the chosen trigger release all HI
jobs are kept NORMAL (A<=C_LO), while higher-priority LO jobs use their formal
upper demand bounds.  At the trigger, the target is made ABNORMAL and all
higher-priority future jobs use their formal upper demand bounds.

A run is reported as a concrete counterexample only if:
  * no HI deadline miss occurred before the trigger;
  * no LO->HI switch occurred before the trigger;
  * the first LO->HI switch occurs exactly at the trigger release; and
  * the selected target job misses its deadline.

Failure to find a counterexample is diagnostic only and is NOT a proof of safety.
"""
from __future__ import annotations

import argparse
import copy
import json
from math import gcd
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _find_workspace(root: Path, seed: str) -> Path:
    direct = root / seed
    if (direct / "request/proof_request.json").is_file():
        return direct
    matches = [p.parent.parent for p in root.rglob("request/proof_request.json") if seed in str(p.parent.parent)]
    matches = sorted({p.resolve() for p in matches})
    if len(matches) != 1:
        raise ValueError(f"expected one workspace matching {seed!r}, found {len(matches)}")
    return matches[0]


class WorkspaceInput:
    def __init__(self, path: Path, seed: str) -> None:
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        if path.is_dir():
            root = path.resolve()
        elif path.is_file() and path.suffix.lower() == ".zip":
            self._tmp = tempfile.TemporaryDirectory(prefix="s603_prehistory_")
            root = Path(self._tmp.name)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(root)
        else:
            raise ValueError("--formal must be a proof-result directory or ZIP")
        self.workspace = _find_workspace(root, seed)

    def close(self) -> None:
        if self._tmp:
            self._tmp.cleanup()


def _load_formal_target(source_root: Path, workspace: Path) -> Any:
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


def _formal_demand_bounds(target: Any) -> dict[str, tuple[int, int]]:
    raw = target.provenance.get("actual_demand_bounds_by_task")
    if not isinstance(raw, dict):
        raise ValueError("target provenance lacks actual_demand_bounds_by_task")
    return {str(k): (int(v["min"]), int(v["max"])) for k, v in raw.items()}


def _proof_deadline_case(workspace: Path, target_name: str, deadline: int) -> dict[str, Any]:
    receipts = _read_json(workspace / "verified/proof_receipts.json")
    rows = [r for r in receipts.get("pcssc_targets", []) if isinstance(r, dict) and r.get("target") == target_name]
    if len(rows) != 1:
        return {}
    tested = rows[0].get("tested_horizons", [])
    exact = [r for r in tested if isinstance(r, dict) and r.get("terminal") == "POINTWISE" and int(r.get("R", -1)) == deadline]
    return dict(exact[-1].get("maximizing_case", {})) if exact else {}


def _build_scenario(target: Any, target_name: str, trigger_index: int, bounds: dict[str, tuple[int, int]]):
    from amc_py.runtime_scenarios import ExecutionScenario
    tasks = list(target.ordered_tasks)
    names = [t.name for t in tasks]
    ti = names.index(target_name)
    trigger_time = int(trigger_index) * int(tasks[ti].period)

    def resolver(task: Any, release_index: int) -> int:
        name = str(task.name)
        lo, hi = bounds[name]
        release_time = int(release_index) * int(task.period)
        priority = names.index(name)
        if priority > ti:
            return lo
        if name == target_name:
            if int(release_index) == int(trigger_index):
                if hi <= int(task.c_lo):
                    raise ValueError("target upper demand cannot create ABNORMAL release")
                return hi
            return lo if release_time < trigger_time else hi
        if release_time < trigger_time:
            if str(task.criticality.value) == "HI":
                return min(hi, int(task.c_lo))
            return hi
        return hi

    return ExecutionScenario(name=f"s603_reachable_prehistory_trigger_{trigger_index}", resolver=resolver)


def _mode_value(mode: Any) -> str:
    return str(getattr(mode, "value", mode))


def _remaining_carry_jobs(env: Any, target_priority: int, trigger_time: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total = 0
    priority_by_name = {str(task.name): index for index, task in enumerate(env.ordered_tasks)}
    for job in env._engine.state.active_jobs:
        if int(priority_by_name[str(job.task.name)]) >= int(target_priority) or int(job.release_time) >= int(trigger_time):
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
        if remaining <= 0:
            continue
        rows.append({
            "task": job.task.name,
            "release_index": int(job.release_index),
            "release_time": int(job.release_time),
            "actual_cost": int(job.actual_cost),
            "runtime_budget_at_release": job.runtime_budget_at_release,
            "executed_time": int(job.executed_time),
            "effective_remaining": int(remaining),
        })
        total += remaining
    return rows, int(total)


def _run_trigger(target: Any, policy: Any, target_name: str, trigger_index: int, bounds: dict[str, tuple[int, int]]) -> dict[str, Any]:
    env = copy.deepcopy(target.runtime_adapter.environment)
    tasks = list(env.ordered_tasks)
    target_task = next(t for t in tasks if t.name == target_name)
    trigger_time = int(trigger_index) * int(target_task.period)
    stop_time = trigger_time + int(target_task.deadline)
    if env.runtime_config.end_time is not None and stop_time > int(env.runtime_config.end_time):
        return {"trigger_index": trigger_index, "trigger_time": trigger_time, "status": "SKIPPED_BEYOND_RUNTIME_END"}
    env.scenario = _build_scenario(target, target_name, trigger_index, bounds)
    obs = env.reset()
    trigger_snapshot: dict[str, Any] | None = None
    action_rows: list[dict[str, Any]] = []
    while int(env._engine.current_time) < stop_time and not env._done:
        now = int(env._engine.current_time)
        mask = tuple(bool(x) for x in env.formal_valid_action_mask())
        action_id, trace = policy.select_action_id(tuple(obs.state_vector), mask, include_decision_trace=True)
        if abs(now - trigger_time) <= int(env.agent_period):
            action_rows.append({"time": now, "action_id": action_id, "leaf": trace.get("tree_leaf_id"), "rank": trace.get("tree_selected_rank")})
        step = env.step(action_id)
        obs = step.observation
        if trigger_snapshot is None and int(env._engine.current_time) >= trigger_time:
            jobs, carry = _remaining_carry_jobs(env, tasks.index(target_task), trigger_time)
            result = env._engine.finish()
            switches_before = [e for e in result.mode_switches if int(e.switch_time) < trigger_time]
            switches_at = [e for e in result.mode_switches if int(e.switch_time) == trigger_time]
            hi_miss_before = [m for m in result.deadline_misses if int(m.absolute_deadline) < trigger_time and str(m.task).startswith("mc_sd_hi_")]
            trigger_snapshot = {
                "time": int(env._engine.current_time),
                "mode_after_boundary": _mode_value(env._engine.state.mode),
                "carry_jobs": jobs,
                "concrete_hp_carry": int(carry),
                "switches_before_trigger": len(switches_before),
                "switches_at_trigger": len(switches_at),
                "hi_misses_before_trigger": len(hi_miss_before),
                "runtime_budgets": dict(env._engine.runtime_budgets.budgets),
            }
    result = env._engine.finish()
    matches = [j for j in result.jobs if j.task.name == target_name and int(j.release_index) == int(trigger_index)]
    if len(matches) != 1:
        target_outcome = {"found": False}
    else:
        job = matches[0]
        deadline = int(job.absolute_deadline)
        miss = job.completion_time is None or int(job.completion_time) > deadline
        target_outcome = {
            "found": True,
            "release_time": int(job.release_time),
            "absolute_deadline": deadline,
            "actual_cost": int(job.actual_cost),
            "executed_time": int(job.executed_time),
            "completion_time": job.completion_time,
            "miss": bool(miss),
            "remaining_at_end": max(0, int(job.actual_cost) - int(job.executed_time)),
        }
    first_switch_time = min((int(e.switch_time) for e in result.mode_switches), default=None)
    pre = trigger_snapshot or {}
    valid_prefix = bool(pre) and int(pre.get("switches_before_trigger", 1)) == 0 and int(pre.get("hi_misses_before_trigger", 1)) == 0
    first_switch_at_trigger = first_switch_time == trigger_time
    concrete_counterexample = valid_prefix and first_switch_at_trigger and bool(target_outcome.get("miss"))
    return {
        "trigger_index": int(trigger_index),
        "trigger_time": int(trigger_time),
        "theta": 0,
        "status": "DONE",
        "valid_reachable_prefix": bool(valid_prefix),
        "first_switch_time": first_switch_time,
        "first_switch_at_trigger": bool(first_switch_at_trigger),
        "trigger_snapshot": trigger_snapshot,
        "target_outcome": target_outcome,
        "concrete_counterexample": bool(concrete_counterexample),
        "actions_near_trigger": action_rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# s603 reachable-prehistory diagnostic", "", f"Verdict: **{report['verdict']}**", ""]
    lines += [f"Proof carry bound at deadline: `{report.get('proof_carry_bound')}`", "", "| q | trigger | carry | valid prefix | switch@trigger | target miss |", "|---:|---:|---:|:---:|:---:|:---:|"]
    for r in report["runs"]:
        snap = r.get("trigger_snapshot") or {}
        out = r.get("target_outcome") or {}
        lines.append(f"| {r['q']} | {r.get('trigger_time')} | {snap.get('concrete_hp_carry')} | {r.get('valid_reachable_prefix')} | {r.get('first_switch_at_trigger')} | {out.get('miss')} |")
    lines += ["", "Interpretation: failure to find a counterexample in this bounded scan is not a safety proof."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", required=True, type=Path, help="formal.zip or extracted result root")
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--seed-workspace", default="s603_best_overall_v9_1_e2e")
    ap.add_argument("--target", default="mc_sd_hi_0")
    ap.add_argument("--q-start", type=int, default=1)
    ap.add_argument("--q-count", type=int, default=6)
    ap.add_argument("--output-dir", type=Path, default=Path("diagnostic_s603_reachable_prehistory"))
    args = ap.parse_args()
    wi = WorkspaceInput(args.formal, args.seed_workspace)
    try:
        target = _load_formal_target(args.source_root.resolve(), wi.workspace)
        policy = _load_integer_policy(wi.workspace)
        bounds = _formal_demand_bounds(target)
        task = next(t for t in target.ordered_tasks if t.name == args.target)
        agent_period = int(target.runtime_config.agent_period)
        step = agent_period // gcd(int(task.period), agent_period)
        if step <= 0:
            raise ValueError("invalid theta=0 release-index step")
        proof_case = _proof_deadline_case(wi.workspace, args.target, int(task.deadline))
        proof_carry = None
        if isinstance(proof_case.get("carry_in_model"), dict):
            proof_carry = proof_case["carry_in_model"].get("carry_in")
        runs = []
        for q in range(args.q_start, args.q_start + args.q_count):
            trigger_index = q * step
            row = _run_trigger(target, policy, args.target, trigger_index, bounds)
            row["q"] = q
            runs.append(row)
        counterexamples = [r for r in runs if r.get("concrete_counterexample")]
        valid = [r for r in runs if r.get("valid_reachable_prefix")]
        max_carry = max((int((r.get("trigger_snapshot") or {}).get("concrete_hp_carry", 0)) for r in valid), default=None)
        if counterexamples:
            verdict = "CONCRETE_REACHABLE_PREHISTORY_COUNTEREXAMPLE_FOUND"
        elif valid:
            verdict = "NO_COUNTEREXAMPLE_FOUND_IN_SCANNED_REACHABLE_PREHISTORIES"
        else:
            verdict = "NO_VALID_PREHISTORY_CANDIDATE_IN_SCAN"
        report = {
            "schema_version": "s603_reachable_prehistory_diagnostic_v1",
            "diagnostic_only": True,
            "seed_workspace": args.seed_workspace,
            "target": args.target,
            "theta": 0,
            "theta_zero_release_index_step": step,
            "proof_carry_bound": None if proof_carry is None else int(proof_carry),
            "max_concrete_carry_in_valid_scan": max_carry,
            "verdict": verdict,
            "counterexample_count": len(counterexamples),
            "runs": runs,
            "nonproof_warning": "bounded concrete search; absence of a counterexample is not a proof",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "s603_reachable_prehistory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output_dir / "s603_reachable_prehistory.md").write_text(_markdown(report), encoding="utf-8")
        print(json.dumps({"verdict": verdict, "counterexamples": len(counterexamples), "max_concrete_carry": max_carry}, ensure_ascii=False))
        return 0
    finally:
        wi.close()


if __name__ == "__main__":
    raise SystemExit(main())
