"""复合 runtime handler 的有限 micro-step 分解检查。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def _function(tree: ast.Module, qualified: str) -> ast.FunctionDef | None:
    if "." in qualified:
        cls, name = qualified.split(".", 1)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                found = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == name]
                return found[0] if len(found) == 1 else None
    found = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == qualified]
    return found[0] if len(found) == 1 else None


def _calls(function: ast.FunctionDef) -> list[str]:
    nodes = sorted((node for node in ast.walk(function) if isinstance(node, ast.Call)),
                   key=lambda node: (int(getattr(node, "lineno", 0)),
                                     int(getattr(node, "col_offset", 0))))
    return [ast.unparse(node.func) for node in nodes]


def _ordered_subsequence(calls: list[str], required: tuple[str, ...]) -> bool:
    required = tuple(item for item in required
                     if not item.startswith("EventType.")
                     and item != "budget_update_events")
    cursor = 0
    for call in calls:
        if cursor < len(required) and required[cursor] in call:
            cursor += 1
    return cursor == len(required)


def build_handler_decomposition_certificate(
        source_root: str | Path, *, context_hash: str,
        transition_case_certificates: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    path = Path(source_root) / "amc_py/event_runtime.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    requirements = {
        "boot": ("EventRuntimeEngine.build", ("queue.push",)),
        "arrival_batch": ("EventRuntimeEngine._process_job_arrival_batch",
                          ("pop_all_matching", "events.sort", "_maybe_enter_c_amc_sem_hi_mode_at_arrival",
                           "_process_single_arrival_in_priority_order", "_reschedule")),
        "event_handler": ("EventRuntimeEngine._process_event",
                           ("_advance_time", "EventType.BUDGET_UPDATE", "EventType.JOB_ARRIVAL",
                            "EventType.DEADLINE_CHECK", "EventType.JOB_COMPLETION", "_reschedule")),
        "controller_engine": ("EventRuntimeEngine.apply_budget_updates",
                               ("_advance_time", "apply_updates", "budget_update_events", "_reschedule")),
    }
    rows = []
    failures = []
    for name, (qualified, required) in requirements.items():
        function = _function(tree, qualified)
        calls = _calls(function) if function else []
        text = ast.unparse(function) if function else ""
        missing = [token for token in required if token not in text]
        if name == "boot" and "Event" not in text:
            missing.append("Event")
        ordered = _ordered_subsequence(calls, required)
        if not ordered:
            missing.append("CALL_ORDER:" + "→".join(required))
        if function is None or missing:
            failures.append({"component": name, "missing": missing or ["FUNCTION"]})
        rows.append({"component": name, "function": qualified,
                     "line_start": getattr(function, "lineno", None),
                     "line_end": getattr(function, "end_lineno", None),
                     "calls": calls, "missing": missing,
                     "call_order_verified": ordered,
                     "source_hash": sha256_object(text)})
    initializer = _function(tree, "EventRuntimeEngine.__post_init__")
    initializer_text = ast.unparse(initializer) if initializer else ""
    if initializer is None or "EventType.JOB_ARRIVAL" not in initializer_text:
        failures.append({"component": "boot_initializer", "missing": ["EventType.JOB_ARRIVAL"]})
    rows.append({"component": "boot_initializer", "function": "EventRuntimeEngine.__post_init__",
                 "line_start": getattr(initializer, "lineno", None),
                 "line_end": getattr(initializer, "end_lineno", None),
                 "calls": _calls(initializer) if initializer else [],
                 "missing": [] if initializer and "EventType.JOB_ARRIVAL" in initializer_text else ["EventType.JOB_ARRIVAL"],
                 "source_hash": sha256_object(initializer_text)})
    wrapper = Path(source_root) / "amc_py/rl/runtime_wrapper.py"
    wrapper_text = wrapper.read_text(encoding="utf-8")
    wrapper_bound = "engine.apply_budget_updates(updates)" in wrapper_text
    if not wrapper_bound:
        failures.append({"component": "controller_wrapper", "missing": ["engine.apply_budget_updates(updates)"]})
    acceptance_source = (Path(source_root) / "scripts/run_phase_ijk_acceptance.py").read_text(encoding="utf-8")
    boot_preclosed_bound = "engine.run_until(0, include_boundary=True)" in acceptance_source
    if not boot_preclosed_bound:
        failures.append({"component": "boot_preclosed0",
                         "missing": ["engine.run_until(0, include_boundary=True)"]})
    hashes_path = Path(__file__).resolve().parents[1] / "theory" / "hashes.json"
    statements = json.loads(hashes_path.read_text(encoding="utf-8")).get("statements", {})
    theorem_ids = ("ARRIVAL_BATCH_LOOP_DECOMPOSITION", "EVENT_HANDLER_MICROSTEP_DECOMPOSITION")
    missing_theorems = [item for item in theorem_ids if item not in statements]
    failures.extend({"component": "theory", "missing": [item]} for item in missing_theorems)
    # 复合 handler 只有在其 child transition proof 已经闭合时才可以 PASS。
    # 这里消费 proof hash，而不是把“调用存在”当作组合证明。
    proofs = {str(item.get("case_id", item.get("inputs", {}).get("case_id"))): item
              for item in (transition_case_certificates or [])}
    composition_cases = {
        "boot": ("BOOT_TO_PRECLOSED_0",),
        "arrival_batch": ("ARRIVAL_BATCH_NO_SWITCH", "ARRIVAL_BATCH_SWITCH_S0",
                          "PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE"),
        "event_handler": ("ONE_SERVICE_TICK", "NORMAL_COMPLETION", "DEGRADED_COMPLETION",
                           "HI_COMPLETION", "PRIMARY_LO_CANCELLATION",
                           "DEADLINE_OBSERVATION_NO_MISS", "DEADLINE_OBSERVATION_FIRST_HI_MISS"),
        "controller_engine": ("CONTROLLER_NO_ACTION", "CONTROLLER_SELECTED_ACTION"),
    }
    compositions = {}
    for component, case_ids in composition_cases.items():
        missing = [case_id for case_id in case_ids if case_id not in proofs]
        invalid = [case_id for case_id in case_ids
                   if case_id in proofs and proofs[case_id].get(
                       "z3_proof_result", proofs[case_id].get("witness", {}).get("z3_proof_result")) != "PASS"]
        steps = []
        for case_id in case_ids:
            proof = proofs.get(case_id, {})
            steps.append({"case_id": case_id,
                          "concrete_delta_hash": proof.get("concrete_delta_hash",
                              proof.get("witness", {}).get("concrete_delta_hash")),
                          "reference_delta_hash": proof.get("projected_reference_delta_hash",
                              proof.get("witness", {}).get("projected_reference_delta_hash")),
                          "relation_preservation": proof.get("relation_preservation",
                              proof.get("witness", {}).get("relation_preservation"))})
        compositions[component] = {
            "ordered_case_ids": list(case_ids), "steps": steps,
            "missing_transition_proofs": missing, "invalid_transition_proofs": invalid,
            "post_state_equivalence": (
                "HANDLER_POST_STATE = SEQUENTIAL_MICRO_STEP_POST_STATE"),
            "composition_formula_hash": sha256_object({"component": component, "steps": steps}),
            "proof_status": "PASS" if not missing and not invalid else "UNRESOLVED",
        }
    if transition_case_certificates is None:
        failures.append({"component": "transition_proofs", "missing": ["ALL_MICRO_STEP_PROOFS"]})
    elif any(item["proof_status"] != "PASS" for item in compositions.values()):
        failures.append({"component": "micro_step_composition", "missing": [
            name for name, item in compositions.items() if item["proof_status"] != "PASS"]})
    result = {"status": "PASS" if not failures else "UNRESOLVED",
              "schema_version": "handler_decomposition_v2_composed_transitions", "context_hash": context_hash,
              "components": rows, "controller_wrapper_bound": wrapper_bound,
              "boot_preclosed0_bound": boot_preclosed_bound,
              "theorem_refs": {item: statements[item] for item in theorem_ids if item in statements},
              "phase_dag": {"batch_pop": ["batch_sort"], "batch_sort": ["mode_switch"],
                             "mode_switch": ["release_loop"], "release_loop": ["dispatch"],
                             "dispatch": ["child_events"], "child_events": []},
              "failures": failures, "compositions": compositions,
              "transition_proof_hashes": {
                  case_id: {"concrete_delta_hash": proof.get("concrete_delta_hash"),
                            "projected_reference_delta_hash": proof.get("projected_reference_delta_hash"),
                            "proof_result": proof.get("z3_proof_result")}
                  for case_id, proof in proofs.items()}}
    result["artifact_hash"] = sha256_object(result)
    return result
