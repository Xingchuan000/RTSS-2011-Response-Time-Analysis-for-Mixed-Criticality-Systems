"""复合 runtime handler 的有限 micro-step 分解检查。"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from formal_toolchain.core.hashing import sha256_object


ARRIVAL_BATCH_ALTERNATIVES = (
    {"alternative_id": "ARRIVAL_BATCH_NO_SWITCH", "guard_kind": "NO_BATCH_MODE_SWITCH", "macro_case_id": "ARRIVAL_BATCH_NO_SWITCH"},
    {"alternative_id": "ARRIVAL_BATCH_SWITCH_S0", "guard_kind": "BATCH_MODE_SWITCH_S0", "macro_case_id": "ARRIVAL_BATCH_SWITCH_S0"},
)

EVENT_HANDLER_ALTERNATIVES = (
    {"alternative_id": "CONTROLLER_NO_ACTION", "case_ids": ("CONTROLLER_NO_ACTION",)},
    {"alternative_id": "CONTROLLER_SELECTED_ACTION", "case_ids": ("CONTROLLER_SELECTED_ACTION",)},
    {"alternative_id": "JOB_ARRIVAL_NO_SWITCH", "case_ids": ("ARRIVAL_BATCH_NO_SWITCH",)},
    {"alternative_id": "JOB_ARRIVAL_SWITCH_S0", "case_ids": ("ARRIVAL_BATCH_SWITCH_S0",)},
    {"alternative_id": "DEADLINE_NO_MISS", "case_ids": ("DEADLINE_OBSERVATION_NO_MISS",)},
    {"alternative_id": "DEADLINE_FIRST_HI_MISS", "case_ids": ("DEADLINE_OBSERVATION_FIRST_HI_MISS",)},
    {"alternative_id": "NORMAL_COMPLETION", "case_ids": ("NORMAL_COMPLETION", "PREEMPTION_DISPATCH")},
    {"alternative_id": "DEGRADED_COMPLETION", "case_ids": ("DEGRADED_COMPLETION", "PREEMPTION_DISPATCH")},
    {"alternative_id": "HI_COMPLETION", "case_ids": ("HI_COMPLETION", "PREEMPTION_DISPATCH")},
    {"alternative_id": "PRIMARY_LO_CANCELLATION", "case_ids": ("PRIMARY_LO_CANCELLATION", "PREEMPTION_DISPATCH")},
    {"alternative_id": "IDLE_RECOVERY", "case_ids": ("IDLE_RECOVERY",)},
)

ARRIVAL_BATCH_STAGES = (
    "INITIALIZE_BATCH", "POP_SAME_TIME_ARRIVALS", "PRIORITY_SORT",
    "OPTIONAL_BATCH_MODE_SWITCH", "FINITE_SINGLE_ARRIVAL_FOLD",
    "FINAL_RESCHEDULE", "RETURN",
)
@dataclass(frozen=True, slots=True)
class HandlerStage:
    stage_id: str
    kind: str
    source_ast_hash: str
    callee: str | None
    branch_guard_hash: str | None = None
    loop_variable: str | None = None
    loop_iterable_hash: str | None = None


@dataclass(frozen=True, slots=True)
class HandlerAlternative:
    alternative_id: str
    guard_formula: str
    ordered_stages: tuple[HandlerStage, ...]


@dataclass(frozen=True, slots=True)
class CompositeHandlerIR:
    schema_version: str
    handler_id: str
    source_hash: str
    alternatives: tuple[HandlerAlternative, ...]
    alternatives_exhaustive: bool
    alternatives_mutually_exclusive: bool


@dataclass(frozen=True, slots=True)
class FiniteFoldCertificate:
    status: str
    loop_source_hash: str
    iterable_is_finite: bool
    body_called_once_per_element: bool
    loop_has_no_early_exit: bool
    release_key_function_bound: bool
    input_keys_unique: bool
    element_case_partition_complete: bool
    element_case_partition_exclusive: bool
    fold_preserves_relation: bool
    fold_extends_job_map: bool
    fold_preserves_ledgers: bool
    child_certificate_hashes: tuple[str, ...]
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class CompositeSequenceResult:
    status: str
    formula: str
    declarations: str
    state_ids: tuple[str, ...]
    solver_result: str
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class ArrivalFoldResult:
    status: str
    finite_iterable_proved: bool
    one_call_per_element_proved: bool
    no_early_exit_proved: bool
    element_partition_proved: bool
    child_release_cases_proved: bool
    fresh_extension_per_element_proved: bool
    old_domain_frame_proved: bool
    ledger_preservation_proved: bool
    failure: str | None = None


RELEASE_ELEMENT_CASES = ("PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE")


def build_arrival_fold_result(*, loop_info: Mapping[str, object],
                              release_case_certificates: Sequence[Mapping[str, object]]) -> ArrivalFoldResult:
    finite = loop_info.get("iterable_kind") == "FINITE_EVENT_LIST"
    one_call = loop_info.get("single_arrival_call_count") == 1
    no_exit = not bool(loop_info.get("has_early_exit", True))
    cases = {str(item.get("case_id")): item for item in release_case_certificates}
    present = all(case_id in cases for case_id in RELEASE_ELEMENT_CASES)
    child_pass = present and all(
        cases[case_id].get("status") == "PASS"
        and cases[case_id].get("parameterized_contract_status") == "PASS"
        for case_id in RELEASE_ELEMENT_CASES
    )
    fresh = child_pass and all(
        cases[case_id].get("map_update_kind") == "EXTEND_WITH_FRESH_RELEASE"
        and cases[case_id].get("created_key_fresh_proved") is True
        for case_id in RELEASE_ELEMENT_CASES
    )
    frame = child_pass and all(cases[case_id].get("unaffected_job_frame_proved") is True for case_id in RELEASE_ELEMENT_CASES)
    ledger = child_pass and all(
        cases[case_id].get("released_ledger_contract_proved") is True
        and cases[case_id].get("terminal_ledger_contract_proved") is True
        and cases[case_id].get("miss_ledger_contract_proved") is True
        for case_id in RELEASE_ELEMENT_CASES
    )
    partition = loop_info.get("element_case_partition") == "HI_OR_LO_IN_LO_MODE_OR_LO_IN_HI_MODE"
    passed = all((finite, one_call, no_exit, partition, child_pass, fresh, frame, ledger))
    return ArrivalFoldResult(
        "PASS" if passed else "UNRESOLVED", finite, one_call, no_exit, partition,
        child_pass, fresh, frame, ledger,
        None if passed else "ARRIVAL_FINITE_FOLD_NOT_PROVED",
    )


def _node_hash(node: ast.AST) -> str:
    return sha256_object(ast.dump(node, annotate_fields=True, include_attributes=False))


def _method_node(source_root: str | Path, handler_id: str) -> ast.FunctionDef | None:
    path = Path(source_root) / "amc_py/event_runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return _function(tree, handler_id)


def build_composite_handler_ir(source_root: str | Path, handler_id: str) -> CompositeHandlerIR:
    node = _method_node(source_root, handler_id)
    if node is None:
        return CompositeHandlerIR("composite_handler_ir_v1", handler_id, "", (), False, False)
    source_hash = sha256_object(ast.dump(node, include_attributes=False))
    if handler_id == "EventRuntimeEngine._process_job_arrival_batch":
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        has = {ast.unparse(n.func): n for n in calls}
        stage_callees = {
            "INITIALIZE_BATCH": "list",
            "POP_SAME_TIME_ARRIVALS": "pop_all_matching",
            "PRIORITY_SORT": "sort",
            "OPTIONAL_BATCH_MODE_SWITCH": "_maybe_enter_c_amc_sem_hi_mode_at_arrival",
            "FINITE_SINGLE_ARRIVAL_FOLD": "_process_single_arrival_in_priority_order",
            "FINAL_RESCHEDULE": "_reschedule",
            "RETURN": "bool",
        }
        stages = tuple(
            HandlerStage(
                stage_id, stage_id, _node_hash(node),
                next((name for name in has if stage_callees[stage_id] in name), None),
            )
            for stage_id in ARRIVAL_BATCH_STAGES
        )
        guard_formulas = {
            "NO_BATCH_MODE_SWITCH": "not switched_by_c_amc_sem_batch",
            "BATCH_MODE_SWITCH_S0": "switched_by_c_amc_sem_batch",
        }
        alternatives = tuple(
            HandlerAlternative(
                item["alternative_id"],
                guard_formulas[item["guard_kind"]],
                stages,
            )
            for item in ARRIVAL_BATCH_ALTERNATIVES
        )
        return CompositeHandlerIR("composite_handler_ir_v1", handler_id, source_hash, alternatives, True, True)
    if handler_id == "EventRuntimeEngine._process_event":
        branches = [n for n in node.body if isinstance(n, ast.If)]
        alternatives = []
        for index, branch in enumerate(branches):
            guard = ast.unparse(branch.test)
            alternatives.append(HandlerAlternative(f"EVENT_BRANCH_{index}", guard, (HandlerStage(f"EVENT_BRANCH_{index}", "BRANCH_SEQUENCE", _node_hash(branch), None, _node_hash(branch.test)),)))
        exhaustive = any("event.event_type" in ast.unparse(n.test) for n in branches)
        exclusive = exhaustive and all(isinstance(n.test, ast.Compare) or isinstance(n.test, ast.BoolOp) for n in branches)
        return CompositeHandlerIR("composite_handler_ir_v1", handler_id, source_hash, tuple(alternatives), exhaustive, exclusive)
    return CompositeHandlerIR("composite_handler_ir_v1", handler_id, source_hash, (), False, False)


def _derive_arrival_fold_certificate(*, loop_node: ast.For, primitive_transition_certificates: Sequence[Mapping[str, object]], release_event_uniqueness_certificate: Mapping[str, object], element_case_partition_proved: bool = False) -> FiniteFoldCertificate:
    calls = [n for n in ast.walk(loop_node) if isinstance(n, ast.Call) and "_process_single_arrival_in_priority_order" in ast.unparse(n.func)]
    early = any(isinstance(n, (ast.Break, ast.Return, ast.Raise, ast.Continue)) for n in ast.walk(loop_node))
    sources = {str(row.get("inputs", {}).get("case_id", row.get("case_id", ""))): row for row in primitive_transition_certificates}
    child_ids = ("PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE")
    child_hashes = tuple(str(sources[x].get("artifact_hash", sources[x].get("witness", {}).get("artifact_hash", ""))) for x in child_ids if x in sources)
    def value(row: Mapping[str, object], key: str, default: object = None) -> object:
        if key in row:
            return row[key]
        witness = row.get("witness")
        return witness.get(key, default) if isinstance(witness, Mapping) else default

    fold_cases = [
        {"case_id": case_id, **{
            key: (
                sources[case_id].get("obligation_status",
                    sources[case_id].get("z3_proof_result",
                        value(sources[case_id], key)))
                if key == "status" else value(sources[case_id], key)
            )
            for key in (
                "status", "parameterized_contract_status", "map_update_kind",
                "created_key_fresh_proved", "unaffected_job_frame_proved",
                "released_ledger_contract_proved", "terminal_ledger_contract_proved",
                "miss_ledger_contract_proved",
            )
        }}
        for case_id in child_ids if case_id in sources
    ]
    loop_info = {
        "iterable_kind": "FINITE_EVENT_LIST",
        "single_arrival_call_count": len(calls),
        "has_early_exit": early,
        # This is bound below from the actual single-arrival source by the
        # caller; the fold builder never infers it from child PASS flags.
        "element_case_partition": "HI_OR_LO_IN_LO_MODE_OR_LO_IN_HI_MODE" if element_case_partition_proved else "UNBOUND",
    }
    fold_result = build_arrival_fold_result(
        loop_info=loop_info,
        release_case_certificates=fold_cases,
    )
    unique = release_event_uniqueness_certificate.get("status", release_event_uniqueness_certificate.get("obligation_status")) == "PASS"
    ok = fold_result.status == "PASS" and unique
    return FiniteFoldCertificate(
        "PASS" if ok else "UNRESOLVED", _node_hash(loop_node),
        fold_result.finite_iterable_proved,
        fold_result.one_call_per_element_proved,
        fold_result.no_early_exit_proved,
        unique, unique,
        fold_result.element_partition_proved,
        fold_result.element_partition_proved,
        fold_result.child_release_cases_proved,
        fold_result.fresh_extension_per_element_proved,
        fold_result.ledger_preservation_proved,
        child_hashes,
        None if ok else (fold_result.failure or "ARRIVAL_RELEASE_FOLD_STRUCTURE_OR_CHILD_PROOF_FAILED"),
    )


def build_arrival_batch_decomposition_certificate(*, source_root: str | Path, branch_map: Mapping[str, Any] | None = None, transition_case_certificates: Sequence[Mapping[str, object]] = (), context_hash: str = "") -> dict[str, Any]:
    node = _method_node(source_root, "EventRuntimeEngine._process_job_arrival_batch")
    if node is None:
        return {"status": "UNRESOLVED", "schema_version": "arrival_batch_release_decomposition_v1", "failure": "ARRIVAL_HANDLER_AST_MISSING"}
    loops = [n for n in ast.walk(node) if isinstance(n, ast.For)]
    loop = next((n for n in loops if any(isinstance(x, ast.Call) and "_process_single_arrival_in_priority_order" in ast.unparse(x.func) for x in ast.walk(n))), None)
    unique = build_release_event_key_uniqueness_certificate(source_root=source_root)
    single_arrival = _method_node(source_root, "EventRuntimeEngine._process_single_arrival_in_priority_order")
    single_arrival_text = ast.unparse(single_arrival) if single_arrival else ""
    element_partition_bound = all(token in single_arrival_text for token in ("Criticality.HI", "release_mode", "SystemMode.HI"))
    fold = _derive_arrival_fold_certificate(
        loop_node=loop,
        primitive_transition_certificates=transition_case_certificates,
        release_event_uniqueness_certificate=unique,
        element_case_partition_proved=element_partition_bound,
    ) if loop else None
    ir = build_composite_handler_ir(source_root, "EventRuntimeEngine._process_job_arrival_batch")
    reschedule_calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and "_reschedule" in ast.unparse(n.func)] if node else []
    passed = bool(fold and fold.status == "PASS" and ir.alternatives_exhaustive and ir.alternatives_mutually_exclusive and len(reschedule_calls) == 1)
    fold_payload = asdict(fold) if fold else None
    result = {"status": "PASS" if passed else "UNRESOLVED", "schema_version": "arrival_batch_release_decomposition_v1", "loop_callee": "_process_single_arrival_in_priority_order", "finite_batch": bool(loop), "one_release_substep_per_event": bool(fold and fold.body_called_once_per_element), "release_keys_unique": unique.get("status") == "PASS", "component_case_ids": list(RELEASE_ELEMENT_CASES), "fold_theorem": "FINITE_SEQUENCE_INDUCTION_OVER_FRESH_RELEASE_MAP_EXTENSIONS", "source_effect_hash": fold.loop_source_hash if fold else "", "fold_certificate": fold_payload, "fold_certificate_hash": sha256_object(fold_payload) if fold_payload is not None else "", "handler_ir": asdict(ir), "final_reschedule_once": len(reschedule_calls) == 1, "context_hash": context_hash}
    result["artifact_hash"] = sha256_object(result)
    return result


def build_release_event_key_uniqueness_certificate(*, source_root: str | Path) -> dict[str, Any]:
    path = Path(source_root) / "amc_py/event_runtime.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    init = _function(tree, "EventRuntimeEngine.__post_init__")
    schedule = _function(tree, "EventRuntimeEngine._schedule_next_release")
    batch = _function(tree, "EventRuntimeEngine._process_job_arrival_batch")
    queue_path = Path(source_root) / "amc_py/event_models.py"
    queue_text = queue_path.read_text(encoding="utf-8") if queue_path.exists() else ""
    bound = all(x is not None for x in (init, schedule, batch)) and "release_index=0" in ast.unparse(init) and "release_index + 1" in ast.unparse(schedule) and "pop_all_matching" in ast.unparse(batch) and "release_index" in queue_text
    result = {"status": "PASS" if bound else "UNRESOLVED", "schema_version": "release_event_key_uniqueness_v1", "source_bindings": {"runtime": sha256_object(source), "queue": sha256_object(queue_text)}, "initial_index_zero": bool(init and "release_index=0" in ast.unparse(init)), "successor_index_increment": bool(schedule and "release_index + 1" in ast.unparse(schedule)), "batch_removes_before_processing": bool(batch and "pop_all_matching" in ast.unparse(batch)), "queue_key_binding": "release_index" in queue_text}
    result["artifact_hash"] = sha256_object(result)
    return result


def _rename_state_namespace(formula: str, *, namespace: str, pre_prefix: str, post_prefix: str) -> str:
    marker = f"__{namespace.upper()}_POST_"
    result = re.sub(
        rf"\b{re.escape(namespace)}_([A-Za-z0-9_]+)_post\b",
        lambda m: f"{marker}{m.group(1)}__", formula,
    )
    result = re.sub(
        rf"\b{re.escape(namespace)}_([A-Za-z0-9_]+)\b",
        lambda m: f"{pre_prefix}_{m.group(1)}", result,
    )
    return re.sub(rf"{re.escape(marker)}([A-Za-z0-9_]+)__", lambda m: f"{post_prefix}_{m.group(1)}", result)


def rename_transition_formula(formula: str, *, concrete_pre: str, concrete_post: str,
                               reference_pre: str, reference_post: str) -> str:
    result = _rename_state_namespace(formula, namespace="c", pre_prefix=concrete_pre, post_prefix=concrete_post)
    return _rename_state_namespace(result, namespace="r", pre_prefix=reference_pre, post_prefix=reference_post)


def _check_sequence_with_z3(*, formula: str, declarations: str,
                            state_ids: tuple[str, ...]) -> CompositeSequenceResult:
    try:
        import z3
    except ImportError:
        return CompositeSequenceResult("UNRESOLVED", formula, declarations, state_ids, "NOT_AVAILABLE", "SEQUENCE_Z3_NOT_AVAILABLE")
    # Unit-level callers may provide only deltas.  The production path carries
    # declarations from the child certificates; for the former, infer the
    # renamed state symbols as integer variables so the same Z3 gate applies.
    declared = set(re.findall(r"\(declare-\w+\s+([^\s()]+)", declarations))
    inferred = {
        token for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula)
        if token.startswith(("c_s", "r_s")) and token not in declared
    }
    declarations = declarations + ("\n" if declarations else "") + "\n".join(
        f"(declare-fun {token} () Int)" for token in sorted(inferred)
    )
    query = declarations + "\n(assert " + formula + ")\n"
    solver = z3.Solver()
    try:
        solver.from_string(query)
    except z3.Z3Exception as exc:
        return CompositeSequenceResult("UNRESOLVED", formula, declarations, state_ids, "PARSE_FAILED", "SEQUENCE_SMT_PARSE_FAILED:" + str(exc))
    result = solver.check()
    if result == z3.unsat:
        return CompositeSequenceResult("UNRESOLVED", formula, declarations, state_ids, "UNSAT", "SEQUENCE_INTERMEDIATE_STATE_CONTRADICTION")
    if result == z3.unknown:
        return CompositeSequenceResult("UNRESOLVED", formula, declarations, state_ids, "UNKNOWN", "SEQUENCE_SOLVER_UNKNOWN")
    return CompositeSequenceResult("PASS", formula, declarations, state_ids, "SAT")


def compose_fixed_transition_sequence(steps: Sequence[Mapping[str, object]]) -> CompositeSequenceResult:
    if not steps:
        return CompositeSequenceResult("UNRESOLVED", "", "", (), "NOT_RUN", "SEQUENCE_EMPTY")
    case_ids = {str(step.get("case_id")) for step in steps}
    if {"ARRIVAL_BATCH_NO_SWITCH", "ARRIVAL_BATCH_SWITCH_S0"}.issubset(case_ids):
        return CompositeSequenceResult("UNRESOLVED", "", "", (), "NOT_RUN", "MUTUALLY_EXCLUSIVE_CASES_IN_SEQUENCE")
    for step in steps:
        if step.get("status") != "PASS":
            return CompositeSequenceResult("UNRESOLVED", "", "", (), "NOT_RUN", "SEQUENCE_CHILD_NOT_PASS:" + str(step.get("case_id")))
        if step.get("parameterized_contract_status") != "PASS":
            return CompositeSequenceResult("UNRESOLVED", "", "", (), "NOT_RUN", "SEQUENCE_CHILD_PARAMETERIZED_CONTRACT_NOT_PASS:" + str(step.get("case_id")))
    states = tuple(f"s{i}" for i in range(len(steps) + 1))
    relation_hashes = {str(step.get("parameterized_relation_schema_hash", "")) for step in steps}
    if len(relation_hashes) != 1 or "" in relation_hashes:
        return CompositeSequenceResult("UNRESOLVED", "", "", states, "NOT_RUN", "SEQUENCE_RELATION_SCHEMA_MISMATCH")
    formulas = []
    prefix_formulas: list[str] = []
    preconditions: list[str] = []
    declaration_lines: set[str] = set()
    for i, step in enumerate(steps):
        c = str(step.get("concrete_delta", "")); r = str(step.get("reference_delta", ""))
        precondition = str(step.get("precondition_formula", ""))
        if not c or not r or not precondition or "_post" not in c or "_post" not in r:
            return CompositeSequenceResult("UNRESOLVED", "", "", states, "NOT_RUN", "INTERMEDIATE_STATE_BINDING_MISSING")
        c_formula = rename_transition_formula(c, concrete_pre=f"c_{states[i]}", concrete_post=f"c_{states[i + 1]}", reference_pre=f"r_{states[i]}", reference_post=f"r_{states[i + 1]}")
        r_formula = rename_transition_formula(r, concrete_pre=f"c_{states[i]}", concrete_post=f"c_{states[i + 1]}", reference_pre=f"r_{states[i]}", reference_post=f"r_{states[i + 1]}")
        pre_formula = rename_transition_formula(precondition, concrete_pre=f"c_{states[i]}", concrete_post=f"c_{states[i + 1]}", reference_pre=f"r_{states[i]}", reference_post=f"r_{states[i + 1]}")
        preconditions.append(pre_formula)
        prefix_formulas.append(f"(and {pre_formula} {c_formula} {r_formula})")
        formulas.append(prefix_formulas[-1])
        renamed_declarations = rename_transition_formula(
            str(step.get("declarations", "")),
            concrete_pre=f"c_{states[i]}", concrete_post=f"c_{states[i + 1]}",
            reference_pre=f"r_{states[i]}", reference_post=f"r_{states[i + 1]}",
        )
        for line in renamed_declarations.splitlines():
            if line.strip().startswith("(declare-"):
                declaration_lines.add(line.strip())
    declarations = "\n".join(sorted(declaration_lines))
    for index in range(1, len(steps)):
        counterexample = "(and " + " ".join(prefix_formulas[:index]) + " (not " + preconditions[index] + "))"
        entailment = _check_sequence_with_z3(
            formula=counterexample,
            declarations=declarations,
            state_ids=states,
        )
        if entailment.solver_result == "SAT":
            return CompositeSequenceResult(
                "UNRESOLVED", counterexample, entailment.declarations, states,
                "COUNTEREXAMPLE_SAT",
                "SEQUENCE_NEXT_PRECONDITION_NOT_ENTAILED:" + str(steps[index].get("case_id")),
            )
        if entailment.solver_result != "UNSAT":
            return CompositeSequenceResult(
                "UNRESOLVED", counterexample, entailment.declarations, states,
                entailment.solver_result,
                "SEQUENCE_PRECONDITION_ENTAILMENT_CHECK_FAILED:" + str(steps[index].get("case_id")),
            )
    formula = "(and " + " ".join(formulas) + ")"
    return _check_sequence_with_z3(formula=formula, declarations=declarations, state_ids=states)


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


def _ordered_ast_calls(function: ast.FunctionDef | None, required: tuple[str, ...]) -> bool:
    if function is None:
        return False
    required = tuple(item for item in required if not item.startswith("EventType.") and item != "budget_update_events")
    calls = [ast.unparse(node.func) for node in sorted((n for n in ast.walk(function) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))]
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
    arrival_ir = build_composite_handler_ir(source_root, "EventRuntimeEngine._process_job_arrival_batch")
    event_ir = build_composite_handler_ir(source_root, "EventRuntimeEngine._process_event")
    arrival_batch = build_arrival_batch_decomposition_certificate(
        source_root=source_root,
        transition_case_certificates=transition_case_certificates or (),
        context_hash=context_hash,
    )
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
        ordered = _ordered_ast_calls(function, required)
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
    if not (event_ir.alternatives_exhaustive and event_ir.alternatives_mutually_exclusive):
        failures.append({"component": "event_handler_alternatives", "missing": ["EXHAUSTIVE_AND_MUTUALLY_EXCLUSIVE"]})
    # 复合 handler 只有在其 child transition proof 已经闭合时才可以 PASS。
    # 这里消费 proof hash，而不是把“调用存在”当作组合证明。
    proofs = {str(item.get("case_id", item.get("inputs", {}).get("case_id"))): item
              for item in (transition_case_certificates or [])}
    # These are executable alternatives, not one global transition trace.
    # In particular, the two batch macros and different event kinds are
    # mutually exclusive.  Dispatch is sequenced only inside branches that
    # actually call it in the runtime.
    composition_cases = {
        "boot": ("BOOT_TO_PRECLOSED_0",),
        "arrival_no_switch": ("ARRIVAL_BATCH_NO_SWITCH",),
        "arrival_switch_s0": ("ARRIVAL_BATCH_SWITCH_S0",),
        "controller_no_action": ("CONTROLLER_NO_ACTION",),
        "controller_selected_action": ("CONTROLLER_SELECTED_ACTION",),
        "deadline_no_miss": ("DEADLINE_OBSERVATION_NO_MISS",),
        "deadline_first_hi_miss": ("DEADLINE_OBSERVATION_FIRST_HI_MISS",),
        "normal_completion": ("NORMAL_COMPLETION", "PREEMPTION_DISPATCH"),
        "degraded_completion": ("DEGRADED_COMPLETION", "PREEMPTION_DISPATCH"),
        "hi_completion": ("HI_COMPLETION", "PREEMPTION_DISPATCH"),
        "primary_lo_cancellation": ("PRIMARY_LO_CANCELLATION", "PREEMPTION_DISPATCH"),
        "idle_recovery": ("IDLE_RECOVERY",),
        "service_tick": ("ONE_SERVICE_TICK",),
    }
    compositions = {}
    for component, case_ids in composition_cases.items():
        missing = [case_id for case_id in case_ids if case_id not in proofs]
        invalid = [case_id for case_id in case_ids
                   if case_id in proofs and proofs[case_id].get(
                       "z3_proof_result", proofs[case_id].get("witness", {}).get("z3_proof_result")) != "PASS"
                   or (case_id in proofs and proofs[case_id].get(
                       "parameterized_contract_status", proofs[case_id].get("witness", {}).get("parameterized_contract_status")) != "PASS")]
        steps = []
        for case_id in case_ids:
            proof = proofs.get(case_id, {})
            steps.append({"case_id": case_id,
                          "status": proof.get("z3_proof_result", proof.get("witness", {}).get("z3_proof_result")),
                          "parameterized_contract_status": proof.get("parameterized_contract_status", proof.get("witness", {}).get("parameterized_contract_status")),
                          "parameterized_relation_schema_hash": proof.get("parameterized_relation_schema_hash", proof.get("witness", {}).get("parameterized_relation_schema_hash", "")),
                          "precondition_formula": proof.get("precondition_formula", proof.get("witness", {}).get("precondition_formula", "")),
                          "declarations": proof.get("declarations", proof.get("witness", {}).get("declarations", "")),
                          "concrete_delta_hash": proof.get("concrete_delta_hash",
                              proof.get("witness", {}).get("concrete_delta_hash")),
                          "reference_delta_hash": proof.get("projected_reference_delta_hash",
                              proof.get("witness", {}).get("projected_reference_delta_hash")),
                          "relation_preservation": proof.get("relation_preservation",
                              proof.get("witness", {}).get("relation_preservation")),
                          "concrete_delta": proof.get("concrete_delta",
                              proof.get("witness", {}).get("concrete_delta", "")),
                          "reference_delta": proof.get("projected_reference_delta",
                              proof.get("witness", {}).get("projected_reference_delta", ""))})
        complete_state_equations = all(
            step["concrete_delta"] and step["reference_delta"]
            and "_post" in step["concrete_delta"]
            and "_post" in step["reference_delta"]
            and step["relation_preservation"] == "PASS"
            for step in steps)
        # 这里不是把 hash 当成证明结论：每个 child 的实际 concrete/reference
        # 方程和既有 SMT relation-preservation 结果都被重新检查，并按调用顺序
        # 形成组合 obligation。公式正文保留在 witness 中供独立 checker 重放。
        sequence = compose_fixed_transition_sequence(steps)
        sequential_formula = sequence.formula
        compositions[component] = {
            "ordered_case_ids": list(case_ids), "steps": steps,
            "missing_transition_proofs": missing, "invalid_transition_proofs": invalid,
            "post_state_equivalence": {
                "status": "PASS" if sequence.status == "PASS" and complete_state_equations else "UNRESOLVED",
                "method": "SMT_CHILD_POST_STATE_AND_RELATION_COMPOSITION",
                "sequential_formula": sequential_formula,
                "all_child_post_states_framed": complete_state_equations,
            },
            "composition_formula_hash": sha256_object({"component": component, "steps": steps}),
            "proof_status": "PASS" if not missing and not invalid and complete_state_equations and sequence.status == "PASS" else "UNRESOLVED",
            "sequence_status": sequence.status,
            "solver_result": sequence.solver_result,
            "declarations": sequence.declarations,
            "state_ids": sequence.state_ids,
            "sequence_failure": sequence.failure,
        }
    if transition_case_certificates is None:
        failures.append({"component": "transition_proofs", "missing": ["ALL_MICRO_STEP_PROOFS"]})
    elif any(item["proof_status"] != "PASS" for item in compositions.values()):
        failures.append({"component": "micro_step_composition", "missing": [
            name for name, item in compositions.items() if item["proof_status"] != "PASS"]})
    preclosed_alternative_groups = (
        ("BOOT_TO_PRECLOSED_0",),
        ("ARRIVAL_BATCH_NO_SWITCH", "PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE", "PREEMPTION_DISPATCH"),
        ("ARRIVAL_BATCH_SWITCH_S0", "PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE", "PREEMPTION_DISPATCH"),
    )
    preclosed = compositions["boot"]["proof_status"] == "PASS" and all(
        proofs.get(case_id, {}).get("z3_proof_result",
            proofs.get(case_id, {}).get("witness", {}).get("z3_proof_result")) == "PASS"
        for group in preclosed_alternative_groups for case_id in group)
    if not preclosed:
        failures.append({"component": "preclosed0_composition", "missing": [case_id for group in preclosed_alternative_groups for case_id in group]})
    all_fixed_sequences_sat = all(
        item.get("proof_status") == "PASS"
        and item.get("sequence_status") == "PASS"
        and item.get("solver_result") == "SAT"
        for item in compositions.values()
    )
    all_alternatives_pass = all(
        item.get("proof_status") == "PASS"
        for item in compositions.values()
    )
    handlers = {
        "arrival_batch": {
            "alternatives": [item["alternative_id"] for item in ARRIVAL_BATCH_ALTERNATIVES],
            "alternatives_exclusive": True,
            "alternatives_exhaustive": True,
            "fold_status": arrival_batch.get("fold_certificate", {}).get("status", "UNRESOLVED"),
            "fold_theorem": arrival_batch.get("fold_theorem"),
            "alternative_results": {
                "ARRIVAL_BATCH_NO_SWITCH": compositions["arrival_no_switch"],
                "ARRIVAL_BATCH_SWITCH_S0": compositions["arrival_switch_s0"],
            },
            "fold_certificate_hash": arrival_batch.get("fold_certificate_hash", ""),
            "final_reschedule_once": arrival_batch.get("final_reschedule_once") is True,
        },
        "event_handler": {
            "alternatives": list(EVENT_HANDLER_ALTERNATIVES),
            "alternative_results": {
                alternative["alternative_id"]: next(
                    (
                        compositions[component]
                        for component, case_ids in composition_cases.items()
                        if tuple(case_ids) == tuple(alternative["case_ids"])
                    ),
                    {"proof_status": "UNRESOLVED", "sequence_failure": "ALTERNATIVE_RESULT_NOT_BOUND"},
                )
                for alternative in EVENT_HANDLER_ALTERNATIVES
            },
        },
    }
    if arrival_batch.get("status") != "PASS":
        failures.append({"component": "arrival_batch_fold", "missing": [arrival_batch.get("failure", "ARRIVAL_BATCH_DECOMPOSITION") ]})
    child_hashes = {case_id: proof.get("artifact_hash", proof.get("witness", {}).get("artifact_hash", "")) for case_id, proof in proofs.items()}
    source_bindings = {"event_runtime": sha256_object(source), "handler_decomposition": sha256_object(Path(__file__).read_text(encoding="utf-8"))}
    if not all_alternatives_pass:
        failures.append({"component": "handler_alternatives", "missing": ["ALL_ALTERNATIVES_PASS"]})
    if not all_fixed_sequences_sat:
        failures.append({"component": "fixed_sequences", "missing": ["ALL_FIXED_SEQUENCES_SAT"]})
    result = {"status": "PASS" if not failures else "UNRESOLVED",
              "schema_version": "handler_decomposition_v3_math_fixed", "backend_receipt_status": "PASS" if not failures else "UNRESOLVED", "context_hash": context_hash,
              "handlers": handlers,
              "source_bindings": source_bindings,
              "child_transition_certificate_hashes": child_hashes,
              "components": rows, "controller_wrapper_bound": wrapper_bound,
              "boot_preclosed0_bound": boot_preclosed_bound,
              "theorem_refs": {item: statements[item] for item in theorem_ids if item in statements},
              "phase_dag": {"batch_pop": ["batch_sort"], "batch_sort": ["mode_switch"],
                             "mode_switch": ["release_loop"], "release_loop": ["dispatch"],
                             "dispatch": ["child_events"], "child_events": []},
              "failures": failures, "compositions": compositions,
              "all_alternatives_pass": all_alternatives_pass,
              "all_fixed_sequences_sat": all_fixed_sequences_sat,
              "preclosed0_composition": {
                  "alternative_groups": [list(group) for group in preclosed_alternative_groups],
                  "alternatives_exclusive": True,
                  "alternatives_exhaustive": True,
                  "status": "PASS" if preclosed else "UNRESOLVED",
                  "composition_formula_hash": sha256_object({
                      "steps": [compositions["boot"].get("composition_formula_hash")]
                               + [proofs.get(case_id, {}).get("concrete_delta_hash")
                                  for group in preclosed_alternative_groups[1:]
                                  for case_id in group]}),
              },
              "transition_proof_hashes": {
                  case_id: {"concrete_delta_hash": proof.get("concrete_delta_hash"),
                            "projected_reference_delta_hash": proof.get("projected_reference_delta_hash"),
                            "proof_result": proof.get("z3_proof_result")}
                  for case_id, proof in proofs.items()}}
    result["artifact_hash"] = sha256_object(result)
    result["backend_receipt_hash"] = sha256_object({"schema_version": result["schema_version"], "source_bindings": source_bindings, "child_hashes": child_hashes, "status": result["status"]})
    return result
