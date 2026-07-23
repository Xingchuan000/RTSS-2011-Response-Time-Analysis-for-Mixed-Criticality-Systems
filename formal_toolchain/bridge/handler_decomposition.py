"""复合 runtime handler 的有限 micro-step 分解检查。"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.z3_resources import new_context, new_solver


ARRIVAL_BATCH_ALTERNATIVES = (
    {"alternative_id": "ARRIVAL_BATCH_NO_SWITCH", "guard_kind": "NO_BATCH_MODE_SWITCH", "macro_case_id": "ARRIVAL_BATCH_NO_SWITCH"},
    {"alternative_id": "ARRIVAL_BATCH_SWITCH_S0", "guard_kind": "BATCH_MODE_SWITCH_S0", "macro_case_id": "ARRIVAL_BATCH_SWITCH_S0"},
)

RESCHEDULE_ALTERNATIVES = (
    "RESCHEDULE_KEEP_SAME", "RESCHEDULE_TO_IDLE", "PREEMPTION_DISPATCH",
)


def prove_reschedule_partition() -> dict[str, object]:
    """Machine-check the exhaustive, pairwise-disjoint reschedule partition."""
    try:
        import z3
        context = new_context(z3)
        selected_eq_previous = z3.Bool("selected_eq_previous", ctx=context)
        force = z3.Bool("force", ctx=context)
        selected_is_none = z3.Bool("selected_is_none", ctx=context)
        keep = z3.And(selected_eq_previous, z3.Not(force))
        idle = z3.And(z3.Not(keep), selected_is_none)
        dispatch = z3.And(z3.Not(keep), z3.Not(selected_is_none))
        exhaustive_solver = new_solver(z3, context=context)
        exhaustive = exhaustive_solver.check(
            z3.Not(z3.Or(keep, idle, dispatch))
        ) == z3.unsat
        exclusive = all(
            new_solver(z3, context=context).check(z3.And(left, right)) == z3.unsat
            for left, right in ((keep, idle), (keep, dispatch), (idle, dispatch)))
        status = "PASS" if exhaustive and exclusive else "FAIL"
    except ImportError:
        return {"status": "UNRESOLVED", "failure": "Z3_REQUIRED",
                "cases": list(RESCHEDULE_ALTERNATIVES), "exhaustive": False,
                "pairwise_exclusive": False}
    return {"status": status, "cases": list(RESCHEDULE_ALTERNATIVES),
            "exhaustive": exhaustive, "pairwise_exclusive": exclusive}


def prove_handler_reschedule_unreachability() -> dict[str, object]:
    """Discharge context-excluded family alternatives as UNSAT, not PASS."""
    try:
        import z3
        context = new_context(z3)
        previous_none = z3.Bool("handler_previous_none", ctx=context)
        selected_none = z3.Bool("handler_selected_none", ctx=context)
        selected_eq_previous = z3.Bool(
            "handler_selected_eq_previous", ctx=context
        )
        force = z3.Bool("handler_force", ctx=context)
        keep = z3.And(selected_eq_previous, z3.Not(force))
        idle = z3.And(z3.Not(keep), selected_none)
        checks = {
            "completion_to_idle": z3.And(previous_none, selected_none, selected_eq_previous, z3.Not(force), idle),
            "controller_force_keep": z3.And(force, keep),
        }
        result = {
            name: "UNSAT"
            if new_solver(z3, context=context).check(formula) == z3.unsat
            else "SAT"
            for name, formula in checks.items()
        }
        return {"status": "PASS" if all(value == "UNSAT" for value in result.values()) else "FAIL",
                "proofs": result}
    except ImportError:
        return {"status": "UNRESOLVED", "failure": "Z3_REQUIRED"}



def prove_arrival_reschedule_partition(
    arrival_batch_certificate: Mapping[str, object],
) -> dict[str, object]:
    """Prove the post-arrival reschedule partition for C-AMC-sem.

    ``events`` is initialized with ``first_event``, so the batch is non-empty.
    Under the P0 C-AMC-sem primitive partition every batch element creates one
    fresh active unfinished job.  Consequently ``selected is None`` is
    unreachable after the fold; the final reschedule is exactly KEEP or
    DISPATCH depending on whether the selected job equals the previous runner.
    """
    fold = arrival_batch_certificate.get("fold_certificate", {})
    if not isinstance(fold, Mapping):
        return {
            "status": "UNRESOLVED",
            "failure": "ARRIVAL_FOLD_CERTIFICATE_REQUIRED",
        }

    structural = (
        arrival_batch_certificate.get("status") == "PASS"
        and arrival_batch_certificate.get("batch_nonempty") is True
        and arrival_batch_certificate.get("one_release_substep_per_event") is True
        and arrival_batch_certificate.get("release_keys_unique") is True
        and fold.get("status") == "PASS"
        and fold.get("iterable_is_finite") is True
        and fold.get("body_called_once_per_element") is True
        and fold.get("loop_has_no_early_exit") is True
        and fold.get("element_case_partition_complete") is True
        and fold.get("element_case_partition_exclusive") is True
        and fold.get("fold_extends_job_map") is True
        and fold.get("fold_preserves_relation") is True
        and arrival_batch_certificate.get("every_element_creates_fresh_job") is True
    )
    if not structural:
        return {
            "status": "UNRESOLVED",
            "failure": "ARRIVAL_POST_FOLD_NONEMPTY_READY_NOT_PROVED",
            "idle_unreachable": False,
            "keep_dispatch_exhaustive": False,
            "keep_dispatch_exclusive": False,
        }

    try:
        import z3
    except ImportError:
        return {
            "status": "UNRESOLVED",
            "failure": "Z3_REQUIRED",
            "idle_unreachable": False,
            "keep_dispatch_exhaustive": False,
            "keep_dispatch_exclusive": False,
        }

    context = new_context(z3)
    selected_eq_previous = z3.Bool(
        "arrival_selected_eq_previous", ctx=context
    )
    selected_is_none = z3.Bool("arrival_selected_is_none", ctx=context)
    selected_nonempty = z3.Not(selected_is_none)
    keep = z3.And(selected_nonempty, selected_eq_previous)
    dispatch = z3.And(selected_nonempty, z3.Not(selected_eq_previous))
    idle = selected_is_none

    idle_solver = new_solver(z3, context=context)
    idle_solver.add(selected_nonempty, idle)
    idle_unreachable = idle_solver.check() == z3.unsat

    exhaustive_solver = new_solver(z3, context=context)
    exhaustive_solver.add(selected_nonempty, z3.Not(z3.Or(keep, dispatch)))
    exhaustive = exhaustive_solver.check() == z3.unsat

    exclusive_solver = new_solver(z3, context=context)
    exclusive_solver.add(keep, dispatch)
    exclusive = exclusive_solver.check() == z3.unsat

    passed = idle_unreachable and exhaustive and exclusive
    return {
        "status": "PASS" if passed else "UNRESOLVED",
        "idle_unreachable": idle_unreachable,
        "keep_dispatch_exhaustive": exhaustive,
        "keep_dispatch_exclusive": exclusive,
        "reachable_cases": [
            "RESCHEDULE_KEEP_SAME",
            "PREEMPTION_DISPATCH",
        ],
        "unreachable_cases": ["RESCHEDULE_TO_IDLE"],
        "basis": (
            "NONEMPTY_FINITE_RELEASE_FOLD_"
            "PLUS_HIGHEST_PRIORITY_SELECTION"
        ),
        "failure": None if passed else "ARRIVAL_RESCHEDULE_PARTITION_FAILED",
    }


EVENT_HANDLER_ALTERNATIVES = (
    {"alternative_id": "CONTROLLER_NO_ACTION_IDLE",
     "component": "controller_no_action_idle"},
    {"alternative_id": "CONTROLLER_NO_ACTION_DISPATCH",
     "component": "controller_no_action_dispatch"},
    {"alternative_id": "CONTROLLER_SELECTED_ACTION_IDLE",
     "component": "controller_selected_action_idle"},
    {"alternative_id": "CONTROLLER_SELECTED_ACTION_DISPATCH",
     "component": "controller_selected_action_dispatch"},
    {"alternative_id": "JOB_ARRIVAL_NO_SWITCH_KEEP",
     "component": "arrival_no_switch_keep"},
    {"alternative_id": "JOB_ARRIVAL_NO_SWITCH_DISPATCH",
     "component": "arrival_no_switch_dispatch"},
    {"alternative_id": "JOB_ARRIVAL_SWITCH_KEEP",
     "component": "arrival_switch_s0_keep"},
    {"alternative_id": "JOB_ARRIVAL_SWITCH_DISPATCH",
     "component": "arrival_switch_s0_dispatch"},
    {"alternative_id": "DEADLINE_NO_MISS",
     "component": "deadline_no_miss"},
    {"alternative_id": "DEADLINE_FIRST_HI_MISS",
     "component": "deadline_first_hi_miss"},
    {"alternative_id": "NORMAL_COMPLETION_KEEP",
     "component": "normal_completion_keep"},
    {"alternative_id": "NORMAL_COMPLETION_DISPATCH",
     "component": "normal_completion_dispatch"},
    {"alternative_id": "DEGRADED_COMPLETION_KEEP",
     "component": "degraded_completion_keep"},
    {"alternative_id": "DEGRADED_COMPLETION_DISPATCH",
     "component": "degraded_completion_dispatch"},
    {"alternative_id": "HI_COMPLETION_KEEP",
     "component": "hi_completion_keep"},
    {"alternative_id": "HI_COMPLETION_DISPATCH",
     "component": "hi_completion_dispatch"},
    {"alternative_id": "PRIMARY_LO_CANCELLATION_KEEP",
     "component": "primary_lo_cancellation_keep"},
    {"alternative_id": "PRIMARY_LO_CANCELLATION_DISPATCH",
     "component": "primary_lo_cancellation_dispatch"},
    {"alternative_id": "IDLE_RECOVERY",
     "component": "idle_recovery"},
    {"alternative_id": "SERVICE_TICK",
     "component": "service_tick"},
)


# Each value is one executable sequence.  Mutually exclusive reschedule
# outcomes are deliberately represented by different dictionary entries.
HANDLER_COMPOSITION_CASES = {
    "boot": ("BOOT_TO_PRECLOSED_0",),

    "arrival_no_switch_keep":
        ("ARRIVAL_BATCH_NO_SWITCH", "RESCHEDULE_KEEP_SAME"),
    "arrival_no_switch_dispatch":
        ("ARRIVAL_BATCH_NO_SWITCH", "PREEMPTION_DISPATCH"),

    "arrival_switch_s0_keep":
        ("ARRIVAL_BATCH_SWITCH_S0", "RESCHEDULE_KEEP_SAME"),
    "arrival_switch_s0_dispatch":
        ("ARRIVAL_BATCH_SWITCH_S0", "PREEMPTION_DISPATCH"),

    # apply_budget_updates invokes _reschedule(..., force=True), hence KEEP
    # is unreachable and only IDLE/DISPATCH are executable.
    "controller_no_action_idle":
        ("CONTROLLER_NO_ACTION", "RESCHEDULE_TO_IDLE"),
    "controller_no_action_dispatch":
        ("CONTROLLER_NO_ACTION", "PREEMPTION_DISPATCH"),
    "controller_selected_action_idle":
        ("CONTROLLER_SELECTED_ACTION", "RESCHEDULE_TO_IDLE"),
    "controller_selected_action_dispatch":
        ("CONTROLLER_SELECTED_ACTION", "PREEMPTION_DISPATCH"),

    "deadline_no_miss": ("DEADLINE_OBSERVATION_NO_MISS",),
    "deadline_first_hi_miss":
        ("DEADLINE_OBSERVATION_FIRST_HI_MISS",),

    # Completion/cancellation clears running_job before _reschedule(force=False).
    # selected=None is therefore KEEP-SAME (None==None); TO_IDLE is unreachable.
    "normal_completion_keep":
        ("NORMAL_COMPLETION", "RESCHEDULE_KEEP_SAME"),
    "normal_completion_dispatch":
        ("NORMAL_COMPLETION", "PREEMPTION_DISPATCH"),
    "degraded_completion_keep":
        ("DEGRADED_COMPLETION", "RESCHEDULE_KEEP_SAME"),
    "degraded_completion_dispatch":
        ("DEGRADED_COMPLETION", "PREEMPTION_DISPATCH"),
    "hi_completion_keep":
        ("HI_COMPLETION", "RESCHEDULE_KEEP_SAME"),
    "hi_completion_dispatch":
        ("HI_COMPLETION", "PREEMPTION_DISPATCH"),
    "primary_lo_cancellation_keep":
        ("PRIMARY_LO_CANCELLATION", "RESCHEDULE_KEEP_SAME"),
    "primary_lo_cancellation_dispatch":
        ("PRIMARY_LO_CANCELLATION", "PREEMPTION_DISPATCH"),

    "idle_recovery": ("IDLE_RECOVERY",),
    "service_tick": ("ONE_SERVICE_TICK",),
}


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
    feasibility_result: str
    precondition_chain_result: str
    relation_chain_result: str
    step_precondition_results: tuple[dict[str, object], ...]
    failure: str | None = None

    @property
    def solver_result(self) -> str:
        """兼容旧报告字段；这里只表示整体 sequence 的可满足性。"""
        return self.feasibility_result


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
    assignments = [n for n in ast.walk(node) if isinstance(n, (ast.Assign, ast.AnnAssign))]
    batch_nonempty = any(
        "events = [first_event]" in ast.unparse(item)
        for item in assignments
    )
    every_element_creates_fresh_job = bool(
        fold
        and fold.status == "PASS"
        and fold.fold_extends_job_map
        and fold.input_keys_unique
        and len(fold.child_certificate_hashes) == len(RELEASE_ELEMENT_CASES)
    )
    passed = bool(
        fold
        and fold.status == "PASS"
        and batch_nonempty
        and every_element_creates_fresh_job
        and ir.alternatives_exhaustive
        and ir.alternatives_mutually_exclusive
        and len(reschedule_calls) == 1
    )
    fold_payload = asdict(fold) if fold else None
    result = {
        "status": "PASS" if passed else "UNRESOLVED",
        "schema_version": "arrival_batch_release_decomposition_v1",
        "loop_callee": "_process_single_arrival_in_priority_order",
        "finite_batch": bool(loop),
        "batch_nonempty": batch_nonempty,
        "one_release_substep_per_event": bool(
            fold and fold.body_called_once_per_element
        ),
        "release_keys_unique": unique.get("status") == "PASS",
        "every_element_creates_fresh_job": every_element_creates_fresh_job,
        "component_case_ids": list(RELEASE_ELEMENT_CASES),
        "fold_theorem": (
            "FINITE_SEQUENCE_INDUCTION_OVER_"
            "FRESH_RELEASE_MAP_EXTENSIONS"
        ),
        "source_effect_hash": fold.loop_source_hash if fold else "",
        "fold_certificate": fold_payload,
        "fold_certificate_hash": (
            sha256_object(fold_payload)
            if fold_payload is not None
            else ""
        ),
        "handler_ir": asdict(ir),
        "final_reschedule_once": len(reschedule_calls) == 1,
        "context_hash": context_hash,
    }
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


def _prepare_sequence_declarations(*, formulas: Sequence[str], declarations: str) -> str:
    """补齐重命名后未显式声明的有限状态变量。"""
    declared = set(re.findall(r"\(declare-\w+\s+([^\s()]+)", declarations))
    text = "\n".join(formulas)
    inferred = {
        token
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
        if token.startswith(("c_s", "r_s")) and token not in declared
    }
    extra = "\n".join(
        f"(declare-fun {token} () Int)"
        for token in sorted(inferred)
    )
    if not extra:
        return declarations
    return declarations + ("\n" if declarations else "") + extra


def _run_sequence_z3_query(*, declarations: str, assertion: str) -> tuple[str, str | None, str]:
    try:
        import z3
    except ImportError:
        return "NOT_AVAILABLE", "SEQUENCE_Z3_NOT_AVAILABLE", declarations

    prepared = _prepare_sequence_declarations(
        formulas=(assertion,),
        declarations=declarations,
    )
    context = new_context(z3)
    solver = new_solver(z3, context=context)
    query = prepared + "\n(assert " + assertion + ")\n"
    try:
        solver.from_string(query)
    except z3.Z3Exception as exc:
        return "PARSE_FAILED", "SEQUENCE_SMT_PARSE_FAILED:" + str(exc), prepared

    result = solver.check()
    if result == z3.sat:
        return "SAT", None, prepared
    if result == z3.unsat:
        return "UNSAT", None, prepared
    return "UNKNOWN", "SEQUENCE_SOLVER_UNKNOWN", prepared


def _sequence_unresolved(
    failure: str,
    *,
    formula: str = "",
    declarations: str = "",
    states: tuple[str, ...] = (),
    feasibility_result: str = "NOT_RUN",
    precondition_chain_result: str = "NOT_RUN",
    relation_chain_result: str = "NOT_RUN",
    step_precondition_results: Sequence[Mapping[str, object]] = (),
) -> CompositeSequenceResult:
    return CompositeSequenceResult(
        status="UNRESOLVED",
        formula=formula,
        declarations=declarations,
        state_ids=states,
        feasibility_result=feasibility_result,
        precondition_chain_result=precondition_chain_result,
        relation_chain_result=relation_chain_result,
        step_precondition_results=tuple(dict(item) for item in step_precondition_results),
        failure=failure,
    )


def compose_fixed_transition_sequence(
    steps: Sequence[Mapping[str, object]],
    *,
    guarded_branch_boundaries: Sequence[int] = (),
) -> CompositeSequenceResult:
    """组合固定顺序的 micro-steps。

    数学门禁分三层：
    1. 每个 child 已分别证明 concrete feasibility、reference totality 与 relation preservation；
    2. 每一步的 relation-preservation 公式被显式放入中间状态链；
    3. 对每个相邻阶段检查 prefix => next precondition，排除仅存在一条偶然 SAT 样例的假阳性。
    """
    if not steps:
        return _sequence_unresolved("SEQUENCE_EMPTY")

    case_ids = {str(step.get("case_id")) for step in steps}
    if {"ARRIVAL_BATCH_NO_SWITCH", "ARRIVAL_BATCH_SWITCH_S0"}.issubset(case_ids):
        return _sequence_unresolved("MUTUALLY_EXCLUSIVE_CASES_IN_SEQUENCE")

    for step in steps:
        case_id = str(step.get("case_id"))
        if step.get("status") != "PASS":
            return _sequence_unresolved("SEQUENCE_CHILD_NOT_PASS:" + case_id)
        if step.get("concrete_feasibility") != "SAT":
            return _sequence_unresolved("SEQUENCE_CHILD_CONCRETE_FEASIBILITY_MISSING:" + case_id)
        if step.get("reference_totality") != "PASS":
            return _sequence_unresolved("SEQUENCE_CHILD_REFERENCE_TOTALITY_MISSING:" + case_id)
        if step.get("relation_preservation") != "PASS":
            return _sequence_unresolved("SEQUENCE_CHILD_RELATION_PRESERVATION_MISSING:" + case_id)
        if step.get("parameterized_contract_status") != "PASS":
            return _sequence_unresolved("SEQUENCE_CHILD_PARAMETERIZED_CONTRACT_NOT_PASS:" + case_id)

        required_formulas = (
            "precondition_formula",
            "concrete_delta",
            "reference_delta",
            "relation_preservation_formula",
        )
        missing = [
            name
            for name in required_formulas
            if not isinstance(step.get(name), str) or not str(step.get(name)).strip()
        ]
        if missing:
            return _sequence_unresolved(
                "SEQUENCE_CHILD_FORMULA_MISSING:"
                + case_id
                + ":"
                + ",".join(missing)
            )

    states = tuple(f"s{i}" for i in range(len(steps) + 1))
    guarded_boundaries = set(int(index) for index in guarded_branch_boundaries)
    if any(index <= 0 or index >= len(steps) for index in guarded_boundaries):
        return _sequence_unresolved(
            "INVALID_GUARDED_BRANCH_BOUNDARY",
            states=states,
        )
    relation_hashes = {
        str(step.get("parameterized_relation_schema_hash", ""))
        for step in steps
    }
    if len(relation_hashes) != 1 or "" in relation_hashes:
        return _sequence_unresolved(
            "SEQUENCE_RELATION_SCHEMA_MISMATCH",
            states=states,
        )

    transition_bodies: list[str] = []
    preconditions: list[str] = []
    declaration_lines: set[str] = set()

    for index, step in enumerate(steps):
        concrete_pre = f"c_{states[index]}"
        concrete_post = f"c_{states[index + 1]}"
        reference_pre = f"r_{states[index]}"
        reference_post = f"r_{states[index + 1]}"

        pre_formula = rename_transition_formula(
            str(step["precondition_formula"]),
            concrete_pre=concrete_pre,
            concrete_post=concrete_post,
            reference_pre=reference_pre,
            reference_post=reference_post,
        )
        concrete_formula = rename_transition_formula(
            str(step["concrete_delta"]),
            concrete_pre=concrete_pre,
            concrete_post=concrete_post,
            reference_pre=reference_pre,
            reference_post=reference_post,
        )
        reference_formula = rename_transition_formula(
            str(step["reference_delta"]),
            concrete_pre=concrete_pre,
            concrete_post=concrete_post,
            reference_pre=reference_pre,
            reference_post=reference_post,
        )
        relation_post_formula = rename_transition_formula(
            str(step["relation_preservation_formula"]),
            concrete_pre=concrete_pre,
            concrete_post=concrete_post,
            reference_pre=reference_pre,
            reference_post=reference_post,
        )

        if "_post" in " ".join((concrete_formula, reference_formula, relation_post_formula)):
            return _sequence_unresolved(
                "INTERMEDIATE_STATE_RENAMING_INCOMPLETE:" + str(step.get("case_id")),
                states=states,
            )

        preconditions.append(pre_formula)
        transition_bodies.append(
            "(and "
            + pre_formula
            + " "
            + concrete_formula
            + " "
            + reference_formula
            + " "
            + relation_post_formula
            + ")"
        )

        renamed_declarations = rename_transition_formula(
            str(step.get("declarations", "")),
            concrete_pre=concrete_pre,
            concrete_post=concrete_post,
            reference_pre=reference_pre,
            reference_post=reference_post,
        )
        for line in renamed_declarations.splitlines():
            stripped = line.strip()
            if stripped.startswith("(declare-"):
                declaration_lines.add(stripped)

    declarations = "\n".join(sorted(declaration_lines))
    combined_formula = "(and " + " ".join(transition_bodies) + ")"

    feasibility_result, failure, prepared_declarations = _run_sequence_z3_query(
        declarations=declarations,
        assertion=combined_formula,
    )
    if feasibility_result == "UNSAT":
        return _sequence_unresolved(
            "SEQUENCE_INTERMEDIATE_STATE_CONTRADICTION",
            formula=combined_formula,
            declarations=prepared_declarations,
            states=states,
            feasibility_result="UNSAT",
            relation_chain_result="PASS",
        )
    if feasibility_result != "SAT":
        return _sequence_unresolved(
            failure or "SEQUENCE_FEASIBILITY_UNKNOWN",
            formula=combined_formula,
            declarations=prepared_declarations,
            states=states,
            feasibility_result=feasibility_result,
            relation_chain_result="PASS",
        )

    precondition_results: list[dict[str, object]] = []
    for index in range(1, len(steps)):
        prefix_formula = "(and " + " ".join(transition_bodies[:index]) + ")"
        if index in guarded_boundaries:
            # This boundary is an explicit handler alternative.  The preceding
            # macro-step need not imply one particular mutually exclusive
            # reschedule guard.  Soundness comes from:
            #   (a) SAT of prefix ∧ selected branch precondition,
            #   (b) the separately machine-checked exhaustive/exclusive
            #       partition, and
            #   (c) the primitive case theorem for that branch.
            guarded_formula = (
                "(and "
                + prefix_formula
                + " "
                + preconditions[index]
                + ")"
            )
            result, entailment_failure, prepared_declarations = (
                _run_sequence_z3_query(
                    declarations=prepared_declarations,
                    assertion=guarded_formula,
                )
            )
            item = {
                "from_case_id": str(steps[index - 1].get("case_id")),
                "to_case_id": str(steps[index].get("case_id")),
                "solver_result": result,
                "check_kind": "GUARDED_HANDLER_ALTERNATIVE",
            }
            precondition_results.append(item)
            if result != "SAT":
                return _sequence_unresolved(
                    entailment_failure
                    or "SEQUENCE_GUARDED_BRANCH_INFEASIBLE:"
                    + str(steps[index].get("case_id")),
                    formula=combined_formula,
                    declarations=prepared_declarations,
                    states=states,
                    feasibility_result="SAT",
                    precondition_chain_result="UNRESOLVED",
                    relation_chain_result="PASS",
                    step_precondition_results=precondition_results,
                )
            continue

        counterexample = (
            "(and "
            + prefix_formula
            + " (not "
            + preconditions[index]
            + "))"
        )
        result, entailment_failure, prepared_declarations = _run_sequence_z3_query(
            declarations=prepared_declarations,
            assertion=counterexample,
        )
        item = {
            "from_case_id": str(steps[index - 1].get("case_id")),
            "to_case_id": str(steps[index].get("case_id")),
            "solver_result": result,
            "check_kind": "UNIVERSAL_PRECONDITION_ENTAILMENT",
        }
        precondition_results.append(item)
        if result == "SAT":
            return _sequence_unresolved(
                "SEQUENCE_NEXT_PRECONDITION_NOT_ENTAILED:"
                + str(steps[index].get("case_id")),
                formula=combined_formula,
                declarations=prepared_declarations,
                states=states,
                feasibility_result="SAT",
                precondition_chain_result="FAIL",
                relation_chain_result="PASS",
                step_precondition_results=precondition_results,
            )
        if result != "UNSAT":
            return _sequence_unresolved(
                entailment_failure
                or "SEQUENCE_PRECONDITION_ENTAILMENT_CHECK_FAILED:"
                + str(steps[index].get("case_id")),
                formula=combined_formula,
                declarations=prepared_declarations,
                states=states,
                feasibility_result="SAT",
                precondition_chain_result="UNRESOLVED",
                relation_chain_result="PASS",
                step_precondition_results=precondition_results,
            )

    return CompositeSequenceResult(
        status="PASS",
        formula=combined_formula,
        declarations=prepared_declarations,
        state_ids=states,
        feasibility_result="SAT",
        precondition_chain_result="PASS",
        relation_chain_result="PASS",
        step_precondition_results=tuple(precondition_results),
    )


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
    composition_cases = HANDLER_COMPOSITION_CASES
    compositions = {}
    reschedule_partition = prove_reschedule_partition()
    unreachable_reschedule = prove_handler_reschedule_unreachability()
    arrival_reschedule_partition = prove_arrival_reschedule_partition(
        arrival_batch
    )
    if reschedule_partition.get("status") != "PASS":
        failures.append({"component": "reschedule_partition",
                         "missing": [reschedule_partition.get("failure", "PASS") ]})
    if unreachable_reschedule.get("status") != "PASS":
        failures.append({
            "component": "handler_reschedule_unreachability",
            "missing": [unreachable_reschedule.get("failure", "UNREACHABILITY_NOT_PROVED")],
        })
    if arrival_reschedule_partition.get("status") != "PASS":
        failures.append({
            "component": "arrival_reschedule_partition",
            "missing": [
                arrival_reschedule_partition.get(
                    "failure",
                    "ARRIVAL_RESCHEDULE_PARTITION_NOT_PROVED",
                )
            ],
        })
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
            witness = proof.get("witness", {}) if isinstance(proof.get("witness"), Mapping) else {}
            steps.append({
                "case_id": case_id,
                "status": proof.get("z3_proof_result", witness.get("z3_proof_result")),
                "concrete_feasibility": proof.get("concrete_feasibility", witness.get("concrete_feasibility")),
                "reference_totality": proof.get("reference_totality", witness.get("reference_totality")),
                "relation_preservation": proof.get("relation_preservation", witness.get("relation_preservation")),
                "parameterized_contract_status": proof.get("parameterized_contract_status", witness.get("parameterized_contract_status")),
                "parameterized_relation_schema_hash": proof.get("parameterized_relation_schema_hash", witness.get("parameterized_relation_schema_hash", "")),
                "precondition_formula": proof.get("precondition_formula", witness.get("precondition_formula", "")),
                "declarations": proof.get("declarations", witness.get("declarations", "")),
                "concrete_delta_hash": proof.get("concrete_delta_hash", witness.get("concrete_delta_hash")),
                "reference_delta_hash": proof.get("projected_reference_delta_hash", witness.get("projected_reference_delta_hash")),
                "concrete_delta": proof.get("concrete_delta", witness.get("concrete_delta", "")),
                "reference_delta": proof.get("projected_reference_delta", witness.get("projected_reference_delta", "")),
                "relation_preservation_formula": proof.get("relation_preservation_formula", witness.get("relation_preservation_formula", "")),
            })
        complete_state_equations = all(
            step["concrete_delta"]
            and step["reference_delta"]
            and step["relation_preservation_formula"]
            and "_post" in step["concrete_delta"]
            and "_post" in step["reference_delta"]
            and "_post" in step["relation_preservation_formula"]
            and step["concrete_feasibility"] == "SAT"
            and step["reference_totality"] == "PASS"
            and step["relation_preservation"] == "PASS"
            for step in steps
        )
        # 这里不是把 hash 当成证明结论：每个 child 的实际 concrete/reference
        # 方程和既有 SMT relation-preservation 结果都被重新检查，并按调用顺序
        # 形成组合 obligation。公式正文保留在 witness 中供独立 checker 重放。
        guarded_boundaries = (
            (1,)
            if len(case_ids) == 2
            and case_ids[1] in RESCHEDULE_ALTERNATIVES
            else ()
        )
        sequence = compose_fixed_transition_sequence(
            steps,
            guarded_branch_boundaries=guarded_boundaries,
        )
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
            "feasibility_result": sequence.feasibility_result,
            "precondition_chain_result": sequence.precondition_chain_result,
            "relation_chain_result": sequence.relation_chain_result,
            "step_precondition_results": list(sequence.step_precondition_results),
            "declarations": sequence.declarations,
            "state_ids": sequence.state_ids,
            "guarded_branch_boundaries": list(guarded_boundaries),
            "sequence_failure": sequence.failure,
        }
    if transition_case_certificates is None:
        failures.append({"component": "transition_proofs", "missing": ["ALL_MICRO_STEP_PROOFS"]})
    elif any(item["proof_status"] != "PASS" for item in compositions.values()):
        failures.append({"component": "micro_step_composition", "missing": [
            name for name, item in compositions.items() if item["proof_status"] != "PASS"]})
    arrival_no_switch_components = (
        "arrival_no_switch_keep",
        "arrival_no_switch_dispatch",
    )
    arrival_switch_components = (
        "arrival_switch_s0_keep",
        "arrival_switch_s0_dispatch",
    )
    preclosed_alternative_groups = (
        tuple(HANDLER_COMPOSITION_CASES[name] for name in arrival_no_switch_components),
        tuple(HANDLER_COMPOSITION_CASES[name] for name in arrival_switch_components),
    )
    preclosed = (
        compositions["boot"]["proof_status"] == "PASS"
        and arrival_batch.get("status") == "PASS"
        and reschedule_partition.get("status") == "PASS"
        and arrival_reschedule_partition.get("status") == "PASS"
        and all(
            compositions[name].get("proof_status") == "PASS"
            for name in arrival_no_switch_components + arrival_switch_components
        )
    )
    if not preclosed:
        failures.append({
            "component": "preclosed0_composition",
            "missing": [
                name
                for name in arrival_no_switch_components + arrival_switch_components
                if compositions[name].get("proof_status") != "PASS"
            ],
        })
    all_fixed_sequences_proved = all(
        item.get("proof_status") == "PASS"
        and item.get("sequence_status") == "PASS"
        and item.get("feasibility_result") == "SAT"
        and item.get("precondition_chain_result") == "PASS"
        and item.get("relation_chain_result") == "PASS"
        for item in compositions.values()
    )
    all_alternatives_pass = all(
        item.get("proof_status") == "PASS"
        for item in compositions.values()
    )
    def aggregate_components(
        names: Sequence[str],
        *,
        partition: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        component_results = {name: compositions[name] for name in names}
        partition_ok = (
            True
            if partition is None
            else partition.get("status") == "PASS"
        )
        passed = (
            partition_ok
            and all(
                item.get("proof_status") == "PASS"
                for item in component_results.values()
            )
        )
        return {
            "proof_status": "PASS" if passed else "UNRESOLVED",
            "components": component_results,
            "alternatives_exclusive": (
                True
                if partition is None
                else bool(
                    partition.get(
                        "keep_dispatch_exclusive",
                        partition.get("pairwise_exclusive"),
                    )
                )
            ),
            "alternatives_exhaustive": (
                True
                if partition is None
                else bool(
                    partition.get(
                        "keep_dispatch_exhaustive",
                        partition.get("exhaustive"),
                    )
                )
            ),
            "partition": dict(partition) if partition is not None else None,
        }

    handlers = {
        "arrival_batch": {
            "alternatives": [item["alternative_id"] for item in ARRIVAL_BATCH_ALTERNATIVES],
            "alternatives_exclusive": True,
            "alternatives_exhaustive": True,
            "fold_status": arrival_batch.get("fold_certificate", {}).get("status", "UNRESOLVED"),
            "fold_theorem": arrival_batch.get("fold_theorem"),
            "alternative_results": {
                "ARRIVAL_BATCH_NO_SWITCH": aggregate_components(
                    arrival_no_switch_components,
                    partition=arrival_reschedule_partition,
                ),
                "ARRIVAL_BATCH_SWITCH_S0": aggregate_components(
                    arrival_switch_components,
                    partition=arrival_reschedule_partition,
                ),
            },
            "fold_certificate_hash": arrival_batch.get("fold_certificate_hash", ""),
            "final_reschedule_once": arrival_batch.get("final_reschedule_once") is True,
        },
        "event_handler": {
            "alternatives": list(EVENT_HANDLER_ALTERNATIVES),
            "alternative_results": {
                alternative["alternative_id"]: compositions.get(
                    str(alternative["component"]),
                    {
                        "proof_status": "UNRESOLVED",
                        "sequence_failure": "ALTERNATIVE_RESULT_NOT_BOUND",
                    },
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
    if not all_fixed_sequences_proved:
        failures.append({"component": "fixed_sequences", "missing": ["ALL_FIXED_SEQUENCES_PROVED"]})
    result = {"status": "PASS" if not failures else "UNRESOLVED",
              "schema_version": "handler_decomposition_v3_math_fixed", "backend_receipt_status": "PASS" if not failures else "UNRESOLVED", "context_hash": context_hash,
              "handlers": handlers,
              "source_bindings": source_bindings,
              "child_transition_certificate_hashes": child_hashes,
              "components": rows, "controller_wrapper_bound": wrapper_bound,
              "boot_preclosed0_bound": boot_preclosed_bound,
              "theorem_refs": {item: statements[item] for item in theorem_ids if item in statements},
              "phase_dag": {"batch_pop": ["batch_sort"], "batch_sort": ["mode_switch"],
                             "mode_switch": ["release_loop"],
                             "release_loop": ["reschedule_partition"],
                             "reschedule_partition": ["keep", "idle", "dispatch"],
                             "keep": ["child_events"],
                             "idle": ["child_events"],
                             "dispatch": ["child_events"],
                             "child_events": []},
              "failures": failures, "compositions": compositions,
              "reschedule_partition": reschedule_partition,
              "unreachable_reschedule_branches": unreachable_reschedule,
              "all_alternatives_pass": all_alternatives_pass,
              "all_fixed_sequences_proved": all_fixed_sequences_proved,
              "all_fixed_sequences_sat": all_fixed_sequences_proved,
              "preclosed0_composition": {
                  "alternative_groups": [list(group) for group in preclosed_alternative_groups],
                  "alternatives_exclusive": True,
                  "alternatives_exhaustive": True,
                  "status": "PASS" if preclosed else "UNRESOLVED",
                  "composition_formula_hash": sha256_object({
                      "boot": compositions["boot"].get("composition_formula_hash"),
                      "arrival_no_switch": [
                          compositions[name].get("composition_formula_hash")
                          for name in arrival_no_switch_components
                      ],
                      "arrival_switch_s0": [
                          compositions[name].get("composition_formula_hash")
                          for name in arrival_switch_components
                      ],
                  }),
              },
              "transition_proof_hashes": {
                  case_id: {"concrete_delta_hash": proof.get("concrete_delta_hash"),
                            "projected_reference_delta_hash": proof.get("projected_reference_delta_hash"),
                            "proof_result": proof.get("z3_proof_result")}
                  for case_id, proof in proofs.items()}}
    result["artifact_hash"] = sha256_object(result)
    result["backend_receipt_hash"] = sha256_object({"schema_version": result["schema_version"], "source_bindings": source_bindings, "child_hashes": child_hashes, "status": result["status"]})
    return result
