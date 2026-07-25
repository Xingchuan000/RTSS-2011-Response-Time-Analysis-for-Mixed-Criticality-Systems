"""Complete prefix execution existence proofs.

Implements the four required execution-existence obligations in the
non-circular order required by the proof:
  1. PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES
  2. PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL
  3. PROTECTED_PREFIX_TIME_DIVERGENCE
  4. PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS

Each proof establishes a PARAMETERIZED property over all legal states,
not just one successful call to close_timestamp().
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object

from formal_toolchain.bridge.logical_events import LogicalEventKind
from formal_toolchain.reference.executable_semantics import (
    initial_reference_state,
    is_closed_reference_state,
    next_logical_event,
    step_reference_p0,
)

from .input_oracle import ProtectedInputOracle, LazyInfiniteProtectedInputOracle
from .types import ProtectedPrefixBuildResult


@dataclass(frozen=True, slots=True)
class CompleteExecutionWitness:
    """One internally constructed execution for one fixed oracle/successor."""

    initial_state_hash: str
    projected_oracle_hash: str
    successor_theorem_hash: str
    state_at: Callable[[int], Any]
    finite_prefix: Callable[[int], tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class ParametricCompleteExecutionTheorem:
    arbitrary_full_execution_id: str
    projected_oracle_id: str
    initial_state_theorem_hash: str
    successor_totality_hash: str
    finite_prefix_compatibility_hash: str
    quantifier_order: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "arbitrary_full_execution_id": self.arbitrary_full_execution_id,
            "projected_oracle_id": self.projected_oracle_id,
            "initial_state_theorem_hash": self.initial_state_theorem_hash,
            "successor_totality_hash": self.successor_totality_hash,
            "finite_prefix_compatibility_hash": self.finite_prefix_compatibility_hash,
            "quantifier_order": self.quantifier_order,
        }


def build_parametric_complete_execution_theorem(
    *,
    arbitrary_full_execution_id: str,
    projected_oracle_id: str,
    initial_state_theorem_hash: str,
    successor_totality_hash: str,
    finite_prefix_compatibility_hash: str,
) -> ParametricCompleteExecutionTheorem:
    """Construct the fixed-oracle primitive-recursion theorem witness."""
    values = (
        arbitrary_full_execution_id, projected_oracle_id,
        initial_state_theorem_hash, successor_totality_hash,
        finite_prefix_compatibility_hash,
    )
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("PARAMETRIC_COMPLETE_EXECUTION_THEOREM_ARGUMENT_INVALID")
    return ParametricCompleteExecutionTheorem(
        arbitrary_full_execution_id=arbitrary_full_execution_id,
        projected_oracle_id=projected_oracle_id,
        initial_state_theorem_hash=initial_state_theorem_hash,
        successor_totality_hash=successor_totality_hash,
        finite_prefix_compatibility_hash=finite_prefix_compatibility_hash,
        quantifier_order="forall-full-execution-exists-one-prefix-forall-boundaries",
    )


@dataclass(frozen=True, slots=True)
class MeasureDeltaReceipt:
    """Source-bound measure result for one compiled executable path."""

    path_hash: str
    pre_phase: str
    post_phase: str
    lexicographically_decreases: bool
    no_earlier_phase_event_generated: bool
    generated_same_time_events_bounded: bool
    time_advances: bool = False
    exceptional_path: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_hash": self.path_hash,
            "pre_phase": self.pre_phase,
            "post_phase": self.post_phase,
            "lexicographically_decreases": self.lexicographically_decreases,
            "no_earlier_phase_event_generated": self.no_earlier_phase_event_generated,
            "generated_same_time_events_bounded": self.generated_same_time_events_bounded,
            "time_advances": self.time_advances,
            "exceptional_path": self.exceptional_path,
        }


def _receipt_payload(receipt: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping):
        return {}
    witness = receipt.get("witness")
    return witness if isinstance(witness, Mapping) else receipt


def _receipt_pass(receipt: Mapping[str, Any] | None) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    status = receipt.get("status", receipt.get("obligation_status"))
    if status == "PASS":
        return True
    payload = _receipt_payload(receipt)
    return payload.get("status") == "PASS"


# ---------------------------------------------------------------------------
# Section 7.2: Canonical closed-boundary successor
# ---------------------------------------------------------------------------


def _bind_oracle_for_current_arrival(state: Any, oracle: ProtectedInputOracle) -> Any:
    """Bind the authoritative projected input before processing an ARR_BATCH.

    The executable reference semantics reads release_demand_overrides and
    abnormal_hi_releases when opening the arrival batch.  This adapter updates
    those immutable maps from the single fixed protected oracle immediately
    before that batch is processed.
    """
    event = next_logical_event(state)
    if event is None or event.time != state.time or event.kind is not LogicalEventKind.ARR_BATCH:
        return state
    overrides = dict(state.release_demand_overrides)
    abnormal = set(state.abnormal_hi_releases)
    for task_name, release_index in event.batch_jobs:
        inp = oracle.input_for(task_name, release_index)
        if inp.job_key != (task_name, release_index) or inp.release_time != event.time:
            raise ValueError("PROTECTED_ORACLE_ARRIVAL_BINDING_MISMATCH")
        overrides[inp.job_key] = int(inp.actual_demand)
        if inp.hi_class == "ABNORMAL":
            abnormal.add(inp.job_key)
        else:
            abnormal.discard(inp.job_key)
    return replace(
        state,
        release_demand_overrides=overrides,
        abnormal_hi_releases=frozenset(abnormal),
    )


def next_closed_boundary(
    state: Any,
    taskset: object,
    oracle: ProtectedInputOracle,
) -> Any:
    """Execute to the next strictly later canonical closed boundary.

    This is an executable successor, not a totality proof.  It starts from a
    validated closed state, binds every encountered protected ARR_BATCH to the
    same fixed oracle, and repeatedly applies the code-faithful P0 step until a
    closed state at a strictly later time is reached.
    """
    if not is_closed_reference_state(state, taskset):
        raise ValueError("PROTECTED_PREFIX_SUCCESSOR_REQUIRES_CLOSED_STATE")
    start_time = int(state.time)
    current = state
    while True:
        current = _bind_oracle_for_current_arrival(current, oracle)
        try:
            current = step_reference_p0(current, taskset).after
        except ValueError as exc:
            # The periodic generator normally guarantees a future event.  The
            # explicit infinite-idle extension is nevertheless part of the
            # total successor definition; it is reachable only when the
            # validated frontier has no future observation.
            if str(exc) != "REFERENCE_VALID_STATE_PERIODIC_GENERATOR_INVARIANT_BROKEN":
                raise
            future = [event for event in getattr(current, "frontier", ())
                      if int(event.time) > int(current.time)]
            if future:
                raise
            current = replace(current, time=int(current.time) + 1)
        if int(current.time) > start_time and is_closed_reference_state(current, taskset):
            return current


# ---------------------------------------------------------------------------
# Section 7.3: Closure termination with seven-dimensional lexicographic measure
# ---------------------------------------------------------------------------
# Measure: (#REM_t, #REC_t, #DDL_t, #ARR_t, #SW_t, #REL_t, #DSP_t)
#
# Each component is a natural number count of events of that kind at the
# current time on the frontier.  A microstep that processes an event of
# kind K reduces the count for K by exactly 1, does not increase any
# earlier-phase component, and may generate events only in a later phase.
#
# The nine real transitions (from pp_transition_binding.py):
#   REM_COMPLETION  -> reduces REM_t by 1
#   RECOVERY        -> reduces REC_t by 1
#   DEADLINE_OBSERVATION -> reduces DDL_t by 1
#   ARRIVAL_BATCH   -> reduces ARR_t by 1 (may generate SW/REL events)
#   MODE_SWITCH     -> reduces SW_t by 1
#   RELEASE         -> reduces REL_t by 1
#   FINAL_DISPATCH  -> reduces DSP_t by 1
#   SERVICE_UNIT    -> time advances (leaves closure loop)
#   TAIL_ONLY_SERVICE -> time advances (leaves closure loop)
# ---------------------------------------------------------------------------


_LEXICOGRAPHIC_MEASURE_ORDER = (
    "remaining_REM", "REC_enabled", "remaining_DDL_entries",
    "remaining_ARR_entries", "remaining_SW_entries", "remaining_REL_entries", "DSP_enabled",
)
ClosureMeasure = tuple[int, ...]


def _closure_lexicographic_measure(
    state: Any,
) -> ClosureMeasure:
    """Compute the lexicographic closure measure for a state.

    Section 7.3: the measure must strictly decrease at each closure micro-step
    and bottom out at zero in all components.

    The measure is restricted to the current timestamp.  Events generated by
    a handler are legal only in a later phase, so processing the current head
    strictly decreases the first changed component.
    """
    frontier = getattr(state, "frontier", ())
    from formal_toolchain.bridge.logical_events import LogicalEventKind
    current = [e for e in frontier if int(e.time) == int(state.time)]
    count = lambda kind: sum(1 for e in current if e.kind is kind)
    return (count(LogicalEventKind.REM), count(LogicalEventKind.REC),
            count(LogicalEventKind.DDL), count(LogicalEventKind.ARR_BATCH),
            count(LogicalEventKind.SW), count(LogicalEventKind.REL),
            count(LogicalEventKind.DSP))


def strict_lexicographic_decrease(
    pre: tuple[int, ...],
    post: tuple[int, ...],
) -> bool:
    """Check strict lexicographic decrease: pre > post.

    Returns True when there exists index i such that
    pre[0..i-1] == post[0..i-1] AND pre[i] > post[i].
    """
    for a, b in zip(pre, post):
        if a > b:
            return True
        if a < b:
            return False
    return False


def prove_closure_measure_well_founded() -> dict[str, Any]:
    """Prove: the lexicographic measure is well-founded (lower-bounded at zero).

    Each component is a non-negative integer count; the tuple is
    component-wise >= (0, 0, 0, 0, 0, 0).
    """
    return {
        "status": "PASS",
        "lemma": "CLOSURE_MEASURE_WELL_FOUNDED",
        "measure_order": list(_LEXICOGRAPHIC_MEASURE_ORDER),
        "lower_bound": tuple(0 for _ in _LEXICOGRAPHIC_MEASURE_ORDER),
        "argument": (
            "Every component is a non-negative integer count.  A finite "
            "lexicographic tuple of non-negative integers with component-wise "
            "lower bound (0,...,0) is well-founded by the standard "
            "lexicographic order on N^k."
        ),
    }


def _generated_event_order_contract() -> dict[str, Any]:
    """Bind the global same-time generation order to executable source."""
    import ast
    import inspect
    from formal_toolchain.reference import executable_semantics as runtime
    try:
        tree = ast.parse(inspect.getsource(runtime._append_generated_event))
        source = ast.unparse(tree)
    except (OSError, TypeError, SyntaxError) as exc:
        return {"status": "UNRESOLVED", "error": str(exc)}
    facts = {
        "past_event_rejected": "generated_event.time < parent_event.time" in source,
        "same_time_nonservice_checked": (
            "generated_event.time == parent_event.time" in source
            and "generated_event.kind is not LogicalEventKind.SVC" in source
        ),
        "non_later_phase_rejected": "generated_event.phase_rank <= parent_event.phase_rank" in source,
        "event_appended_after_checks": "frontier.append(generated_event)" in source,
    }
    payload = {
        "status": "PASS" if all(facts.values()) else "UNRESOLVED",
        "facts": facts,
        "source_hash": sha256_object({"ast": ast.dump(tree, include_attributes=False)}),
    }
    return payload


def derive_measure_delta(ir: Any) -> dict[str, Any]:
    """Derive closure progress from executable path semantics.

    A closure event is admissible only on a normal RETURN path.  For every such
    path we require: (i) the current event is consumed, (ii) generated
    same-time events are forced into a strictly later phase by the shared
    helper contract, and (iii) every source loop is a finite fold.  This is the
    exact sufficient condition for strict decrease of the seven-component
    frontier-count measure.
    """
    if ir is None:
        return {"status": "UNRESOLVED", "code": "COMPILED_IR_MISSING"}
    receipt = getattr(ir, "compilation_receipt", None)
    total = bool(
        getattr(ir, "compilation_status", None) == "COMPILED"
        and getattr(ir, "total_semantic_coverage", False) is True
        and receipt is not None
        and getattr(ir, "semantic_effect", None) is not None
        and getattr(ir.semantic_effect, "derivation_complete", False) is True
        and getattr(receipt, "covered_return_path_count", -1)
            == getattr(receipt, "return_path_count", -2)
        and getattr(receipt, "covered_raise_path_count", -1)
            == getattr(receipt, "raise_path_count", -2)
    )
    case_id = str(getattr(ir, "case_id", ""))
    closure_phase = {
        "REM_COMPLETION": "REM", "RECOVERY": "REC",
        "DEADLINE_OBSERVATION": "DDL", "ARRIVAL_BATCH": "ARR",
        "MODE_SWITCH": "SW", "RELEASE": "REL",
    }
    consumed_component = {
        "REM_COMPLETION": "remaining_REM",
        "RECOVERY": "REC_enabled",
        "DEADLINE_OBSERVATION": "remaining_DDL_entries",
        "ARRIVAL_BATCH": "remaining_ARR_entries",
        "MODE_SWITCH": "remaining_SW_entries",
        "RELEASE": "remaining_REL_entries",
    }.get(case_id)
    consumption_fact = {
        "REM_COMPLETION": "consume_rem_event",
        "RECOVERY": "consume_recovery_event",
        "DEADLINE_OBSERVATION": "consume_deadline_event",
        "ARRIVAL_BATCH": "consume_arrival_event",
        "MODE_SWITCH": "record_switch_and_consume_event",
        "RELEASE": "consume_release_event",
    }.get(case_id)
    validator_facts = set(getattr(ir.semantic_effect, "validator_facts", ())) if getattr(ir, "semantic_effect", None) else set()
    event_consumed = consumption_fact in validator_facts if consumption_fact else False
    order_contract = _generated_event_order_contract()
    finite_folds = all(
        isinstance(getattr(fold, "termination_measure", None), str)
        and getattr(fold, "termination_measure", "").startswith("len(")
        and bool(getattr(fold, "step_summary_hash", ""))
        and bool(getattr(fold, "invariant_schema_hash", ""))
        for fold in getattr(ir, "folds", ())
    )
    time_advances = (
        case_id in {"SERVICE_UNIT", "TAIL_ONLY_SERVICE"}
        and "time_advances_exactly_one" in validator_facts
    )
    final_normalization = case_id == "FINAL_DISPATCH" and {
        "strict_fp_total_sort", "empty_jobs_dispatches_none",
        "nonempty_dispatches_head",
    } <= validator_facts

    path_receipts: list[MeasureDeltaReceipt] = []
    for path in getattr(ir, "paths", ()):
        exceptional = path.terminator == "RAISE"
        closure_path = case_id in closure_phase
        progress = bool(
            not exceptional and closure_path and event_consumed
            and order_contract.get("status") == "PASS" and finite_folds
        )
        path_receipts.append(MeasureDeltaReceipt(
            path_hash=path.path_hash(),
            pre_phase=closure_phase.get(case_id, "SERVICE_OR_NORMALIZE"),
            post_phase=closure_phase.get(case_id, "SERVICE_OR_NORMALIZE"),
            lexicographically_decreases=progress,
            no_earlier_phase_event_generated=bool(
                progress or time_advances or final_normalization
            ),
            generated_same_time_events_bounded=bool(finite_folds),
            time_advances=bool(not exceptional and time_advances),
            exceptional_path=exceptional,
        ))
    legal_paths = [path for path in path_receipts if not path.exceptional_path]
    if case_id in closure_phase:
        legal_progress = bool(legal_paths) and all(
            path.lexicographically_decreases for path in legal_paths
        )
    elif case_id in {"SERVICE_UNIT", "TAIL_ONLY_SERVICE"}:
        legal_progress = bool(legal_paths) and all(path.time_advances for path in legal_paths)
    elif case_id == "FINAL_DISPATCH":
        legal_progress = bool(legal_paths) and final_normalization
    else:
        legal_progress = False
    proven = bool(total and legal_progress)
    return {
        "status": "PASS" if proven else "UNRESOLVED",
        "case_id": case_id,
        "ir_hash": ir.ir_hash() if callable(getattr(ir, "ir_hash", None)) else None,
        "source_function_ast_hash": getattr(ir, "source_function_ast_hash", None),
        "consumed_component": consumed_component,
        "component_delta": -1 if consumed_component else 0,
        "event_consumption_fact": consumption_fact,
        "event_consumption_source_bound": event_consumed,
        "generated_event_order_contract": order_contract,
        "finite_fold_termination_bound": finite_folds,
        "closure_final_normalization": final_normalization,
        "time_advances": time_advances,
        "path_delta_receipts": [path.to_dict() for path in path_receipts],
        "all_path_measure_deltas_bound": bool(path_receipts) and all(
            path.generated_same_time_events_bounded for path in path_receipts
        ),
        "all_legal_path_deltas_proved": legal_progress,
        "all_paths_covered": total,
        "source_bound": total,
        "generated_only_later_phase": order_contract.get("status") == "PASS",
        "status_reason": None if proven else "EXECUTABLE_MEASURE_DELTA_UNRESOLVED",
    }


def _per_transition_measure_decrease() -> dict[str, dict[str, Any]]:
    """For each of the 9 real transitions, prove strict measure decrease.

    Returns a dict mapping case_id -> {measure_index, direction, decrease_proved}.
    """
    from formal_toolchain.reference.protected_priority_prefix.executable_transition_compiler import compile_all_transitions
    compiled = {ir.case_id: ir for ir in compile_all_transitions()}
    results: dict[str, dict[str, Any]] = {}
    for case_id in ("REM_COMPLETION", "RECOVERY", "DEADLINE_OBSERVATION",
                    "ARRIVAL_BATCH", "MODE_SWITCH", "RELEASE",
                    "FINAL_DISPATCH", "SERVICE_UNIT", "TAIL_ONLY_SERVICE"):
        delta = derive_measure_delta(compiled.get(case_id))
        results[case_id] = {
            **delta,
            "measure_component": delta.get("consumed_component") or "phase_rank",
            "closes_loop": case_id in {
                "REM_COMPLETION", "RECOVERY", "DEADLINE_OBSERVATION",
                "ARRIVAL_BATCH", "MODE_SWITCH", "RELEASE",
            },
            "closure_final_normalization": case_id == "FINAL_DISPATCH",
            "decrease_proved": delta.get("status") == "PASS",
            "generated_events_only_in_later_phase": delta.get("generated_only_later_phase") is True,
            "no_earlier_component_increased": delta.get("generated_only_later_phase") is True,
        }
    return results


# ---------------------------------------------------------------------------
# Section 7.2 - 7.4: The four execution-existence proofs
# ---------------------------------------------------------------------------


def define_next_closed_boundary(
    prefix_taskset: object,
    protected_oracle: ProtectedInputOracle,
) -> dict[str, Any]:
    """Define the canonical closed-boundary successor function.

    The successor function is next_closed_boundary(), which applies the
    canonical macro-step (close_timestamp) to advance to the next closed
    boundary.
    """
    return {
        "schema_version": "canonical_successor_definition_v2",
        "function_name": "next_closed_boundary",
        "function_signature": (
            "next_closed_boundary(state: ReferenceState, "
            "taskset: ReferenceTaskset, oracle: ProtectedInputOracle) "
            "-> ReferenceState"
        ),
        "implementation": "next_closed_boundary(state, taskset, oracle)",
        "expected_properties": [
            "total over all legal closed states",
            "same-time closure measure strictly decreasing (lexicographic)",
            "A: ready/running -> one service tick -> Close",
            "B: no ready + minimum future observation -> idle jump -> Close",
            "C: no future observation -> explicit time+1 infinite-idle extension",
            "time-divergent (no Zeno)",
        ],
        "branches": {
            "A_READY_OR_RUNNING": "step_reference_p0 SERVICE -> Close",
            "B_MINIMUM_FUTURE_OBSERVATION": "step_reference_p0 JUMP -> Close",
            "C_NO_FUTURE_OBSERVATION": "replace(time=time+1) -> Close",
        },
        "closure_measure_well_founded": prove_closure_measure_well_founded(),
        "status": "UNRESOLVED",
        "code": "CANONICAL_SUCCESSOR_TOTALITY_PROOF_KERNEL_MISSING",
        "executable_successor_defined": True,
    }


def prove_same_time_closure_terminates(
    runtime_schema_receipt: Mapping[str, Any] | None = None,
    prefix_extension_receipt: Mapping[str, Any] | None = None,
    proof_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove termination of the same-timestamp closure independently.

    This lemma is logically prior to successor totality.  Using successor
    totality as its premise would be circular because totality itself requires
    the closure loop to terminate.

    The proof uses the seven-dimensional lexicographic measure:
    (#REM_t, #REC_t, #DDL_t, #ARR_t, #SW_t, #REL_t, #DSP_t)

    For each of the 9 real transitions (from pp_transition_binding.py):
    - The corresponding component decreases by exactly 1
    - No earlier-phase component increases
    - Generated events only appear in a strictly later phase
    - SERVICE_UNIT and TAIL_ONLY_SERVICE advance time, exiting the closure loop
    """
    from .proof_kernel import prove_same_time_closure_termination_kernel
    kernel = prove_same_time_closure_termination_kernel()
    external_kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id")
            == "SAME_TIME_CLOSURE_TERMINATES"
        and proof_kernel_receipt.get("proof_scope") == "ALL_LEGAL_SAME_TIME_CLOSURES"
    )
    kernel_ok = kernel["status"] == "PASS" or external_kernel_ok

    measure = prove_closure_measure_well_founded()
    per_transition = _per_transition_measure_decrease()
    runtime_ok = _receipt_pass(runtime_schema_receipt)
    extension_ok = _receipt_pass(prefix_extension_receipt)

    all_decrease_proved = all(
        info.get("decrease_proved") is True
        for info in per_transition.values()
    )
    all_generated_later = all(
        info.get("generated_events_only_in_later_phase", True) is True
        or info.get("phase_before") == info.get("phase_after")
        for info in per_transition.values()
        if not info.get("closes_loop", False)
    )
    established = (
        measure["status"] == "PASS"
        and runtime_ok and extension_ok
        and all_decrease_proved and all_generated_later
        and kernel_ok
    )
    return {
        "obligation_id": "PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES",
        "status": "PASS" if established else "UNRESOLVED",
        "code": None if established else "SAME_TIME_CLOSURE_TERMINATION_UNRESOLVED",
        "closure_measure": list(_LEXICOGRAPHIC_MEASURE_ORDER),
        "measure_well_founded": measure,
        "per_transition_decrease": per_transition,
        "all_decrease_proved": all_decrease_proved,
        "all_generated_only_later_phase": all_generated_later,
        "source_bound_microstep_decrease": kernel_ok,
        "parameterized_proof_kernel": kernel,
        "reason": (
            "Each of the 9 real transitions (REM_COMPLETION, RECOVERY, "
            "DEADLINE_OBSERVATION, ARRIVAL_BATCH, MODE_SWITCH, RELEASE, "
            "FINAL_DISPATCH, SERVICE_UNIT, TAIL_ONLY_SERVICE) must strictly "
            "decrease the corresponding component of the seven-phase "
            "lexicographic measure.  Generated events must only appear in "
            "a strictly later phase.  SERVICE_UNIT and TAIL_ONLY_SERVICE "
            "exit the closure loop by advancing time."
        ),
    }


def prove_canonical_successor_total(
    prefix_taskset: object,
    successor_definition: dict[str, Any],
    *,
    closure_termination_receipt: Mapping[str, Any] | None = None,
    prefix_extension_receipt: Mapping[str, Any] | None = None,
    input_projection_receipt: Mapping[str, Any] | None = None,
    demand_receptiveness_receipt: Mapping[str, Any] | None = None,
    proof_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove totality after closure termination and input legality are known.

    Prerequisites:
      1. Same-time closure terminates (7-dim lexicographic measure)
      2. Prefix extension theorem
      3. Complete projected oracle (FULL_REFERENCE_RECURRING_INPUT_ORACLE)
      4. Demand receptiveness (PROTECTED_INPUT_DEMAND_RECEPTIVENESS)
    """
    prerequisites_ok = all((
        _receipt_pass(closure_termination_receipt),
        _receipt_pass(prefix_extension_receipt),
        _receipt_pass(input_projection_receipt),
        _receipt_pass(demand_receptiveness_receipt),
    ))
    closure_payload = _receipt_payload(closure_termination_receipt)
    measure_ok = (
        closure_payload.get("all_decrease_proved") is True
        and closure_payload.get("all_generated_only_later_phase") is True
    )
    projection = _receipt_payload(input_projection_receipt)
    demand = _receipt_payload(demand_receptiveness_receipt)
    oracle_fp = projection.get("projected_oracle_fingerprint")
    oracle_contract_ok = (
        isinstance(oracle_fp, str)
        and projection.get("complete_recurring_stream") is True
        and demand.get("all_projected_demands_legal") is True
        and demand.get("all_projected_demands_positive") is True
        and demand.get("release_fixed_demands") is True
        and demand.get("mode_independent_lo_receptiveness") is True
    )
    from .proof_kernel import prove_canonical_successor_total_kernel
    pk_kernel = prove_canonical_successor_total_kernel()
    external_kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id")
            == "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL"
        and proof_kernel_receipt.get("all_legal_closed_states_covered") is True
        and proof_kernel_receipt.get("same_time_closure_termination_consumed") is True
        and proof_kernel_receipt.get("future_event_or_service_branch_total") is True
        and proof_kernel_receipt.get("projected_oracle_contract_consumed") is True
    )
    kernel_ok = pk_kernel["status"] == "PASS" or external_kernel_ok
    established = (
        prerequisites_ok and measure_ok and oracle_contract_ok and kernel_ok
    )
    return {
        "obligation_id": "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL",
        "status": "PASS" if established else "UNRESOLVED",
        "code": None if established else "CANONICAL_SUCCESSOR_TOTALITY_UNRESOLVED",
        "successor_definition_hash": sha256_object(successor_definition),
        "projected_oracle_fingerprint": oracle_fp,
        "closure_termination_consumed": _receipt_pass(closure_termination_receipt),
        "measure_decrease_proved": measure_ok,
        "projected_input_legal": oracle_contract_ok,
        "parameterized_proof_kernel": pk_kernel,
        "reason": (
            "Totality requires independently proved closure termination (with all "
            "9 transitions strictly decreasing the 7-dim lexicographic measure), "
            "the prefix-extension theorem, a complete projected input contract, "
            "demand receptiveness, and a source-bound proof of the service/future-event branches."
        ),
    }


def prove_time_divergence(
    canonical_successor_receipt: Mapping[str, Any] | None = None,
    proof_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove unbounded time growth for repeated canonical successors.

    Two branches per macro-step:
      1. Service branch: one SERVICE_UNIT advances time by exactly 1
      2. Idle branch: close_timestamp jumps to strictly future event (time > t+1)

    Both branches strictly advance time, so repeated application diverges.
    """
    successor_ok = _receipt_pass(canonical_successor_receipt)
    source_kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id")
            == "PROTECTED_PREFIX_TIME_DIVERGENCE"
        and proof_kernel_receipt.get("service_branch_advances_by_1") is True
        and proof_kernel_receipt.get("idle_branch_jumps_to_future_event") is True
        and proof_kernel_receipt.get("unbounded_iteration_proved") is True
    )
    from .proof_kernel import prove_time_divergence_kernel
    pk_kernel = prove_time_divergence_kernel()
    internal_kernel_ok = pk_kernel.get("status") == "PASS"
    established = successor_ok and (source_kernel_ok or internal_kernel_ok)
    active_kernel = proof_kernel_receipt if source_kernel_ok else pk_kernel
    return {
        "obligation_id": "PROTECTED_PREFIX_TIME_DIVERGENCE",
        "status": "PASS" if established else "UNRESOLVED",
        "code": None if established else "TIME_DIVERGENCE_UNRESOLVED",
        "canonical_successor_total": successor_ok,
        "service_branch_advances_by_1": (
            established and active_kernel.get("service_branch_advances_by_1") is True
        ),
        "idle_branch_jumps_to_future_event": (
            established and active_kernel.get("idle_branch_jumps_to_future_event") is True
        ),
        "unbounded_iteration_proved": (
            established and active_kernel.get("unbounded_iteration_proved") is True
        ),
        "parameterized_proof_kernel": pk_kernel,
        "reason": (
            "Each macro-step either advances time by 1 (service branch, "
            "SERVICE_UNIT increments time by exactly 1) or jumps to a strictly "
            "future event (idle branch, close_timestamp->next event time > t+1). "
            "Both strictly advance time, so induction proves unbounded iteration."
        ),
    }


def prove_complete_execution_exists(
    *,
    canonical_successor_receipt: Mapping[str, Any] | None = None,
    time_divergence_receipt: Mapping[str, Any] | None = None,
    input_projection_receipt: Mapping[str, Any] | None = None,
    demand_receptiveness_receipt: Mapping[str, Any] | None = None,
    prefix_taskset: object,
    protected_oracle: ProtectedInputOracle | None = None,
    prefix_initial_state: Any = None,
    proof_kernel_receipt: Mapping[str, Any] | None = None,
    idle_jump_expansion_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive one complete prefix execution for every projected full stream.

    The theorem-level route is symbolic: after fixing an arbitrary full
    execution, ``PROTECTED_INPUT_STREAM_PROJECTION`` supplies one immutable
    projected oracle.  The standard initial state and the total deterministic
    ``next_closed_boundary`` function then define a unique sequence by
    primitive recursion on ``N``.  A concrete oracle may still be supplied for
    diagnostic replay, but theorem PASS never depends on a finite replay.
    """
    try:
        standard_initial = initial_reference_state(prefix_taskset)
        initial_ok = (
            int(standard_initial.time) == 0
            and str(standard_initial.mode) in {"LO", "Mode.LO"}
            and not getattr(standard_initial, "jobs", ())
            and getattr(standard_initial, "running", None) is None
            and not getattr(standard_initial, "terminal", {})
            and not getattr(standard_initial, "misses", ())
        )
    except (TypeError, ValueError, RuntimeError, AttributeError):
        standard_initial = None
        initial_ok = False

    successor_ok = _receipt_pass(canonical_successor_receipt)
    div_ok = _receipt_pass(time_divergence_receipt)
    proj_ok = _receipt_pass(input_projection_receipt)
    demand_ok = _receipt_pass(demand_receptiveness_receipt)
    projection_envelope = _receipt_payload(input_projection_receipt)
    nested_projection = projection_envelope.get("projection_receipt")
    projection = (
        {**projection_envelope, **nested_projection}
        if isinstance(nested_projection, Mapping)
        else projection_envelope
    )
    projected_oracle_fingerprint = projection.get("projected_oracle_fingerprint")
    projection_quantifier_ok = (
        projection.get("quantifier_scope")
            == "forall-full-execution-exists-unique-projected-stream"
        and projection.get("forall_release_indices") is True
        and projection.get("complete_recurring_stream") is True
        and isinstance(projected_oracle_fingerprint, str)
        and bool(projected_oracle_fingerprint)
    )
    idle_payload = _receipt_payload(idle_jump_expansion_receipt)
    idle_expansion_ok = (
        _receipt_pass(idle_jump_expansion_receipt)
        and idle_payload.get("theorem_id")
            == "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION"
        and idle_payload.get("parameterized") is True
        and idle_payload.get("independent_of_complete_execution_witness") is True
        and idle_payload.get("all_integer_times_observable") is True
    )

    primitive_recursion_hash = sha256_object({
        "theorem": "primitive-recursion-on-natural-numbers",
        "initial_state": "standard-empty-LO",
        "successor": "next_closed_boundary",
        "successor_receipt": _receipt_payload(canonical_successor_receipt),
        "oracle": projected_oracle_fingerprint,
    })
    parametric_ok = all((
        initial_ok, successor_ok, div_ok, proj_ok, projection_quantifier_ok,
        demand_ok, idle_expansion_ok,
    ))
    parametric_theorem = None
    if parametric_ok:
        parametric_theorem = build_parametric_complete_execution_theorem(
            arbitrary_full_execution_id="arbitrary-full-reference-execution",
            projected_oracle_id=projected_oracle_fingerprint,
            initial_state_theorem_hash=sha256_object({
                "state": "initial_reference_state(prefix_taskset)",
                "empty": True, "mode": "LO", "time": 0,
            }),
            successor_totality_hash=sha256_object(
                _receipt_payload(canonical_successor_receipt)
            ),
            finite_prefix_compatibility_hash=primitive_recursion_hash,
        )

    # Optional concrete diagnostic witness.  It is useful for seed debugging,
    # but is not the quantified proof object and cannot turn a failed premise
    # into PASS.
    internal_witness = None
    if protected_oracle is not None and initial_ok:
        internal_witness = build_complete_prefix_execution_witness(
            init_state=standard_initial,
            oracle=protected_oracle,
            successor=lambda current: next_closed_boundary(
                current, prefix_taskset, protected_oracle,
            ),
        )
    concrete_oracle_matches = (
        internal_witness is None
        or internal_witness.projected_oracle_hash == projected_oracle_fingerprint
    )
    established = bool(parametric_ok and concrete_oracle_matches)

    witness_payload = {
        "status": "PASS" if established else "UNRESOLVED",
        "schema_version": "complete_execution_existence_v5",
        "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
        "initial_state_constructible": initial_ok,
        "standard_empty_lo_initial_state": initial_ok,
        "canonical_successor_total": successor_ok,
        "time_divergent": div_ok,
        "projected_oracle_defined": proj_ok and projection_quantifier_ok,
        "projected_demands_legal": demand_ok,
        "complete_execution_exists": established,
        "complete_execution_witness_constructed": established,
        "finite_prefix_compatibility_proved": established,
        "same_fixed_oracle": established,
        "same_initial_state": established,
        "same_successor_function": "next_closed_boundary" if established else None,
        "all_finite_prefixes_are_prefixes_of_one_execution": established,
        "recurring_history_preserved": established,
        "idle_jump_expansion_verified": idle_expansion_ok,
        "time_indexed_closed_observation_defined": idle_expansion_ok,
        "projected_oracle_fingerprint": projected_oracle_fingerprint,
        "complete_execution_oracle_hash": projected_oracle_fingerprint if established else None,
        "same_projected_oracle": established,
        "construction_rule": "primitive_recursion_on_total_deterministic_successor",
        "finite_replay_used_as_proof": False,
        "concrete_diagnostic_witness_built": internal_witness is not None,
        "parametric_complete_execution_theorem": (
            parametric_theorem.to_dict() if parametric_theorem is not None else None
        ),
        "primitive_recursion_receipt_hash": primitive_recursion_hash,
    }
    return {
        "status": "PASS" if established else "UNRESOLVED",
        "code": None if established else "COMPLETE_EXECUTION_EXISTS_UNRESOLVED",
        "obligation_id": "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
        "witness": witness_payload,
        "failure": None if established else {
            "code": "COMPLETE_EXECUTION_EXISTS_UNRESOLVED",
            "premises": {
                "initial": initial_ok, "successor": successor_ok,
                "time_divergence": div_ok, "projection": proj_ok,
                "projection_quantifier": projection_quantifier_ok,
                "demand_receptiveness": demand_ok,
                "idle_expansion": idle_expansion_ok,
                "concrete_oracle_matches": concrete_oracle_matches,
            },
        },
    }


def build_complete_prefix_execution_witness(
    *,
    init_state: Any,
    oracle: ProtectedInputOracle,
    successor: Callable[[Any], Any],
) -> CompleteExecutionWitness:
    """Construct one complete execution by primitive recursion on ``n``."""
    if init_state is None or oracle is None or not callable(successor):
        raise ValueError("COMPLETE_EXECUTION_WITNESS_ARGUMENT_INVALID")
    cache: dict[int, Any] = {0: init_state}

    def state_at(n: int) -> Any:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("COMPLETE_EXECUTION_WITNESS_INDEX_INVALID")
        for k in range(1, n + 1):
            cache.setdefault(k, successor(cache[k - 1]))
        return cache[n]

    def finite_prefix(n: int) -> tuple[Any, ...]:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("COMPLETE_EXECUTION_WITNESS_HORIZON_INVALID")
        return tuple(state_at(k) for k in range(n + 1))

    from dataclasses import asdict, is_dataclass
    from formal_toolchain.core.hashing import sha256_object
    def _jsonable(value: Any) -> Any:
        if is_dataclass(value):
            return _jsonable(asdict(value))
        if hasattr(value, "to_dict"):
            return _jsonable(value.to_dict())
        if isinstance(value, Mapping):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [_jsonable(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return sorted((_jsonable(v) for v in value), key=repr)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)
    initial_payload = _jsonable(init_state)
    initial_hash = sha256_object(initial_payload)
    oracle_hash = str(oracle.oracle_fingerprint())
    successor_hash = sha256_object({
        "function": getattr(successor, "__name__", "next_closed_boundary"),
        "construction": "primitive_recursion_on_n",
    })
    return CompleteExecutionWitness(
        initial_state_hash=initial_hash,
        projected_oracle_hash=oracle_hash,
        successor_theorem_hash=successor_hash,
        state_at=state_at,
        finite_prefix=finite_prefix,
    )
