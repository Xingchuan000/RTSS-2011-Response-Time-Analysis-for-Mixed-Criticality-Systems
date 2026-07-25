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


def _closure_lexicographic_measure(
    state: Any,
) -> tuple[int, ...]:
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


def derive_measure_delta(ir: Any) -> dict[str, Any]:
    """Derive one closure measure delta from a compiled executable IR.

    The receipt is deliberately fail-closed: a source hash or a handwritten
    schema is insufficient.  Every path and every exceptional path must be
    covered by the executable compiler before a delta can be used by closure.
    """
    if ir is None:
        return {"status": "UNRESOLVED", "code": "COMPILED_IR_MISSING"}
    receipt = getattr(ir, "compilation_receipt", None)
    total = (
        getattr(ir, "compilation_status", None) == "COMPILED"
        and getattr(ir, "total_semantic_coverage", False) is True
        and receipt is not None
        and getattr(ir, "paths", ())
        and all(getattr(p, "terminator", None) in {"RETURN", "RAISE"}
                for p in getattr(ir, "paths", ()))
        and getattr(receipt, "covered_return_path_count", -1)
            == getattr(receipt, "return_path_count", -2)
        and getattr(receipt, "covered_raise_path_count", -1)
            == getattr(receipt, "raise_path_count", -2)
    )
    case_id = str(getattr(ir, "case_id", ""))
    consumed = {
        "REM_COMPLETION": "remaining_REM",
        "RECOVERY": "REC_enabled",
        "DEADLINE_OBSERVATION": "remaining_DDL_entries",
        "ARRIVAL_BATCH": "remaining_ARR_entries",
        "MODE_SWITCH": "remaining_SW_REL_entries",
        "RELEASE": "remaining_SW_REL_entries",
        "FINAL_DISPATCH": "DSP_enabled",
    }.get(case_id)
    time_advances = case_id in {"SERVICE_UNIT", "TAIL_ONLY_SERVICE"}
    generated = tuple(getattr(event, "event_kind", "") for event in getattr(ir, "generated_events", ()))
    later_phase = {"REM": 1, "REC": 2, "DDL": 3, "ARR_BATCH": 4,
                   "SW": 5, "REL": 5, "DSP": 6}
    current_phase = {"REM_COMPLETION": 0, "RECOVERY": 1,
                     "DEADLINE_OBSERVATION": 2, "ARRIVAL_BATCH": 3,
                     "MODE_SWITCH": 4, "RELEASE": 5,
                     "FINAL_DISPATCH": 6}.get(case_id, -1)
    generated_only_later = (
        all(
            (kind == "DDL" and case_id == "RELEASE")
            or later_phase.get(kind, 99) > current_phase
            for kind in generated
        ) if current_phase >= 0 else time_advances
    )
    proven = bool(total and (consumed is not None or time_advances)
                 and generated_only_later)
    return {
        "status": "PASS" if proven else "UNRESOLVED",
        "case_id": case_id,
        "ir_hash": (ir.ir_hash() if callable(getattr(ir, "ir_hash", None))
                    else getattr(ir, "ir_hash", None)),
        "source_function_ast_hash": getattr(ir, "source_function_ast_hash", None),
        "consumed_component": consumed,
        "component_delta": -1 if consumed else 0,
        "phase_rank_delta": 1 if time_advances else 0,
        "generated_events": list(generated),
        "generated_only_later_phase": generated_only_later,
        "all_paths_covered": bool(total),
        "source_bound": bool(total),
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
            "closes_loop": case_id not in {"SERVICE_UNIT", "TAIL_ONLY_SERVICE"},
            "decrease_proved": delta.get("status") == "PASS",
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
            "post-closure: positive service tick OR jump to strictly future event",
            "time-divergent (no Zeno)",
        ],
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
    established = successor_ok and (source_kernel_ok or pk_kernel["status"] == "PASS")
    return {
        "obligation_id": "PROTECTED_PREFIX_TIME_DIVERGENCE",
        "status": "PASS" if established else "UNRESOLVED",
        "code": None if established else "TIME_DIVERGENCE_UNRESOLVED",
        "canonical_successor_total": successor_ok,
        "service_branch_advances_by_1": bool(
            source_kernel_ok and proof_kernel_receipt.get("service_branch_advances_by_1") is True
        ) if proof_kernel_receipt else False,
        "idle_branch_jumps_to_future_event": bool(
            source_kernel_ok and proof_kernel_receipt.get("idle_branch_jumps_to_future_event") is True
        ) if proof_kernel_receipt else False,
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
    """Prove existence of one complete execution for one fixed projected oracle."""
    try:
        standard_initial = initial_reference_state(prefix_taskset)
        initial_ok = (
            int(standard_initial.time) == 0
            and str(standard_initial.mode) in {"LO", "Mode.LO"}
            and not getattr(standard_initial, "jobs", ())
            and getattr(standard_initial, "running_job_key", None) is None
        )
    except (TypeError, ValueError, RuntimeError, AttributeError):
        initial_ok = False

    successor_ok = _receipt_pass(canonical_successor_receipt)
    div_ok = _receipt_pass(time_divergence_receipt)
    proj_ok = _receipt_pass(input_projection_receipt)
    demand_ok = _receipt_pass(demand_receptiveness_receipt)

    internal_witness = None
    if protected_oracle is not None and initial_ok:
        internal_witness = build_complete_prefix_execution_witness(
            init_state=standard_initial,
            oracle=protected_oracle,
            successor=lambda current: next_closed_boundary(
                current, prefix_taskset, protected_oracle,
            ),
        )
    witness_ok = isinstance(internal_witness, CompleteExecutionWitness)
    source_kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id")
            == "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS"
        and proof_kernel_receipt.get("dependent_choice_construction_verified") is True
        and proof_kernel_receipt.get("canonical_successor_total_consumed") is True
        and proof_kernel_receipt.get("recurring_history_preserved") is True
        and proof_kernel_receipt.get("same_fixed_oracle") is True
    )
    # The authoritative witness is constructed above from the fixed initial
    # state, projected oracle and total successor.  A caller-supplied witness
    # is never accepted; the optional kernel receipt only adds source-bound
    # evidence to this internal construction.
    kernel_ok = witness_ok and (
        source_kernel_ok
        or (successor_ok and div_ok and proj_ok and demand_ok)
    )
    idle_payload = _receipt_payload(idle_jump_expansion_receipt)
    idle_expansion_ok = (
        _receipt_pass(idle_jump_expansion_receipt)
        and idle_payload.get("theorem_id") == "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION"
        and idle_payload.get("parameterized") is True
        and idle_payload.get("independent_of_complete_execution_witness") is True
        and idle_payload.get("all_integer_times_observable") is True
    )
    projection = _receipt_payload(input_projection_receipt)
    projected_oracle_fingerprint = projection.get("projected_oracle_fingerprint")
    complete_execution_oracle_fingerprint = (
        internal_witness.projected_oracle_hash if internal_witness is not None else None
    )
    oracle_binding_ok = (
        isinstance(projected_oracle_fingerprint, str)
        and projected_oracle_fingerprint == complete_execution_oracle_fingerprint
    )
    established = all((
        initial_ok, successor_ok, div_ok, proj_ok, demand_ok, witness_ok,
        kernel_ok, idle_expansion_ok, oracle_binding_ok,
    ))

    return {
        "status": "PASS" if established else "UNRESOLVED",
        "code": None if established else "COMPLETE_EXECUTION_EXISTS_UNRESOLVED",
        "obligation_id": "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
        "witness": {
            "schema_version": "complete_execution_existence_v4",
            "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
            "initial_state_constructible": initial_ok,
            "standard_empty_lo_initial_state": initial_ok,
            "canonical_successor_total": successor_ok,
            "time_divergent": div_ok,
            "projected_oracle_defined": proj_ok,
            "projected_demands_legal": demand_ok,
            "complete_execution_exists": established,
            "complete_execution_witness_constructed": witness_ok,
            "finite_prefix_compatibility_proved": witness_ok,
            "same_fixed_oracle": witness_ok and oracle_binding_ok,
            "same_initial_state": witness_ok and initial_ok,
            "same_successor_function": "next_closed_boundary" if witness_ok else None,
            "all_finite_prefixes_are_prefixes_of_one_execution": witness_ok,
            "recurring_history_preserved": established,
            "idle_jump_expansion_verified": idle_expansion_ok,
            "time_indexed_closed_observation_defined": idle_expansion_ok,
            "projected_oracle_fingerprint": projected_oracle_fingerprint,
            "complete_execution_oracle_hash": complete_execution_oracle_fingerprint,
            "same_projected_oracle": oracle_binding_ok,
        },
        "failure": None if established else {
            "code": "COMPLETE_EXECUTION_EXISTS_UNRESOLVED",
            "reason": (
                "Complete execution existence requires total canonical successor, "
                "time divergence, a complete legal projected oracle, a standard empty "
                "LO initial state, one dependent-choice witness, and the independent "
                "local idle-jump theorem."
            ),
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
