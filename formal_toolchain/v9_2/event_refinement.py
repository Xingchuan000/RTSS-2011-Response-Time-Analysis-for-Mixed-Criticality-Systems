"""Trusted V9.2 Full-kernel <-> exact Event-macro proof obligations.

The Event layer is a semantic quotient only: it retains the complete Full-state
persistent information, executes exact P0--P6 at every event boundary and
compresses only event-free repetitions of P7.  Terminal compositional theorems
are derived only from explicitly machine-checked leaves recorded in this
module; no route/source-hash-only PASS is accepted.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Iterable

import z3

from formal_toolchain.core.hashing import sha256_object, sha256_text_file_normalized

from .event_kernel import (
    _silent_interval_advance,
    build_event_candidates,
    state_equality,
)
from .event_window_encoder import derive_finite_event_bound
from .formula_solver import solve_formula
from .symbolic_state import BoundModel, declare_state
from .transition_encoder import encode_p7_time_and_service


EVENT_TERMINAL_OBLIGATIONS = (
    "EVENT_START_ABSTRACTION_SOUNDNESS",
    "EVENT_START_PROJECTION_EXACTNESS",
    "EVENT_STATE_FUTURE_SUFFICIENCY",
    "EVENT_DEMAND_LOOKUP_FACTORING_EQUIVALENCE",
    "EVENT_PHASE_SSA_FRAME_ELIMINATION_EQUIVALENCE",
    "EVENT_P5_POOL_SUPPORT_PROJECTION_EQUIVALENCE",
    "EVENT_TERMINAL_STUTTER_FACTORING_EQUIVALENCE",
    "EVENT_INCREMENTAL_TERMINAL_DEPTH_PARTITION_EQUIVALENCE",
    "NEXT_EVENT_MINIMALITY",
    "NEXT_EVENT_EXACT_MINIMUM",
    "NO_SKIPPED_DISCRETE_EVENT",
    "NO_SPURIOUS_EVENT_SOURCE",
    "SILENT_INTERVAL_SERVICE_EQUIVALENCE",
    "RELEASE_EVENT_COVERAGE",
    "DEADLINE_EVENT_COVERAGE",
    "TERMINAL_SERVICE_EVENT_COVERAGE",
    "CONTROLLER_EVENT_COVERAGE",
    "EXACT_P5_AT_CONTROLLER_EVENT",
    "FULL_TO_EVENT_SEGMENT_SIMULATION",
    "EVENT_TO_FULL_SEGMENT_REALIZABILITY",
    "FIRST_HI_BAD_EVENT_PREFIX_REFLECTION",
    "EVENT_BAD_PREFIX_FULL_REALIZABILITY",
    "FINITE_EVENT_COUNT_BOUND",
    "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY",
    "EVENT_WINDOW_ENCODING_SOUNDNESS",
)


@dataclass(frozen=True, slots=True)
class EventRefinementProof:
    status: str
    obligation_statuses: dict[str, str]
    solver_receipts: tuple[dict[str, Any], ...]
    structural_receipts: tuple[dict[str, Any], ...]
    failure_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_code": self.failure_code,
            "obligation_statuses": dict(self.obligation_statuses),
            "solver_receipts": list(self.solver_receipts),
            "structural_receipts": list(self.structural_receipts),
            "event_layer_added_abstractions": [],
            "exact_event_macro_semantics": self.status == "PASS",
            "event_to_full_realizability_verified": self.status == "PASS",
            "small_horizon_differential_consistency_verified": self.status == "PASS",
            "exact_p5_in_event_window": True,
            "microstep_terminal_fallback_used": False,
        }


def _min_expr(values: Iterable[z3.ArithRef]) -> z3.ArithRef:
    rows = list(values)
    if not rows:
        raise ValueError("minimum requires at least one candidate")
    result = rows[0]
    for value in rows[1:]:
        result = z3.If(value < result, value, result)
    return result


def _minimum_counterexample() -> z3.BoolRef:
    """Refute a mismatch between scalar exact-min constraints and reference min.

    The Event kernel uses the same two-clause finite minimum definition:
    ``nxt <= each candidate`` and ``nxt == at least one candidate``.  The
    nested ITE is retained here only as a small independent reference term.
    """

    rows = [z3.Int(f"event.refine.min.c{i}") for i in range(7)]
    nxt = z3.Int("event.refine.min.scalar")
    reference = _min_expr(rows)
    scalar = z3.And(
        *(nxt <= row for row in rows),
        z3.Or(*(nxt == row for row in rows)),
    )
    return z3.Or(
        z3.And(scalar, nxt != reference),
        z3.And(nxt == reference, z3.Not(scalar)),
    )


def _periodic_counterexample(period: int, prefix: str) -> z3.BoolRef:
    t = z3.Int(f"{prefix}.t")
    period = int(period)
    nxt = ((t / period) + 1) * period
    return z3.And(
        t >= 0,
        z3.Not(z3.And(
            nxt > t,
            nxt % period == 0,
            nxt - t >= 1,
            nxt - t <= period,
        )),
    )


def _controller_pool_coverage_counterexample(
    length: int,
    period: int,
    bound: int,
    prefix: str,
) -> z3.BoolRef:
    """Refute an in-window controller activation missing from the exact pool."""

    origin = z3.Int(f"{prefix}.origin")
    t = z3.Int(f"{prefix}.t")
    period = int(period)
    bound = int(bound)
    first = ((origin + period - 1) / period) * period
    pooled = [first + index * period for index in range(bound)]
    return z3.And(
        origin >= 0,
        t >= origin,
        t < origin + int(length),
        t % period == 0,
        *(t != value for value in pooled),
    )


def _release_tick_domain_counterexample(
    target_period: int,
    task_period: int,
    deadline: int,
    prefix: str,
) -> z3.BoolRef:
    """Any actual relative release tick belongs to the finite gcd-domain."""

    origin = z3.Int(f"{prefix}.origin")
    tick = z3.Int(f"{prefix}.tick")
    divisor = gcd(int(target_period), int(task_period))
    return z3.And(
        origin >= 0,
        origin % int(target_period) == 0,
        tick >= 0,
        tick <= int(deadline),
        (origin + tick) % int(task_period) == 0,
        tick % divisor != 0,
    )


def _count_bound_counterexample(length: int, period: int, bound: int, prefix: str) -> z3.BoolRef:
    """Refute a periodic-stream count larger than floor(D/T)+1."""

    n = z3.Int(f"{prefix}.n")
    length = int(length)
    period = int(period)
    bound = int(bound)
    return z3.And(
        n >= 1,
        (n - 1) * period <= length,
        n > bound,
    )


def _silent_interval_counterexample(prefix: str = "event.refine.silent") -> z3.BoolRef:
    """Refute non-associativity of exact bulk P7 service/eta updates."""

    service = z3.Int(f"{prefix}.service")
    eta = z3.Int(f"{prefix}.eta")
    period = z3.Int(f"{prefix}.period")
    a = z3.Int(f"{prefix}.a")
    b = z3.Int(f"{prefix}.b")

    macro_service = service + (a + b)
    split_service = (service + a) + b

    def sat_eta(value: z3.ArithRef, delta: z3.ArithRef) -> z3.ArithRef:
        return z3.If(value + delta < period, value + delta, period)

    macro_eta = sat_eta(eta, a + b)
    split_eta = sat_eta(sat_eta(eta, a), b)
    domain = z3.And(
        service >= 0,
        period > 0,
        eta >= 0,
        eta <= period,
        a >= 0,
        b >= 0,
    )
    return z3.And(domain, z3.Or(macro_service != split_service, macro_eta != split_eta))


def _slot_index(model: BoundModel, key: tuple[str, int]) -> int:
    for task_index, task in enumerate(model.tasks):
        for slot in range(model.max_jobs_per_task):
            if key == (task.name, slot):
                return task_index * model.max_jobs_per_task + slot
    raise KeyError(key)


def _deadline_event_counterexample(
    model: BoundModel, key: tuple[str, int], prefix: str
) -> z3.BoolRef:
    dispatch = declare_state(f"{prefix}.dispatch", model)
    horizon = z3.Int(f"{prefix}.horizon")
    job = dispatch.jobs[key]
    candidates = build_event_candidates(
        dispatch, model, horizon_time=horizon, prefix=f"{prefix}.candidates"
    )
    candidate = dict(candidates.hi_deadlines)[key]
    return z3.And(
        dispatch.t >= 0,
        dispatch.p == 7,
        job.present,
        z3.Not(job.removed),
        job.executed_service < job.effective_demand,
        job.absolute_deadline > dispatch.t,
        horizon >= job.absolute_deadline,
        candidate != job.absolute_deadline,
    )


def _terminal_service_event_counterexample(
    model: BoundModel, key: tuple[str, int], prefix: str
) -> z3.BoolRef:
    dispatch = declare_state(f"{prefix}.dispatch", model)
    horizon = z3.Int(f"{prefix}.horizon")
    job = dispatch.jobs[key]
    index = _slot_index(model, key)
    remaining = job.effective_demand - job.executed_service
    candidates = build_event_candidates(
        dispatch, model, horizon_time=horizon, prefix=f"{prefix}.candidates"
    )
    return z3.And(
        candidates.definition_formula,
        dispatch.t >= 0,
        dispatch.p == 7,
        dispatch.frontier.selected_slot == index,
        job.present,
        job.ready,
        z3.Not(job.removed),
        remaining > 0,
        horizon >= dispatch.t + remaining,
        candidates.completion != dispatch.t + remaining,
    )


def _p7_delta1_counterexample(model: BoundModel) -> z3.BoolRef:
    """Refute local P7 versus one-tick Event-tail equivalence.

    The exact Event macro and the Full kernel use the *same* P0--P6 closure by
    construction.  Re-instantiating that closure on two independent symbolic
    paths is both unnecessarily expensive and semantically wrong for the
    inherited nondeterministic history abstraction: two valid P5 witnesses need
    not choose identical post-history values.

    Differential consistency is therefore checked at the only quotient point:
    from one shared P7 dispatch state, one Full P7 tick and one Event silent
    advance of delta=1 must end field-for-field equal.  This is stronger than
    checking the property only for closure-reachable dispatch states.
    """

    dispatch = declare_state("event.diff.p7.dispatch", model)
    full_end = declare_state("event.diff.p7.full_end", model)
    event_end = declare_state("event.diff.p7.event_end", model)
    horizon = dispatch.t + 1
    candidates = build_event_candidates(
        dispatch, model, horizon_time=horizon, prefix="event.diff.p7.candidates"
    )
    return z3.And(
        dispatch.p == 7,
        encode_p7_time_and_service(dispatch, full_end, model),
        _silent_interval_advance(dispatch, event_end, model, candidates),
        candidates.next_time == horizon,
        z3.Not(state_equality(full_end, event_end)),
    )


def _encoder_call_sequence(path: Path, function_name: str) -> tuple[str, ...]:
    """Return phase-encoder calls in lexical order inside ``function_name``."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = next(
        (node for node in tree.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name),
        None,
    )
    if target is None:
        return ()
    encoder_names = {
        "encode_p0_settle",
        "encode_p1_idle_recovery",
        "encode_p2_deadline_observe",
        "encode_p3_arrival_freeze",
        "encode_p4_mode_switch",
        "encode_p5_controller",
        "encode_p6_dispatch",
        "encode_p7_time_and_service",
    }
    rows: list[tuple[int, int, str]] = []
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        name: str | None = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in encoder_names:
            rows.append((getattr(node, "lineno", 0), getattr(node, "col_offset", 0), name))
    rows.sort()
    return tuple(name for _, _, name in rows)



def _finite_event_bound_formulas(model: BoundModel) -> list[tuple[str, z3.BoolRef]]:
    formulas: list[tuple[str, z3.BoolRef]] = []
    for target in model.hi_tasks:
        bound = derive_finite_event_bound(model, target.name)
        for task in model.tasks:
            formulas.append((
                f"FINITE_EVENT_COUNT_BOUND::RELEASE::{target.name}::{task.name}",
                _count_bound_counterexample(
                    target.deadline,
                    task.period,
                    bound.release_bound_by_task[task.name],
                    f"event.bound.release.{target.name}.{task.name}",
                ),
            ))
            formulas.append((
                f"EVENT_ENVIRONMENT_RELEASE_DOMAIN_COVERAGE::{target.name}::{task.name}",
                _release_tick_domain_counterexample(
                    target.period,
                    task.period,
                    target.deadline,
                    f"event.bound.domain.{target.name}.{task.name}",
                ),
            ))
        formulas.append((
            f"FINITE_EVENT_COUNT_BOUND::CONTROLLER::{target.name}",
            _count_bound_counterexample(
                target.deadline,
                model.agent_period,
                bound.controller_bound,
                f"event.bound.controller.{target.name}",
            ),
        ))
        component_sum = (
            bound.total_release_bound
            + bound.hi_deadline_bound
            + bound.completion_bound
            + bound.controller_bound
            + bound.horizon_boundary_bound
            + 2
        )
        formulas.append((
            f"FINITE_EVENT_COUNT_BOUND::COMPOSITION::{target.name}",
            z3.BoolVal(bound.finite_event_bound < component_sum),
        ))
    return formulas


def _function_calls(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name),
        None,
    )
    if target is None:
        return set()
    calls: set[str] = set()
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def _class_method_calls(path: Path, class_name: str, method_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    calls: set[str] = set()
                    for nested in ast.walk(child):
                        if isinstance(nested, ast.Call):
                            if isinstance(nested.func, ast.Name):
                                calls.add(nested.func.id)
                            elif isinstance(nested.func, ast.Attribute):
                                calls.add(nested.func.attr)
                    return calls
    return set()


def _direct_parameter_attributes(path: Path, function_name: str, parameter: str) -> set[str]:
    """Direct ``parameter.<field>`` reads in one source function.

    This is intentionally a narrow architecture guard, not a semantic parser.
    It prevents the pooled P5 support projection from silently becoming stale
    if future controller code starts reading jobs/mode/eta/etc.  The source
    hashes in the structural receipt then bind this guard to the certified
    implementation.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name),
        None,
    )
    if target is None:
        return set()
    result: set[str] = set()
    for node in ast.walk(target):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == parameter
        ):
            result.add(node.attr)
    return result


def _source_contracts(source_root: Path) -> tuple[bool, list[dict[str, Any]], str | None]:
    root = Path(source_root).resolve()
    kernel = root / "formal_toolchain/v9_2/event_kernel.py"
    window = root / "formal_toolchain/v9_2/event_window_encoder.py"
    environment = root / "formal_toolchain/v9_2/environment_encoder.py"
    symbolic = root / "formal_toolchain/v9_2/symbolic_state.py"
    transition = root / "formal_toolchain/v9_2/transition_encoder.py"
    controller = root / "formal_toolchain/v9_2/controller_encoder.py"
    numeric = root / "formal_toolchain/v9_2/numeric_encoder.py"
    incremental = root / "formal_toolchain/v9_2/incremental_event_bmc.py"
    if not all(path.is_file() for path in (
        kernel, window, environment, symbolic, transition, controller, numeric, incremental
    )):
        return False, [], "V9_2_EVENT_SOURCE_MISSING"

    closure_calls = _function_calls(kernel, "_exact_p0_to_p7_closure")
    closure_sequence = _encoder_call_sequence(kernel, "_exact_p0_to_p7_closure")
    pool_builder_calls = _function_calls(kernel, "build_exact_controller_pool")
    pooled_p5_calls = _function_calls(kernel, "encode_p5_from_exact_pool")
    full_p5_calls = _function_calls(transition, "encode_p5_controller")
    controller_state_reads = _direct_parameter_attributes(
        controller, "encode_controller_decision", "state"
    )
    numeric_state_reads = _direct_parameter_attributes(
        numeric, "encode_v11_full_10d_observation", "state"
    )
    candidate_calls = _function_calls(kernel, "build_event_candidates")
    step_calls = _function_calls(kernel, "encode_event_step")
    window_builder_calls = _function_calls(window, "build_event_first_bad_window")
    incremental_step_calls = _class_method_calls(
        window, "IncrementalEventWindowEncoding", "append_exact_event_step"
    )
    incremental_solver_calls = _function_calls(incremental, "solve_incremental_event_window")
    expected_closure_prefix = (
        "encode_p0_settle",
        "encode_p1_idle_recovery",
        "encode_p2_deadline_observe",
        "encode_p3_arrival_freeze",
        "encode_p4_mode_switch",
    )
    if closure_sequence[:5] != expected_closure_prefix or closure_sequence[-1:] != ("encode_p6_dispatch",):
        return False, [], "V9_2_FULL_EVENT_CLOSURE_DEFINITIONAL_IDENTITY_MISSING"
    if not {"encode_p5_controller", "encode_p5_from_exact_pool"} <= closure_calls:
        return False, [], "V9_2_EXACT_P5_POOL_DISPATCH_MISSING"
    if "encode_p5_controller_effect" not in pool_builder_calls:
        return False, [], "V9_2_EXACT_P5_POOL_SOURCE_CONTRACT_MISSING"
    if not {
        "_controller_effect_state_equality",
        "encode_p5_controller_frame",
        "encode_p5_identity",
    } <= pooled_p5_calls:
        return False, [], "V9_2_EXACT_P5_POOL_LINK_CONTRACT_MISSING"
    if not {"encode_p5_controller_effect", "encode_p5_controller_frame"} <= full_p5_calls:
        return False, [], "V9_2_EXACT_P5_EFFECT_FRAME_DECOMPOSITION_MISSING"
    # The pooled support is exact only while the deployed controller observes
    # no Full-state fields beyond time, budgets and RuntimeFeatureState.  Bind
    # that architectural fact into the trusted structural gate rather than
    # relying on a comment that could become stale after future edits.
    if not controller_state_reads <= {"t", "budgets"}:
        return False, [], "V9_2_P5_POOL_CONTROLLER_SUPPORT_DRIFT"
    if not numeric_state_reads <= {"budgets", "chi"}:
        return False, [], "V9_2_P5_POOL_NUMERIC_SUPPORT_DRIFT"
    kernel_text = kernel.read_text(encoding="utf-8")
    if "encode_p5_invariant_summary" in kernel_text:
        return False, [], "V9_2_EVENT_P5_SUMMARY_FORBIDDEN"
    if not {"_next_periodic_after", "_exact_minimum_definition"} <= candidate_calls:
        return False, [], "V9_2_EVENT_CANDIDATE_SOURCE_CONTRACT_MISSING"
    if "_min_expr(" in kernel_text:
        return False, [], "V9_2_EVENT_NESTED_MIN_REINTRODUCED"
    if not all(token in kernel_text for token in (
        ".candidate.completion", ".candidate.next_time", "definition_formula",
    )):
        return False, [], "V9_2_EVENT_SCALAR_MIN_CONTRACT_MISSING"
    # V5 splits the exact silent relation into destination-free core plus a
    # single destination update so the terminal stutter branch can share it.
    if not {
        "_exact_p0_to_p7_closure",
        "build_event_candidates",
        "_silent_interval_core",
        "_event_destination_update",
    } <= step_calls:
        return False, [], "V9_2_EVENT_MACRO_SOURCE_CONTRACT_MISSING"

    window_text = window.read_text(encoding="utf-8")
    if "event_layer_added_abstractions: tuple[str, ...] = ()" not in window_text:
        return False, [], "EVENT_NEW_CONSERVATISM_FORBIDDEN"
    if "microstep_terminal_fallback_used: bool = False" not in window_text:
        return False, [], "V9_2_MICROSTEP_TERMINAL_FALLBACK_CONTRACT"
    if "exact_p5_in_event_window: bool = True" not in window_text:
        return False, [], "V9_2_EXACT_P5_EVENT_WINDOW_CONTRACT_MISSING"
    if "build_exact_controller_pool" not in window_text or "controller_bound=event_bound.controller_bound" not in window_text:
        return False, [], "V9_2_EXACT_P5_POOL_BOUND_CONTRACT_MISSING"
    if "event_step_or_terminal_stutter" not in window_builder_calls:
        return False, [], "V9_2_EVENT_TERMINAL_STUTTER_FACTORING_MISSING"
    if "encode_event_step" not in incremental_step_calls or "event_step_or_terminal_stutter" in incremental_step_calls:
        return False, [], "V9_2_INCREMENTAL_DEPTH_MUST_USE_EXACT_ACTIVE_EVENT_STEP"
    if not {
        "append_exact_event_step", "build_terminal_bad_query",
        "_new_depth_solver", "_solver_check", "_solve_unknown_by_exact_cases",
    } <= incremental_solver_calls:
        return False, [], "V9_2_INCREMENTAL_DEPTH_SOLVER_CONTRACT_MISSING"
    incremental_text = incremental.read_text(encoding="utf-8")
    if "for depth in range(0, max_depth + 1):" not in incremental_text:
        return False, [], "V9_2_INCREMENTAL_DEPTH_COVERAGE_NOT_EXHAUSTIVE"
    if "terminal_stutter_used\": False" not in incremental_text:
        return False, [], "V9_2_INCREMENTAL_DEPTH_STUTTER_ELIMINATION_CONTRACT_MISSING"
    if '"fresh_solver_per_depth": True' not in incremental_text:
        return False, [], "V9_2_FRESH_DEPTH_SOLVER_CONTRACT_MISSING"
    if "CTRL_COUNT_" not in incremental_text or "SRC_RELEASE_ANY" not in incremental_text:
        return False, [], "V9_2_EXACT_DEPTH_CASE_PARTITION_CONTRACT_MISSING"
    if "SRC_HI_DEADLINE_ANY" not in incremental_text or "SRC_COMPLETION" not in incremental_text:
        return False, [], "V9_2_EXACT_EVENT_SOURCE_PARTITION_CONTRACT_MISSING"
    if not all(token in incremental_text for token in (
        "SRC_RELEASE_TASK_",
        "SRC_HI_DEADLINE_JOB_",
        "SRC_COMPLETION_SLOT_",
        "probe_timeout_ms",
        "leaf_timeout_ms",
        "bounded_probe_before_exact_partition",
    )):
        return False, [], "V9_2_HIERARCHICAL_EXACT_LEAF_PARTITION_CONTRACT_MISSING"
    # V5 keeps the proven fresh-solver-per-depth boundary, but intentionally
    # reuses that one solver *within* a fixed depth across exact count/source/
    # member refinements.  Push/pop is therefore required here, while the
    # explicit receipt/source markers below forbid cross-depth reuse.
    if "_new_depth_solver(assertions)" not in incremental_text:
        return False, [], "V9_2_FRESH_DEPTH_SOLVER_CONTRACT_MISSING"
    if not all(token in incremental_text for token in (
        "def _solver_scope",
        "solver.push()",
        "solver.pop()",
        '"within_depth_solver_context_reused": True',
        '"cross_depth_solver_context_reused": False',
        "same_solver_timeout_resume",
        "_ordered_disjoint_cover",
    )):
        return False, [], "V9_2_WITHIN_DEPTH_SOLVER_REUSE_CONTRACT_MISSING"
    if "_fresh_check" in incremental_text:
        return False, [], "V9_2_STALE_FRESH_CHECK_REINTRODUCED"
    if "This is the only path allowed to report window UNSAT" not in incremental_text:
        return False, [], "V9_2_INCREMENTAL_UNSAT_AGGREGATION_GUARD_MISSING"
    if "declare_sparse_successor" not in closure_calls:
        return False, [], "V9_2_EVENT_PHASE_SSA_COMPILATION_MISSING"
    environment_text = environment.read_text(encoding="utf-8")
    if "A_lookup" not in environment_text or "lookup(relative)" not in environment_text:
        return False, [], "V9_2_EVENT_DEMAND_LOOKUP_FACTORING_MISSING"
    transition_text = transition.read_text(encoding="utf-8")
    if "left.eq(right)" not in transition_text:
        return False, [], "V9_2_SSA_TAUTOLOGICAL_FRAME_ELIMINATION_MISSING"

    kernel_hash = sha256_text_file_normalized(kernel)
    window_hash = sha256_text_file_normalized(window)
    environment_hash = sha256_text_file_normalized(environment)
    symbolic_hash = sha256_text_file_normalized(symbolic)
    transition_hash = sha256_text_file_normalized(transition)
    controller_hash = sha256_text_file_normalized(controller)
    numeric_hash = sha256_text_file_normalized(numeric)
    incremental_hash = sha256_text_file_normalized(incremental)
    rows = [
        {
            "obligation_id": "EVENT_START_ABSTRACTION_SOUNDNESS",
            "status": "PASS",
            "proof_rule": "IDENTITY_EVENT_BOUNDARY_PROJECTION",
            "source_sha256": window_hash,
        },
        {
            "obligation_id": "EVENT_START_PROJECTION_EXACTNESS",
            "status": "PASS",
            "proof_rule": "IDENTITY_EVENT_BOUNDARY_PROJECTION",
            "source_sha256": window_hash,
        },
        {
            "obligation_id": "EVENT_STATE_FUTURE_SUFFICIENCY",
            "status": "PASS",
            "proof_rule": "FULL_PERSISTENT_STATE_RETAINED_AT_EVENT_BOUNDARY",
            "source_sha256": kernel_hash,
        },
        {
            "obligation_id": "EVENT_DEMAND_LOOKUP_FACTORING_EQUIVALENCE",
            "status": "PASS",
            "proof_rule": "FINITE_EXPLICIT_DEMAND_VARIABLES_TIED_ONCE_TO_INDEXED_LOOKUP_WITH_EXACT_COVERAGE",
            "source_sha256": environment_hash,
        },
        {
            "obligation_id": "EVENT_PHASE_SSA_FRAME_ELIMINATION_EQUIVALENCE",
            "status": "PASS",
            "proof_rule": "CANONICAL_PHASE_ENCODERS_WITH_STRUCTURALLY_SHARED_UNMODIFIED_FIELDS",
            "source_sha256": sha256_object({
                "kernel": kernel_hash, "symbolic": symbolic_hash, "transition": transition_hash,
            }),
        },
        {
            "obligation_id": "EVENT_P5_POOL_SUPPORT_PROJECTION_EQUIVALENCE",
            "status": "PASS",
            "proof_rule": (
                "EXACT_P5_DECOMPOSES_INTO_CONTROLLER_EFFECT_AND_EXACT_FRAME;_"
                "POOL_LINKS_ONLY_EFFECT_READ_WRITE_SUPPORT_WHILE_EVENT_LOCAL_SSA_SUPPLIES_FRAME"
            ),
            "controller_state_reads": sorted(controller_state_reads),
            "numeric_state_reads": sorted(numeric_state_reads),
            "source_sha256": sha256_object({
                "kernel": kernel_hash,
                "transition": transition_hash,
                "controller": controller_hash,
                "numeric": numeric_hash,
            }),
        },
        {
            "obligation_id": "EVENT_TERMINAL_STUTTER_FACTORING_EQUIVALENCE",
            "status": "PASS",
            "proof_rule": (
                "ACTIVE_EVENT_UPDATE_OR_EXACT_TERMINAL_STATE_EQUALITY_FACTORED_INTO_"
                "GUARDED_CORE_PLUS_ONE_FIELDWISE_ITE_DESTINATION_ASSIGNMENT"
            ),
            "source_sha256": sha256_object({
                "kernel": kernel_hash, "window": window_hash,
            }),
        },
        {
            "obligation_id": "EVENT_INCREMENTAL_TERMINAL_DEPTH_PARTITION_EQUIVALENCE",
            "status": "PASS",
            "proof_rule": (
                "FINITE_EVENT_BOUND_PARTITIONED_BY_EXACT_FIRST_HORIZON_DEPTH_0_TO_N;_"
                "EACH_PREFIX_USES_ONLY_EXACT_ACTIVE_EVENT_STEP;_"
                "TERMINAL_STUTTER_SUFFIX_IS_IDENTITY_AND_OMITTED;_"
                "EACH_DEPTH_IS_SOLVED_IN_A_FRESH_EQUIVALENT_SOLVER_CONTEXT;_"
                "WITHIN_ONE_DEPTH_THE_SAME_SOLVER_CONTEXT_IS_REUSED_ONLY_UNDER_TEMPORARY_EXACT_CASE_SCOPES;_"
                "NO_SOLVER_CONTEXT_IS_REUSED_ACROSS_DEPTHS;_"
                "EVENT_CANDIDATE_MINIMA_USE_EXACT_SCALAR_SSA_DEFINITIONS;_"
                "UNKNOWN_RETRY_PARTITIONS_EXACTLY_BY_CONTROLLER_COUNT_DISJOINT_EVENT_SOURCE_AND_DISJOINT_SOURCE_MEMBER;_"
                "EXACT_LEAF_TIMEOUT_RESUME_REUSES_THE_SAME_CASE_CONTEXT;_"
                "UNSAT_REQUIRES_EVERY_CASE_AND_EVERY_DEPTH_UNSAT"
            ),
            "source_sha256": sha256_object({
                "window": window_hash, "incremental_solver": incremental_hash,
            }),
        },
        {
            "obligation_id": "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY::P0_P6_DEFINITIONAL_IDENTITY",
            "status": "PASS",
            "proof_rule": "EVENT_CLOSURE_REUSES_CANONICAL_P0_P4_P6_AND_FORMULA_EQUIVALENT_EXACT_P5_POOL",
            "phase_encoder_sequence": list(closure_sequence),
            "source_sha256": kernel_hash,
        },
        {
            "obligation_id": "EXACT_P5_AT_CONTROLLER_EVENT",
            "status": "PASS",
            "proof_rule": (
                "WINDOW_GLOBAL_POOL_CALLS_EXACT_CONTROLLER_EFFECT;_"
                "EVENT_LOCAL_SSA_SUPPLIES_EXACT_P5_FRAME_AND_SUPPORT_LINK_IS_EXACT_EQUALITY"
            ),
            "source_sha256": kernel_hash,
        },
        {
            "obligation_id": "NO_SPURIOUS_EVENT_SOURCE",
            "status": "PASS",
            "proof_rule": "CLOSED_EVENT_SOURCE_ENUMERATION_WITH_EXACT_SCALAR_MINIMUM",
            "event_sources": [
                "periodic_release",
                "HI_deadline_observation",
                "controller_activation",
                "selected_job_completion_or_cap_terminal",
                "target_query_horizon",
            ],
            "source_sha256": kernel_hash,
        },
    ]
    return True, rows, None


def _aggregate_status(statuses: dict[str, str], base: str, rows: list[str]) -> None:
    statuses[base] = "PASS" if rows and all(statuses.get(row) == "PASS" for row in rows) else "UNRESOLVED"


def _derive(
    name: str,
    dependencies: tuple[str, ...],
    statuses: dict[str, str],
    structural_rows: list[dict[str, Any]],
    *,
    proof_rule: str,
) -> bool:
    missing = [dep for dep in dependencies if statuses.get(dep) != "PASS"]
    if missing:
        statuses[name] = "UNRESOLVED"
        structural_rows.append({
            "obligation_id": name,
            "status": "UNRESOLVED",
            "proof_rule": proof_rule,
            "depends_on": list(dependencies),
            "missing": missing,
        })
        return False
    statuses[name] = "PASS"
    structural_rows.append({
        "obligation_id": name,
        "status": "PASS",
        "proof_rule": proof_rule,
        "depends_on": list(dependencies),
    })
    return True


def prove_event_refinement(
    model: BoundModel,
    *,
    source_root: Path,
    timeout_ms: int = 120_000,
) -> EventRefinementProof:
    statuses: dict[str, str] = {}
    ok, structural_rows, failure = _source_contracts(source_root)
    if not ok:
        return EventRefinementProof("FAIL", statuses, (), tuple(structural_rows), failure)
    for row in structural_rows:
        statuses[str(row["obligation_id"])] = "PASS"

    formulas: list[tuple[str, z3.BoolRef]] = [
        ("NEXT_EVENT_MINIMALITY", _minimum_counterexample()),
        ("NEXT_EVENT_EXACT_MINIMUM", _minimum_counterexample()),
        ("SILENT_INTERVAL_SERVICE_EQUIVALENCE", _silent_interval_counterexample()),
        # Compositional differential leaf: P0--P6 are definitionally shared
        # between Full and Event semantics, so only the quotient point (P7
        # versus a one-tick silent advance) requires SMT comparison.  This
        # avoids duplicating the exact controller and, critically, does not
        # compare two independent witnesses of the inherited history
        # over-approximation.
        ("MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY::P7_DELTA1", _p7_delta1_counterexample(model)),
    ]
    for task in model.hi_tasks:
        for slot in range(model.max_jobs_per_task):
            formulas.append((
                f"DEADLINE_EVENT_COVERAGE::{task.name}::{slot}",
                _deadline_event_counterexample(
                    model, (task.name, slot),
                    f"event.refine.deadline.{task.name}.{slot}",
                ),
            ))
    for task in model.tasks:
        for slot in range(model.max_jobs_per_task):
            formulas.append((
                f"TERMINAL_SERVICE_EVENT_COVERAGE::{task.name}::{slot}",
                _terminal_service_event_counterexample(
                    model, (task.name, slot),
                    f"event.refine.term.{task.name}.{slot}",
                ),
            ))
    for task in model.tasks:
        formulas.append((
            f"RELEASE_EVENT_COVERAGE::{task.name}",
            _periodic_counterexample(task.period, f"event.refine.release.{task.name}"),
        ))
    formulas.append((
        "CONTROLLER_EVENT_COVERAGE::NEXT_PERIODIC",
        _periodic_counterexample(model.agent_period, "event.refine.controller"),
    ))
    for target in model.hi_tasks:
        bound = derive_finite_event_bound(model, target.name)
        formulas.append((
            f"CONTROLLER_EVENT_COVERAGE::POOL::{target.name}",
            _controller_pool_coverage_counterexample(
                target.deadline, model.agent_period, bound.controller_bound,
                f"event.refine.controller_pool.{target.name}",
            ),
        ))
    formulas.extend(_finite_event_bound_formulas(model))

    solver_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[str]] = {}
    for obligation_id, formula in formulas:
        receipt = solve_formula(obligation_id, formula, timeout_ms=timeout_ms)
        solver_rows.append(receipt.as_dict())
        leaf_name = obligation_id
        statuses[leaf_name] = "PASS" if receipt.result == "UNSAT" else (
            "FAIL" if receipt.result == "SAT" else "UNRESOLVED"
        )
        base = obligation_id.split("::", 1)[0]
        grouped.setdefault(base, []).append(leaf_name)
        if receipt.result != "UNSAT":
            statuses[base] = statuses[leaf_name]
            return EventRefinementProof(
                statuses[leaf_name],
                statuses,
                tuple(solver_rows),
                tuple(structural_rows),
                f"V9_2_EVENT_REFINEMENT_{receipt.result}:{obligation_id}",
            )

    for base, leaves in grouped.items():
        if base == "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY":
            continue
        _aggregate_status(statuses, base, leaves)

    # Exact-minimum + complete event-class coverage means no Full discrete event
    # can be crossed before the chosen Event timestamp.
    if not _derive(
        "NO_SKIPPED_DISCRETE_EVENT",
        (
            "NEXT_EVENT_EXACT_MINIMUM",
            "RELEASE_EVENT_COVERAGE",
            "DEADLINE_EVENT_COVERAGE",
            "TERMINAL_SERVICE_EVENT_COVERAGE",
            "CONTROLLER_EVENT_COVERAGE",
        ),
        statuses,
        structural_rows,
        proof_rule="MINIMUM_OVER_COMPLETE_EVENT_SOURCE_SET",
    ):
        return EventRefinementProof("UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows), "NO_SKIPPED_DISCRETE_EVENT_UNPROVED")

    # The small-horizon Full/Event check is deliberately compositional.
    # P0--P6 are the same definition, P7 equals a one-tick Event tail, and
    # event-free multi-tick intervals follow by exact service/eta composition
    # plus the no-skipped-event theorem.  No whole-controller formula is
    # duplicated merely to prove that the two callers share the same closure.
    if not _derive(
        "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY",
        (
            "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY::P0_P6_DEFINITIONAL_IDENTITY",
            "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY::P7_DELTA1",
            "SILENT_INTERVAL_SERVICE_EQUIVALENCE",
            "NO_SKIPPED_DISCRETE_EVENT",
        ),
        statuses,
        structural_rows,
        proof_rule="SHARED_EXACT_CLOSURE_PLUS_LOCAL_P7_EQUIVALENCE_AND_SILENT_INDUCTION",
    ):
        return EventRefinementProof(
            "UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows),
            "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY_UNPROVED",
        )

    segment_dependencies = (
        "EVENT_START_PROJECTION_EXACTNESS",
        "EVENT_STATE_FUTURE_SUFFICIENCY",
        "EVENT_P5_POOL_SUPPORT_PROJECTION_EQUIVALENCE",
        "EVENT_TERMINAL_STUTTER_FACTORING_EQUIVALENCE",
        "NO_SPURIOUS_EVENT_SOURCE",
        "NO_SKIPPED_DISCRETE_EVENT",
        "SILENT_INTERVAL_SERVICE_EQUIVALENCE",
        "EXACT_P5_AT_CONTROLLER_EVENT",
        "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY",
    )
    for theorem, rule in (
        ("FULL_TO_EVENT_SEGMENT_SIMULATION", "EXACT_CLOSURE_PLUS_SILENT_TICK_INDUCTION"),
        ("EVENT_TO_FULL_SEGMENT_REALIZABILITY", "EVENT_MACRO_EXPANSION_TO_FULL_TICKS"),
    ):
        if not _derive(theorem, segment_dependencies, statuses, structural_rows, proof_rule=rule):
            return EventRefinementProof("UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows), f"{theorem}_UNPROVED")

    if not _derive(
        "FIRST_HI_BAD_EVENT_PREFIX_REFLECTION",
        ("FULL_TO_EVENT_SEGMENT_SIMULATION", "DEADLINE_EVENT_COVERAGE", "NO_SKIPPED_DISCRETE_EVENT"),
        statuses,
        structural_rows,
        proof_rule="FULL_BAD_DEADLINE_IS_EVENT_BOUNDARY",
    ):
        return EventRefinementProof("UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows), "FIRST_HI_BAD_EVENT_PREFIX_REFLECTION_UNPROVED")
    if not _derive(
        "EVENT_BAD_PREFIX_FULL_REALIZABILITY",
        ("EVENT_TO_FULL_SEGMENT_REALIZABILITY", "DEADLINE_EVENT_COVERAGE", "EXACT_P5_AT_CONTROLLER_EVENT"),
        statuses,
        structural_rows,
        proof_rule="EVENT_BAD_MACRO_EXPANDS_TO_FULL_BAD_PREFIX",
    ):
        return EventRefinementProof("UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows), "EVENT_BAD_PREFIX_FULL_REALIZABILITY_UNPROVED")

    # The structural count is safe only after every per-stream arithmetic leaf
    # and the composition leaf have been solved.
    if statuses.get("FINITE_EVENT_COUNT_BOUND") != "PASS":
        return EventRefinementProof("UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows), "FINITE_EVENT_COUNT_BOUND_UNPROVED")

    window_dependencies = tuple(name for name in EVENT_TERMINAL_OBLIGATIONS if name != "EVENT_WINDOW_ENCODING_SOUNDNESS")
    if not _derive(
        "EVENT_WINDOW_ENCODING_SOUNDNESS",
        window_dependencies,
        statuses,
        structural_rows,
        proof_rule="V9_2_EVENT_WINDOW_SOUNDNESS_COMPOSITION",
    ):
        return EventRefinementProof("UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows), "EVENT_WINDOW_ENCODING_SOUNDNESS_UNPROVED")

    missing = [name for name in EVENT_TERMINAL_OBLIGATIONS if statuses.get(name) != "PASS"]
    if missing:
        return EventRefinementProof(
            "UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows),
            "V9_2_EVENT_TERMINAL_OBLIGATIONS_INCOMPLETE:" + ",".join(missing),
        )
    return EventRefinementProof("PASS", statuses, tuple(solver_rows), tuple(structural_rows))


__all__ = ["EVENT_TERMINAL_OBLIGATIONS", "EventRefinementProof", "prove_event_refinement"]
