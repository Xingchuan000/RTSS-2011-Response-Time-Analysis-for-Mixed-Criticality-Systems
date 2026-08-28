"""Verifier-side boot reachability for SAT V9.2 first-bad windows.

A window SAT model starts from an arbitrary state satisfying Psi.  That is the
right over-approximation for proving safety by UNSAT, but a SAT model is not a
concrete counterexample until its z0 can also be reached from the deployed boot
state without an earlier HI miss.  This module rebuilds that prefix in a fresh
solver; it never trusts reachability flags supplied by a candidate bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

import z3

from .environment_encoder import declare_environment
from .formula_solver import canonical_formula_text
from .safe_prefix_invariant import SafePrefixInvariant
from .symbolic_state import BoundModel, SymbolicKernelState, declare_state
from .transition_encoder import encode_phase_step
from .event_window_encoder import EventWindowEncoding


@dataclass(frozen=True, slots=True)
class ReachabilityResult:
    status: str
    code: str
    reason: str | None = None
    origin_time: int | None = None
    demand_prefix: Mapping[tuple[str, int], int] | None = None
    formula_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"status": self.status, "code": self.code}
        if self.reason is not None:
            row["reason"] = self.reason
        if self.origin_time is not None:
            row["origin_time"] = self.origin_time
        if self.formula_hash is not None:
            row["formula_hash"] = self.formula_hash
        if self.demand_prefix is not None:
            row["demand_prefix"] = {
                f"{task}:{release_index}": value
                for (task, release_index), value in sorted(self.demand_prefix.items())
            }
        return row


def _eval_int(model: z3.ModelRef, value: z3.ArithRef) -> int:
    evaluated = model.eval(value, model_completion=True)
    if not z3.is_int_value(evaluated):
        raise ValueError(f"expected integer model value for {value}, got {evaluated}")
    return int(evaluated.as_long())


def _pin_job(left: Any, right: Any, right_model: z3.ModelRef) -> list[z3.BoolRef]:
    fields = (
        "present", "release_index", "release_time", "absolute_deadline", "tie_break",
        "release_entry_mode_hi", "classification_abnormal", "budget_at_release",
        "actual_demand", "effective_demand", "executed_service", "removed", "ready",
    )
    return [getattr(left, name) == right_model.eval(getattr(right, name), model_completion=True)
            for name in fields]


def _pin_state(
    left: SymbolicKernelState,
    right: SymbolicKernelState,
    right_model: z3.ModelRef,
) -> z3.BoolRef:
    clauses: list[z3.BoolRef] = [
        left.t == right_model.eval(right.t, model_completion=True),
        left.p == right_model.eval(right.p, model_completion=True),
        left.mode_hi == right_model.eval(right.mode_hi, model_completion=True),
        left.frontier.selected_slot == right_model.eval(right.frontier.selected_slot, model_completion=True),
        left.frontier.running == right_model.eval(right.frontier.running, model_completion=True),
        left.hi_miss_ledger == right_model.eval(right.hi_miss_ledger, model_completion=True),
    ]
    for name in left.budgets:
        clauses.extend((
            left.budgets[name] == right_model.eval(right.budgets[name], model_completion=True),
            left.eta[name] == right_model.eval(right.eta[name], model_completion=True),
            left.chi.recent_cost[name] == right_model.eval(right.chi.recent_cost[name], model_completion=True),
            left.chi.ema_cost[name] == right_model.eval(right.chi.ema_cost[name], model_completion=True),
            left.chi.overrun_ema[name] == right_model.eval(right.chi.overrun_ema[name], model_completion=True),
            left.chi.max_cost_k[name] == right_model.eval(right.chi.max_cost_k[name], model_completion=True),
        ))
    for key in left.jobs:
        clauses.extend(_pin_job(left.jobs[key], right.jobs[key], right_model))
    for left_values, right_values in (
        (left.chi.mode_change_window, right.chi.mode_change_window),
        (left.chi.lo_cancel_window, right.chi.lo_cancel_window),
        (left.chi.hi_overrun_window, right.chi.hi_overrun_window),
        (left.chi.lo_overrun_window, right.chi.lo_overrun_window),
        (left.chi.job_start_window, right.chi.job_start_window),
    ):
        clauses.extend(a == right_model.eval(b, model_completion=True)
                       for a, b in zip(left_values, right_values))
    return z3.And(*clauses)


def _extract_release_demands(
    env: Any,
    solver_model: z3.ModelRef,
    model: BoundModel,
    *,
    start_tick: int,
    stop_tick: int,
) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for task in model.tasks:
        for tick in range(start_tick, stop_tick + 1):
            if tick % task.period != 0:
                continue
            variable = env.actual_demands[(task.name, tick)]
            result[(task.name, tick // task.period)] = _eval_int(solver_model, variable)
    return result


def prove_boot_safe_prefix_reachability(
    encoding: EventWindowEncoding,
    window_model: z3.ModelRef,
    model: BoundModel,
    invariant: SafePrefixInvariant,
    *,
    timeout_ms: int = 120_000,
    max_boot_ticks: int = 2_000,
) -> ReachabilityResult:
    """Freshly prove that the concrete SAT z0 has a symbolic boot-safe prefix.

    The boot prefix uses exactly the same P0..P7 transition encoder, but starts
    from the independently regenerated boot contract at t=0.  The final state is
    pinned field-for-field to the SAT window's z0.  Policy history remains the
    deliberate V9.2 over-approximation, so PASS here is necessary but not
    sufficient for declaring the deployed runtime unsafe; concrete replay is
    still mandatory afterwards.
    """

    z0 = encoding.start_state
    try:
        origin = _eval_int(window_model, z0.t)
    except ValueError as exc:
        return ReachabilityResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE", str(exc))
    if origin < 0:
        return ReachabilityResult("FAIL", "BOOT_SAFE_PREFIX_UNREACHABLE", "negative window origin")
    if origin > max_boot_ticks:
        return ReachabilityResult(
            "UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
            f"boot prefix origin {origin} exceeds verifier replay bound {max_boot_ticks}",
            origin_time=origin,
        )

    boot_env = declare_environment("sat.boot.env", model, release_count=origin + 1)
    states = tuple(declare_state(f"sat.boot.z.{index}", model) for index in range(origin * 8 + 1))
    clauses: list[z3.BoolRef] = [
        *boot_env.constraints,
        boot_env.phase.origin_time == 0,
        *invariant.initial_constraints(states[0]),
        invariant.formula(states[0]),
    ]
    for index in range(len(states) - 1):
        phase = index % 8
        absolute_tick = index // 8
        clauses.append(encode_phase_step(
            states[index], states[index + 1], model, boot_env,
            phase=phase,
            controller_may_fire=(absolute_tick % model.agent_period == 0),
        ))
    # M is sticky, so final M=0 already implies no earlier HI miss.  Keep the
    # explicit per-P2 checks as a regression guard on that semantic fact.
    for tick in range(origin):
        clauses.append(states[tick * 8 + 3].hi_miss_ledger == 0)
    clauses.extend((
        states[-1].t == origin,
        states[-1].p == 0,
        states[-1].hi_miss_ledger == 0,
        _pin_state(states[-1], z0, window_model),
    ))
    formula = z3.And(*clauses)
    text = canonical_formula_text(formula)
    formula_hash = sha256(text.encode("utf-8")).hexdigest()
    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    solver.add(formula)
    result = solver.check()
    if result == z3.unsat:
        return ReachabilityResult(
            "FAIL", "BOOT_SAFE_PREFIX_UNREACHABLE",
            "the SAT window z0 has no boot-safe prefix in the regenerated kernel",
            origin_time=origin, formula_hash=formula_hash,
        )
    if result != z3.sat:
        return ReachabilityResult(
            "UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
            solver.reason_unknown(), origin_time=origin, formula_hash=formula_hash,
        )
    prefix_demands = _extract_release_demands(
        boot_env, solver.model(), model, start_tick=0, stop_tick=max(origin - 1, 0)
    ) if origin > 0 else {}
    return ReachabilityResult(
        "PASS", "Z0_SAFE_PREFIX_REACHABLE", origin_time=origin,
        demand_prefix=prefix_demands, formula_hash=formula_hash,
    )


__all__ = ["ReachabilityResult", "prove_boot_safe_prefix_reachability"]
