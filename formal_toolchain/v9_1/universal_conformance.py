"""Universal frozen-runtime -> V9.1 kernel conformance obligations.

Finite traces are diagnostics only.  The blocking route combines source-bound
machine-checked project lemmas for the frozen runtime with fresh counterexample
queries over the V9.1 phase relation.  A conformance clause passes only when the
source theorem backend re-verifies and its corresponding V9.1 counterexample is
UNSAT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import z3

from formal_toolchain.theory.loader import load_verified_theory_statement

from .environment_encoder import declare_environment
from .formula_solver import FormulaReceipt, solve_formula
from .safe_prefix_invariant import SafePrefixInvariant
from .symbolic_state import BoundModel, SymbolicKernelState, declare_state
from .transition_encoder import (
    encode_p2_deadline_observe,
    encode_p3_arrival_freeze,
    encode_p7_time_and_service,
    encode_step,
)


RUNTIME_THEOREMS = (
    "EVENT_HANDLER_MICROSTEP_DECOMPOSITION",
    "ARRIVAL_BATCH_LOOP_DECOMPOSITION",
    "FINITE_RELEASE_FOLD_PRESERVES_RELATION",
    "CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",
    "FINITE_HI_BAD_PREFIX_REFLECTION",
)


@dataclass(frozen=True, slots=True)
class ConformanceProof:
    status: str
    theorem_receipts: tuple[dict[str, Any], ...]
    solver_receipts: tuple[FormulaReceipt, ...]
    failure_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_code": self.failure_code,
            "theorems": list(self.theorem_receipts),
            "solver_receipts": [row.as_dict() for row in self.solver_receipts],
        }


def _theorem_receipts(source_root: Path) -> tuple[dict[str, Any], ...]:
    theory_dir = Path(source_root).resolve() / "formal_toolchain" / "theory"
    rows: list[dict[str, Any]] = []
    for theorem_id in RUNTIME_THEOREMS:
        theorem = load_verified_theory_statement(theory_dir, theorem_id)
        rows.append({
            "theorem_id": theorem_id,
            "statement_hash": theorem["statement_hash"],
            "assumption_hash": theorem["assumption_hash"],
            "assurance_level": theorem["assurance_level"],
            "status": "PASS",
        })
    return tuple(rows)


def _job_changed_except_phase_owned(z: SymbolicKernelState, zp: SymbolicKernelState) -> z3.BoolRef:
    changes: list[z3.BoolRef] = []
    fields = (
        "present", "release_index", "release_time", "absolute_deadline", "tie_break",
        "release_entry_mode_hi", "classification_abnormal", "budget_at_release",
        "actual_demand", "effective_demand", "executed_service", "removed", "ready",
    )
    for key, job in z.jobs.items():
        other = zp.jobs[key]
        changes.extend(getattr(other, name) != getattr(job, name) for name in fields)
    return z3.Or(*changes) if changes else z3.BoolVal(False)


def _phase_order_counterexample(model: BoundModel) -> z3.BoolRef:
    z = declare_state("conf.phase.z", model)
    zp = declare_state("conf.phase.zp", model)
    env = declare_environment("conf.phase.env", model, release_count=1)
    expected = z3.Or(*(
        z3.And(z.p == phase, zp.p == ((phase + 1) % 8))
        for phase in range(8)
    ))
    inv = SafePrefixInvariant(model)
    return z3.And(
        *env.constraints,
        env.phase.origin_time == z.t,
        inv.formula(z),
        encode_step(z, zp, model, env),
        z3.Not(expected),
    )


def _p2_observe_only_counterexample(model: BoundModel) -> z3.BoolRef:
    z = declare_state("conf.p2.z", model)
    zp = declare_state("conf.p2.zp", model)
    inv = SafePrefixInvariant(model)
    nonledger_change = z3.Or(
        zp.t != z.t,
        zp.mode_hi != z.mode_hi,
        _job_changed_except_phase_owned(z, zp),
        *(zp.budgets[name] != z.budgets[name] for name in z.budgets),
        *(zp.eta[name] != z.eta[name] for name in z.eta),
        zp.frontier.selected_slot != z.frontier.selected_slot,
        zp.frontier.running != z.frontier.running,
    )
    return z3.And(inv.formula(z), encode_p2_deadline_observe(z, zp, model), nonledger_change)


def _p3_snapshot_counterexample(model: BoundModel) -> z3.BoolRef:
    z = declare_state("conf.p3.z", model)
    zp = declare_state("conf.p3.zp", model)
    env = declare_environment("conf.p3.env", model, release_count=1)
    inv = SafePrefixInvariant(model)
    bad: list[z3.BoolRef] = []
    for task in model.tasks:
        due = z3.And(z.t % task.period == 0, z.eta[task.name] == task.period)
        slot = 0 if task.criticality == "HI" else 1
        post = zp.jobs[(task.name, slot)]
        expected_budget = (
            z.budgets[task.name]
            if task.criticality == "HI"
            else z3.If(z.mode_hi, int(task.degraded_cost or task.c_lo), z.budgets[task.name])
        )
        demand = env.actual_demands[(task.name, 0)]
        expected_abnormal = demand > task.c_lo if task.criticality == "HI" else z3.BoolVal(False)
        bad.append(z3.And(due, z3.Or(
            post.release_time != z.t,
            post.absolute_deadline != z.t + task.deadline,
            post.budget_at_release != expected_budget,
            post.actual_demand != demand,
            post.classification_abnormal != expected_abnormal,
        )))
    return z3.And(
        *env.constraints,
        env.phase.origin_time == z.t,
        inv.formula(z),
        encode_p3_arrival_freeze(z, zp, model, env),
        z3.Or(*bad),
    )


def _p7_time_service_counterexample(model: BoundModel) -> z3.BoolRef:
    z = declare_state("conf.p7.z", model)
    zp = declare_state("conf.p7.zp", model)
    inv = SafePrefixInvariant(model)
    deltas = [
        zp.jobs[key].executed_service - job.executed_service
        for key, job in z.jobs.items()
    ]
    expected_service = z3.If(z.frontier.selected_slot >= 0, 1, 0)
    return z3.And(
        inv.formula(z),
        encode_p7_time_and_service(z, zp, model),
        z3.Or(
            zp.t != z.t + 1,
            z3.Sum(*deltas) != expected_service,
        ),
    )


def _hi_miss_reflection_counterexample(model: BoundModel) -> z3.BoolRef:
    z = declare_state("conf.miss.z", model)
    zp = declare_state("conf.miss.zp", model)
    inv = SafePrefixInvariant(model)
    bad_jobs: list[z3.BoolRef] = []
    for task in model.hi_tasks:
        job = z.jobs[(task.name, 0)]
        bad_jobs.append(z3.And(
            job.present,
            job.absolute_deadline == z.t,
            job.executed_service < job.effective_demand,
        ))
    return z3.And(
        inv.formula(z),
        z.hi_miss_ledger == 0,
        z3.Or(*bad_jobs),
        encode_p2_deadline_observe(z, zp, model),
        zp.hi_miss_ledger != 1,
    )


def prove_universal_conformance(
    model: BoundModel,
    *,
    source_root: Path,
    timeout_ms: int = 120_000,
) -> ConformanceProof:
    try:
        theorem_rows = _theorem_receipts(source_root)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return ConformanceProof("FAIL", (), (), f"RUNTIME_THEOREM_REVERIFY_FAILED:{exc}")

    formulas = (
        ("V9_1_PHASE_ORDER_TOTALITY", _phase_order_counterexample(model)),
        ("V9_1_P2_OBSERVE_ONLY", _p2_observe_only_counterexample(model)),
        ("V9_1_P3_RELEASE_SNAPSHOT", _p3_snapshot_counterexample(model)),
        ("V9_1_P7_TIME_AND_SERVICE", _p7_time_service_counterexample(model)),
        ("V9_1_FIRST_HI_MISS_REFLECTION", _hi_miss_reflection_counterexample(model)),
    )
    solver_rows = tuple(
        solve_formula(obligation_id, formula, timeout_ms=timeout_ms)
        for obligation_id, formula in formulas
    )
    non_unsat = next((row for row in solver_rows if row.result != "UNSAT"), None)
    if non_unsat is not None:
        return ConformanceProof(
            "FAIL" if non_unsat.result == "SAT" else "UNRESOLVED",
            theorem_rows,
            solver_rows,
            f"CONFORMANCE_COUNTEREXAMPLE_{non_unsat.result}:{non_unsat.obligation_id}",
        )
    return ConformanceProof("PASS", theorem_rows, solver_rows)


__all__ = ["ConformanceProof", "RUNTIME_THEOREMS", "prove_universal_conformance"]
