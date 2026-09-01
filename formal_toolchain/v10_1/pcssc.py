"""Policy-Constrained Single-Switch Certificate (PCSSC), V10.14 revision.

The fast routes remain BASE/V10.11 pointwise/V10.12 case-consistent.  Only
V10.12 LO-entry canonical cases may be refined by V10.13 joint carry/future
phase coupling.  V10.14 additionally refines a V10.12-unresolved PRE_HI case by
keeping the exact target-release joint phase fixed across that phase subcase's
response recurrence, then aggregating the finite per-phase completion bounds.
All refinements remain finite integer arithmetic; Event Graph search and full
execution-history BMC remain outside the PASS dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from math import ceil
from typing import Any, Iterable, Mapping

from .kernel.carry_in import derive_protected_priority_prefix
from .completion_certificates import (
    BASE_COMPLETION_SOURCE,
    PCSSC_COMPLETION_SOURCE,
    PCSSC_CASE_COMPLETION_THEOREM,
    PCSSC_CONDITIONED_CARRY_COMPLETION_THEOREM,
    PCSSC_REFINED_CASE_COMPLETION_THEOREM_V10_14,
    PCSSC_POINTWISE_COMPLETION_THEOREM,
    CertifiedCompletionBound,
)
from .constants import (
    TARGET_PROVED_PCSSC,
    TARGET_PROVED_PCSSC_CASE_CONSISTENT,
    TARGET_PROVED_PCSSC_CASE_CONDITIONED_CARRY,
    TARGET_PROVED_PCSSC_REFINED_CASES_V10_14,
)
from .base_section4_1 import paper_c_lo_bound
from .carry_in_envelope import (
    CarryInEnvelopeUnresolved,
    CarryTaskSpec,
    fixed_phase_lo_entry_backlog,
    fixed_phase_pre_hi_carry,
    fixed_phase_post_switch_future_work,
    fixed_phase_pre_hi_interference,
    joint_phase_pre_hi_interference,
    target_release_joint_phase_orbit,
    target_release_joint_phase_parameters,
    target_release_joint_phases_at_q,
    phase_relaxed_lo_entry_carry,
    phase_relaxed_single_switch_carry,
)
from .controller_macro import (
    BudgetInterval,
    ControllerMacroPath,
    candidate_controller_times,
    controller_phase_residues,
    required_policy_read_features,
)
from .kernel.symbolic_state import BoundModel, TaskBound
from .periodic_release import compatible_release_phases


@dataclass(frozen=True, slots=True)
class SwitchCell:
    kind: str  # PRE_HI | LO_NO_SWITCH | LO_SWITCH
    lower: int | None = None
    upper: int | None = None

    @property
    def id(self) -> str:
        if self.kind != "LO_SWITCH":
            return self.kind
        return f"LO_SWITCH[{self.lower},{self.upper}]"


@dataclass(frozen=True, slots=True)
class MacroCell:
    lower: int
    upper_exclusive: int

    @property
    def upper(self) -> int:
        return self.upper_exclusive - 1


@dataclass(frozen=True, slots=True)
class CaseKey:
    theta: int
    switch: SwitchCell
    target_classification: str
    canonical_deadline: int

    @property
    def id(self) -> str:
        return (
            f"THETA_{self.theta}__{self.switch.id}__"
            f"TARGET_{self.target_classification}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.id,
            "theta": int(self.theta),
            "switch_kind": self.switch.kind,
            "switch_lower": self.switch.lower,
            "switch_upper": self.switch.upper,
            "target_classification": self.target_classification,
            "canonical_deadline": int(self.canonical_deadline),
        }


@dataclass(frozen=True, slots=True)
class TargetCertificate:
    target: str
    status: str
    response_bound: int | None
    failure_code: str | None
    receipts: tuple[dict[str, Any], ...]
    conservatism_ledger: tuple[dict[str, Any], ...]
    tested_horizons: tuple[dict[str, Any], ...]
    terminal_route: str | None = None
    completion_theorem_basis: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "status": self.status,
            "response_bound": self.response_bound,
            "failure_code": self.failure_code,
            "terminal_route": self.terminal_route,
            "completion_theorem_basis": self.completion_theorem_basis,
            "receipts": list(self.receipts),
            "conservatism_ledger": list(self.conservatism_ledger),
            "tested_horizons": list(self.tested_horizons),
        }


class PCSSCUnresolved(RuntimeError):
    pass


def _feature_basis_id(feature: str) -> str:
    if feature.startswith("budget::"):
        return f"FULL_LEGAL_BUDGET_FEATURE_DOMAIN_SOUND::{feature.split('::', 1)[1]}"
    if feature == "controller_timestamp":
        return "CONTROLLER_TIMESTAMP_CANDIDATE_DOMAIN_SOUND"
    return f"FULL_LEGAL_NUMERIC_FEATURE_DOMAIN_SOUND::{feature}"


def _require_feature_basis(path: ControllerMacroPath, feature: str) -> str:
    expected = _feature_basis_id(feature)
    matches = [
        row for row in path.receipts
        if row.get("obligation_id") == expected and row.get("status") == "PASS"
    ]
    if len(matches) != 1:
        raise PCSSCUnresolved(f"FEATURE_TRANSFER_BASIS_MISSING:{feature}:{expected}")
    return expected


def _controller_prefix_coverage_receipt(
    model: BoundModel,
    target: TaskBound,
    path: ControllerMacroPath,
    horizon: int,
) -> dict[str, Any]:
    """Machine-check the forward controller path prefix actually used at R.

    Only periodic candidate timestamps strictly below R and precomputed forward
    budget boxes are inspected.  No W(R), target completion, terminal safety or
    action after R participates in this check.
    """

    thetas = controller_phase_residues(target.period, model.agent_period)
    required_depth = max(
        (len(candidate_controller_times(theta, model.agent_period, horizon)) for theta in thetas),
        default=0,
    )
    available_depth = len(path.boxes) - 1
    if available_depth < required_depth:
        raise PCSSCUnresolved(
            f"CONTROLLER_PREFIX_COVERAGE_UNRESOLVED:{target.name}:R={horizon}:"
            f"required={required_depth}:available={available_depth}"
        )
    for depth, box in enumerate(path.boxes[: required_depth + 1]):
        for task in model.tasks:
            interval = box.get(task.name)
            if interval is None:
                raise PCSSCUnresolved(
                    f"CONTROLLER_PREFIX_COVERAGE_UNRESOLVED:{target.name}:"
                    f"MISSING_BUDGET:{depth}:{task.name}"
                )
            if (
                int(interval.lower) < int(task.budget_floor)
                or int(interval.upper) > int(task.budget_upper)
                or int(interval.lower) > int(interval.upper)
            ):
                raise PCSSCUnresolved(
                    f"CONTROLLER_PREFIX_COVERAGE_UNRESOLVED:{target.name}:"
                    f"BAD_BOX:{depth}:{task.name}"
                )
    return {
        "obligation_id": f"CONTROLLER_PATH_PREFIX_COVERAGE::{target.name},R={horizon}",
        "status": "PASS",
        "prefix_closed": True,
        "future_independent": True,
        "horizon_consistent": True,
        "required_controller_depth": int(required_depth),
        "available_controller_depth": int(available_depth),
        "theta_cells": list(thetas),
        "construction": (
            "forward-only controller successor hulls; projection uses only candidate "
            "timestamps c_k<R and never prunes by W(R), target completion or future actions"
        ),
    }



def _weight_at_release(
    release: int, cells: tuple[MacroCell, ...], weights: tuple[int, ...]
) -> int:
    for cell, weight in zip(cells, weights):
        if int(cell.lower) <= int(release) < int(cell.upper_exclusive):
            return int(weight)
    raise PCSSCUnresolved(f"EXACT_PERIODIC_RELEASE_OUTSIDE_MACRO_CELLS:{release}")



@lru_cache(maxsize=131072)
def _exact_periodic_phase_workload_cached(
    target_period: int,
    task_period: int,
    controller_period: int,
    theta: int,
    horizon: int,
    cells_key: tuple[tuple[int, int], ...],
    weights: tuple[int, ...],
    raw_carry_cap: int,
    completion_bound: int,
) -> tuple[int, int, int, int]:
    """Return max(total, phase, carry, future) for one hp task.

    This cache contains no solver state.  It is a direct enumeration of the
    finite exact-periodic phase orbit compatible with the fixed controller
    phase.  Cross-task phases are intentionally de-coupled later; that only
    adds combinations and is recorded in the conservatism ledger.
    """

    cells = tuple(MacroCell(lo, hi) for lo, hi in cells_key)
    try:
        phases = compatible_release_phases(
            int(target_period), int(task_period), int(controller_period), int(theta)
        )
    except ValueError as exc:
        raise PCSSCUnresolved(str(exc)) from exc
    best_total = -1
    best_phase = 0
    best_carry = 0
    best_future = 0
    for phase in phases:
        if phase == 0:
            carry = 0
        else:
            age = int(task_period) - int(phase)
            residual_time = int(completion_bound) - age
            carry = 0 if residual_time <= 0 else min(int(raw_carry_cap), int(residual_time))
        future = 0
        release = int(phase)
        while release < int(horizon):
            future += _weight_at_release(release, cells, weights)
            release += int(task_period)
        total = carry + future
        if total > best_total:
            best_total = total
            best_phase = int(phase)
            best_carry = int(carry)
            best_future = int(future)
    if best_total < 0:
        raise PCSSCUnresolved("EXACT_PERIODIC_PHASE_PROFILE_EMPTY")
    return best_total, best_phase, best_carry, best_future


def _exact_periodic_task_workload(
    target: TaskBound,
    task: TaskBound,
    *,
    theta: int,
    horizon: int,
    cells: tuple[MacroCell, ...],
    weights: tuple[int, ...],
    switch_kind: str,
    protected: set[str],
    protected_response_by_task: dict[str, int],
    controller_period: int,
) -> tuple[int, dict[str, int]]:
    raw_carry = _carry_in_bound(task, switch_kind, protected)
    if task.criticality == "LO":
        response = protected_response_by_task.get(task.name)
        if response is None:
            raise PCSSCUnresolved(f"REACHABLE_LO_CARRY_IN_UNRESOLVED:{task.name}")
        completion_bound = int(response)
    else:
        completion_bound = int(task.deadline)
        response = protected_response_by_task.get(task.name)
        if response is not None:
            completion_bound = min(completion_bound, int(response))
    key = tuple((int(cell.lower), int(cell.upper_exclusive)) for cell in cells)
    total, phase, carry, future = _exact_periodic_phase_workload_cached(
        int(target.period), int(task.period), int(controller_period), int(theta),
        int(horizon), key, tuple(int(v) for v in weights), int(raw_carry),
        int(completion_bound),
    )
    return int(total), {
        "release_phase": int(phase),
        "carry_in": int(carry),
        "future_interference": int(future),
        "completion_bound": int(completion_bound),
    }


def _exact_periodic_task_future_only(
    target: TaskBound,
    task: TaskBound,
    *,
    theta: int,
    horizon: int,
    cells: tuple[MacroCell, ...],
    weights: tuple[int, ...],
    controller_period: int,
) -> tuple[int, dict[str, int]]:
    """Maximum non-negative-time workload; excludes all carry-in by construction."""
    key = tuple((int(cell.lower), int(cell.upper_exclusive)) for cell in cells)
    total, phase, carry, future = _exact_periodic_phase_workload_cached(
        int(target.period), int(task.period), int(controller_period), int(theta),
        int(horizon), key, tuple(int(v) for v in weights), 0, 0,
    )
    if int(carry) != 0 or int(total) != int(future):
        raise PCSSCUnresolved("FUTURE_ONLY_PERIODIC_PROFILE_HAS_CARRY")
    return int(future), {
        "release_phase": int(phase),
        "future_interference": int(future),
    }


def _carry_task_specs(
    hp_tasks: tuple[TaskBound, ...], path: ControllerMacroPath
) -> tuple[CarryTaskSpec, ...]:
    start = path.boxes[0]
    rows: list[CarryTaskSpec] = []
    for task in hp_tasks:
        if task.criticality == "LO":
            pre = _primary_cap(task, start[task.name])
            post = _degraded_cap(task)
        else:
            pre = _hi_normal_cap(task)
            post = _hi_high_cap(task)
        rows.append(CarryTaskSpec(
            name=task.name, criticality=task.criticality, period=int(task.period),
            pre_switch_cap=int(pre), post_switch_cap=int(post),
        ))
    return tuple(rows)


def _aggregate_carry_bound(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    *,
    theta: int,
    switch_kind: str,
) -> tuple[int, dict[str, Any]]:
    specs = _carry_task_specs(hp_tasks, path)
    try:
        if switch_kind == "PRE_HI":
            value, details = phase_relaxed_single_switch_carry(
                int(target.period), int(model.agent_period), int(theta), specs
            )
            basis = "SINGLE_SWITCH_AGGREGATE_BACKLOG_WITH_PER_TASK_PHASE_RELAXATION"
        else:
            value, details = phase_relaxed_lo_entry_carry(
                int(target.period), int(model.agent_period), int(theta), specs
            )
            basis = "LO_ENTRY_AGGREGATE_BACKLOG_WITH_PER_TASK_PHASE_RELAXATION"
    except CarryInEnvelopeUnresolved as exc:
        raise PCSSCUnresolved(str(exc)) from exc
    return int(value), {
        "basis": basis,
        "carry_in": int(value),
        **details,
    }


def _joint_pre_hi_interference_if_protected(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    protected_response_by_task: dict[str, int],
    *,
    theta: int,
    horizon: int,
) -> tuple[int, dict[str, Any]] | None:
    if any(task.name not in protected_response_by_task for task in hp_tasks):
        return None
    completion = tuple(int(protected_response_by_task[task.name]) for task in hp_tasks)
    if any(bound <= 0 or bound > int(task.period)
           for task, bound in zip(hp_tasks, completion)):
        return None
    specs = _carry_task_specs(hp_tasks, path)
    try:
        value, details = joint_phase_pre_hi_interference(
            int(target.period), int(model.agent_period), int(theta), int(horizon),
            specs, completion,
        )
    except CarryInEnvelopeUnresolved as exc:
        raise PCSSCUnresolved(str(exc)) from exc
    return int(value), {
        "basis": "JOINT_EXACT_PERIODIC_SINGLE_SWITCH_AGGREGATE_BACKLOG_PLUS_FUTURE",
        **details,
    }


def _budget_depth_at_release(u: int, controller_times: tuple[int, ...]) -> int:
    # A release exactly at c_k is P3 before P5 and therefore sees the pre-k
    # budget.  Only controller candidates strictly before u affect B_rel.
    return sum(1 for value in controller_times if value < int(u))


def _primary_cap(task: TaskBound, interval: BudgetInterval) -> int:
    return min(int(task.actual_demand_upper), int(interval.upper) + 1)


def _degraded_cap(task: TaskBound) -> int:
    return min(int(task.actual_demand_upper), int(task.degraded_cost or task.c_lo))


def _hi_normal_cap(task: TaskBound) -> int:
    return min(int(task.actual_demand_upper), int(task.c_lo))


def _hi_high_cap(task: TaskBound) -> int:
    return int(task.actual_demand_upper)


def _macro_cells(
    horizon: int,
    controller_times: tuple[int, ...],
    switch: SwitchCell,
) -> tuple[MacroCell, ...]:
    boundaries = {0, int(horizon)}
    for value in controller_times:
        if 0 < value < horizon:
            boundaries.add(int(value))
        if 0 < value + 1 < horizon:
            boundaries.add(int(value) + 1)
    if switch.kind == "LO_SWITCH":
        assert switch.lower is not None and switch.upper is not None
        for value in (int(switch.lower), int(switch.lower) + 1,
                      int(switch.upper), int(switch.upper) + 1):
            if 0 < value < horizon:
                boundaries.add(value)
    ordered = sorted(boundaries)
    return tuple(
        MacroCell(ordered[index], ordered[index + 1])
        for index in range(len(ordered) - 1)
        if ordered[index] < ordered[index + 1]
    )


def _switch_breakpoints(
    horizon: int,
    controller_times: tuple[int, ...],
    hp_tasks: tuple[TaskBound, ...],
) -> tuple[int, ...]:
    """Precision-only scalar breakpoints; missing points cannot make the proof unsound.

    The resulting switch cells still cover every integer s.  Period/controller
    translated multiples reduce switch-side ambiguity for periodic releases
    inside a relaxed Sigma.
    """
    if horizon <= 1:
        return ()
    anchors = {0, int(horizon)}
    for c in controller_times:
        anchors.update((int(c), int(c) + 1))
    points = {1, int(horizon) - 1}
    for c in controller_times:
        for value in (c - 1, c, c + 1):
            if 1 <= value < horizon:
                points.add(int(value))
    for task in hp_tasks:
        T = int(task.period)
        if T <= 0:
            continue
        for anchor in anchors:
            # Include translated period multiples on both sides of each
            # controller/response boundary.  These are precision breakpoints
            # for periodic release/switch-side changes; the Sigma cover is
            # sound even if the set is later coarsened.
            k_min = (1 - int(anchor) - (T - 1)) // T
            k_max = (int(horizon) - 1 - int(anchor)) // T
            for k in range(k_min, k_max + 1):
                value = int(anchor) + k * T
                if 1 <= value < horizon:
                    points.add(int(value))
    return tuple(sorted(points))


def _switch_cells(
    horizon: int,
    controller_times: tuple[int, ...],
    hp_tasks: tuple[TaskBound, ...],
) -> tuple[SwitchCell, ...]:
    if horizon <= 0:
        return ()
    rows: list[SwitchCell] = [SwitchCell("LO_SWITCH", 0, 0)]
    if horizon == 1:
        return tuple(rows)
    points = _switch_breakpoints(horizon, controller_times, hp_tasks)
    cursor = 1
    for point in points:
        if point < cursor:
            continue
        if cursor < point:
            rows.append(SwitchCell("LO_SWITCH", cursor, point - 1))
        rows.append(SwitchCell("LO_SWITCH", point, point))
        cursor = point + 1
    if cursor <= horizon - 1:
        rows.append(SwitchCell("LO_SWITCH", cursor, horizon - 1))
    return tuple(rows)


def _weight_for_cell(
    task: TaskBound,
    cell: MacroCell,
    switch: SwitchCell,
    controller_times: tuple[int, ...],
    path: ControllerMacroPath,
) -> int:
    u = int(cell.lower)
    if task.criticality == "LO":
        depth = _budget_depth_at_release(u, controller_times)
        if depth >= len(path.boxes):
            raise PCSSCUnresolved("CONTROLLER_PREFIX_DEPTH_NOT_COVERED")
        primary = _primary_cap(task, path.boxes[depth][task.name])
        paper_c_lo = paper_c_lo_bound(task)
        if primary > paper_c_lo:
            raise PCSSCUnresolved(
                f"PRIMARY_EFFECTIVE_SERVICE_LE_PAPER_C_LO_FAILED:{task.name}"
            )
        degraded = _degraded_cap(task)
        if switch.kind == "PRE_HI":
            return degraded
        if switch.kind == "LO_NO_SWITCH":
            return primary
        assert switch.lower is not None and switch.upper is not None
        # LO at u=s is primary.  A Sigma interval that straddles the relation is
        # conservatively assigned max(primary,degraded).
        if cell.upper <= int(switch.lower):
            return primary
        if cell.lower > int(switch.upper):
            return degraded
        return max(primary, degraded)

    normal = _hi_normal_cap(task)
    high = _hi_high_cap(task)
    if switch.kind == "PRE_HI":
        return high
    if switch.kind == "LO_NO_SWITCH":
        return normal
    assert switch.lower is not None and switch.upper is not None
    # HI at u=s may use C_HI.  Thus normal is strict u<s, high is u>=s.
    if cell.upper < int(switch.lower):
        return normal
    if cell.lower >= int(switch.upper):
        return high
    return max(normal, high)


def _carry_in_bound(task: TaskBound, switch_kind: str, protected: set[str]) -> int:
    if task.criticality == "HI":
        # In a LO-entry first-bad start, a live HI job cannot have been abnormal
        # without already forcing HI mode.  PRE_HI permits the full HI cap.
        return _hi_high_cap(task) if switch_kind == "PRE_HI" else _hi_normal_cap(task)
    if task.name not in protected:
        raise PCSSCUnresolved(f"REACHABLE_LO_CARRY_IN_UNRESOLVED:{task.name}")
    return int(task.actual_demand_upper)


def _target_cap(target: TaskBound, classification: str) -> int:
    if classification == "NORMAL":
        return _hi_normal_cap(target)
    if classification == "ABNORMAL":
        return _hi_high_cap(target)
    raise ValueError("UNKNOWN_TARGET_CLASSIFICATION")


def _classification_possible(target: TaskBound, classification: str) -> bool:
    if classification == "NORMAL":
        return int(target.actual_demand_min) <= int(target.c_lo)
    return int(target.actual_demand_upper) > int(target.c_lo)


def _valid_target_classes(target: TaskBound, switch: SwitchCell) -> tuple[str, ...]:
    candidates: tuple[str, ...]
    if switch.kind == "LO_NO_SWITCH":
        candidates = ("NORMAL",)
    elif switch.kind == "LO_SWITCH" and switch.lower == 0 and switch.upper == 0:
        candidates = ("NORMAL", "ABNORMAL")
    elif switch.kind == "LO_SWITCH":
        # An abnormal target release at 0 itself forces s=0.
        candidates = ("NORMAL",)
    else:  # PRE_HI
        candidates = ("NORMAL", "ABNORMAL")
    return tuple(value for value in candidates if _classification_possible(target, value))


def _workload_case(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    protected: set[str],
    protected_response_by_task: dict[str, int],
    *,
    horizon: int,
    theta: int,
    switch: SwitchCell,
    classification: str,
) -> tuple[int, dict[str, Any]]:
    controller_times = candidate_controller_times(theta, model.agent_period, horizon)
    cells = _macro_cells(horizon, controller_times, switch)
    target_work = _target_cap(target, classification)

    # PRE_HI has constant post-switch service weights.  If every hp task already
    # has a single-job completion envelope, retain the common target-release
    # index across *all* hp tasks and jointly maximize carry plus future work.
    # This removes the old impossible sum of independently worst carry phases.
    if switch.kind == "PRE_HI":
        joint = _joint_pre_hi_interference_if_protected(
            model, target, hp_tasks, path, protected_response_by_task,
            theta=theta, horizon=horizon,
        )
        if joint is not None:
            hp_work, aggregate_details = joint
            return int(target_work + hp_work), {
                "theta": int(theta),
                "switch_profile": switch.id,
                "target_classification": classification,
                "target_demand": int(target_work),
                "controller_times": list(controller_times),
                "carry_in_model": aggregate_details,
                "interference_model": "JOINT_EXACT_PERIODIC_PRE_HI",
                "interference": [],
            }

    carry, carry_details = _aggregate_carry_bound(
        model, target, hp_tasks, path, theta=theta, switch_kind=switch.kind
    )
    rows: list[dict[str, Any]] = []
    future_total = 0
    for task in hp_tasks:
        weights = tuple(
            _weight_for_cell(task, cell, switch, controller_times, path)
            for cell in cells
        )
        future, phase_details = _exact_periodic_task_future_only(
            target, task, theta=theta, horizon=horizon, cells=cells, weights=weights,
            controller_period=model.agent_period,
        )
        future_total += int(future)
        rows.append({
            "task": task.name,
            **phase_details,
            "weights": list(weights),
            "cells": [[cell.lower, cell.upper_exclusive] for cell in cells],
            "release_model": "EXACT_PERIODIC_PHASE_ZERO",
        })

    aggregate_total = int(carry) + int(future_total)

    # Where every LO carry already has a completion theorem, the previous
    # per-task phase-coupled bound is another independent upper bound.  Taking
    # the minimum cannot exclude a real execution and preserves any precision
    # that the aggregate envelope does not improve.
    legacy_total: int | None = 0
    legacy_rows: list[dict[str, Any]] = []
    for task in hp_tasks:
        if task.criticality == "LO" and task.name not in protected:
            legacy_total = None
            legacy_rows = []
            break
        weights = tuple(
            _weight_for_cell(task, cell, switch, controller_times, path)
            for cell in cells
        )
        task_work, phase_details = _exact_periodic_task_workload(
            target, task, theta=theta, horizon=horizon, cells=cells, weights=weights,
            switch_kind=switch.kind, protected=protected,
            protected_response_by_task=protected_response_by_task,
            controller_period=model.agent_period,
        )
        legacy_total += int(task_work)
        legacy_rows.append({"task": task.name, **phase_details})

    if legacy_total is not None and int(legacy_total) < int(aggregate_total):
        hp_work = int(legacy_total)
        chosen = "MIN_OF_AGGREGATE_AND_COMPLETION_PHASE_COUPLED:COMPLETION_PHASE_COUPLED"
    else:
        hp_work = int(aggregate_total)
        chosen = "MIN_OF_AGGREGATE_AND_COMPLETION_PHASE_COUPLED:AGGREGATE" if legacy_total is not None else "AGGREGATE"

    return int(target_work + hp_work), {
        "theta": int(theta),
        "switch_profile": switch.id,
        "target_classification": classification,
        "target_demand": int(target_work),
        "controller_times": list(controller_times),
        "carry_in_model": carry_details,
        "future_interference_total": int(future_total),
        "aggregate_hp_bound": int(aggregate_total),
        "completion_phase_coupled_hp_bound": None if legacy_total is None else int(legacy_total),
        "selected_hp_bound": chosen,
        "interference": rows,
        "completion_phase_coupled_diagnostic": legacy_rows,
    }


def _fixed_phase_future_only(
    task: TaskBound,
    *,
    phase: int,
    horizon: int,
    cells: tuple[MacroCell, ...],
    weights: tuple[int, ...],
) -> int:
    if not (0 <= int(phase) < int(task.period)):
        raise PCSSCUnresolved(f"EXACT_PERIODIC_PHASE_OUT_OF_RANGE:{task.name}:{phase}")
    total = 0
    release = int(phase)
    while release < int(horizon):
        total += _weight_at_release(int(release), cells, weights)
        release += int(task.period)
    return int(total)


def _case_conditioned_joint_phase_interference(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    *,
    horizon: int,
    theta: int,
    switch: SwitchCell,
) -> tuple[int, dict[str, Any]]:
    """V10.13 LO-entry carry/future bound with one shared release index.

    V10.12 may independently maximize the LO-entry carry phase and each
    higher-priority future-interference phase.  Exact periodic phase-zero
    releases imply that all of those phases come from one target-release index.
    This routine enumerates that finite joint orbit and maximizes
    ``carry(q) + future(q)`` without reconstructing an Event Graph.
    """
    if switch.kind not in {"LO_NO_SWITCH", "LO_SWITCH"}:
        raise PCSSCUnresolved("CASE_CONDITIONED_CARRY_REQUIRES_LO_ENTRY")

    controller_times = candidate_controller_times(theta, model.agent_period, horizon)
    cells = _macro_cells(horizon, controller_times, switch)
    specs = _carry_task_specs(hp_tasks, path)
    weights_by_task = tuple(
        tuple(
            _weight_for_cell(task, cell, switch, controller_times, path)
            for cell in cells
        )
        for task in hp_tasks
    )
    orbit = target_release_joint_phase_orbit(
        int(target.period), int(model.agent_period), int(theta), specs
    )
    maximum = -1
    witness_q = 0
    witness_carry = 0
    witness_future = 0
    witness_phases: tuple[int, ...] = ()
    witness_carry_details: dict[str, int] = {}
    witness_future_rows: list[dict[str, int | str]] = []

    for q, phases in orbit:
        carry, carry_details = fixed_phase_lo_entry_backlog(specs, phases)
        future = 0
        future_rows: list[dict[str, int | str]] = []
        for task, phase, weights in zip(hp_tasks, phases, weights_by_task):
            task_future = _fixed_phase_future_only(
                task, phase=int(phase), horizon=int(horizon), cells=cells, weights=weights
            )
            future += int(task_future)
            future_rows.append({
                "task": task.name,
                "release_phase": int(phase),
                "future_interference": int(task_future),
            })
        value = int(carry) + int(future)
        if value > maximum:
            maximum = int(value)
            witness_q = int(q)
            witness_carry = int(carry)
            witness_future = int(future)
            witness_phases = tuple(int(value) for value in phases)
            witness_carry_details = dict(carry_details)
            witness_future_rows = future_rows

    if maximum < 0:
        raise PCSSCUnresolved(f"CASE_CONDITIONED_JOINT_PHASE_ORBIT_EMPTY:{target.name}")
    return int(maximum), {
        "basis": "V10_13_CASE_CONDITIONED_JOINT_TARGET_RELEASE_PHASE",
        "theta": int(theta),
        "switch_profile": switch.id,
        "joint_phase_cycle": len(orbit),
        "witness_q": int(witness_q),
        "witness_phases": list(witness_phases),
        "carry_in": int(witness_carry),
        "future": int(witness_future),
        "joint_hp_bound": int(maximum),
        "carry_details": witness_carry_details,
        "future_rows": witness_future_rows,
        "same_target_release_index_for_carry_and_future": True,
    }


def _workload_case_conditioned_v10_13(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    protected: set[str],
    protected_response_by_task: dict[str, int],
    *,
    horizon: int,
    theta: int,
    switch: SwitchCell,
    classification: str,
) -> tuple[int, dict[str, Any]]:
    legacy, legacy_details = _workload_case(
        model, target, hp_tasks, path, protected, protected_response_by_task,
        horizon=horizon, theta=theta, switch=switch, classification=classification,
    )
    if switch.kind not in {"LO_NO_SWITCH", "LO_SWITCH"}:
        return int(legacy), {
            **legacy_details,
            "v10_13_refinement_applicable": False,
        }
    joint_hp, joint_details = _case_conditioned_joint_phase_interference(
        model, target, hp_tasks, path,
        horizon=horizon, theta=theta, switch=switch,
    )
    target_work = _target_cap(target, classification)
    refined = int(target_work) + int(joint_hp)
    if refined <= int(legacy):
        return int(refined), {
            "theta": int(theta),
            "switch_profile": switch.id,
            "target_classification": classification,
            "target_demand": int(target_work),
            "selected_hp_bound": "V10_13_CASE_CONDITIONED_JOINT_PHASE",
            "case_conditioned_joint_phase": joint_details,
            "v10_12_workload_upper": int(legacy),
            "v10_13_workload_upper": int(refined),
            "sound_intersection": "MIN_OF_TWO_INDEPENDENT_SOUND_UPPER_BOUNDS",
        }
    return int(legacy), {
        **legacy_details,
        "v10_13_refinement_applicable": True,
        "case_conditioned_joint_phase": joint_details,
        "v10_12_workload_upper": int(legacy),
        "v10_13_workload_upper": int(refined),
        "sound_intersection": "MIN_OF_TWO_INDEPENDENT_SOUND_UPPER_BOUNDS",
    }


def _max_workload_at_horizon(
    model: BoundModel,
    target: TaskBound,
    path: ControllerMacroPath,
    protected: set[str],
    protected_response_by_task: dict[str, int],
    horizon: int,
) -> tuple[int, dict[str, Any], int]:
    target_index = next(index for index, task in enumerate(model.tasks) if task.name == target.name)
    hp_tasks = tuple(model.tasks[:target_index])
    thetas = controller_phase_residues(target.period, model.agent_period)
    maximum = -1
    argmax: dict[str, Any] = {}
    switch_case_count = 0
    for theta in thetas:
        controller_times = candidate_controller_times(theta, model.agent_period, horizon)
        profiles: list[SwitchCell] = [SwitchCell("PRE_HI"), SwitchCell("LO_NO_SWITCH")]
        profiles.extend(_switch_cells(horizon, controller_times, hp_tasks))
        for switch in profiles:
            for classification in _valid_target_classes(target, switch):
                switch_case_count += 1
                value, details = _workload_case(
                    model, target, hp_tasks, path, protected, protected_response_by_task,
                    horizon=horizon, theta=theta, switch=switch,
                    classification=classification,
                )
                if value > maximum:
                    maximum = value
                    argmax = details
    if maximum < 0:
        raise PCSSCUnresolved(f"CERTIFICATE_START_COVERAGE_EMPTY:{target.name}")
    return int(maximum), argmax, switch_case_count



def _canonical_switch_partition_receipt(
    target: TaskBound,
    theta: int,
    cells: tuple[SwitchCell, ...],
) -> dict[str, Any]:
    deadline = int(target.deadline)
    expected = 0
    for cell in cells:
        if cell.kind != "LO_SWITCH" or cell.lower is None or cell.upper is None:
            raise PCSSCUnresolved(
                f"DEADLINE_CANONICAL_SWITCH_PARTITION_INVALID:{target.name}:theta={theta}"
            )
        if int(cell.lower) != expected or int(cell.upper) < int(cell.lower):
            raise PCSSCUnresolved(
                f"DEADLINE_CANONICAL_SWITCH_PARTITION_GAP:{target.name}:theta={theta}:"
                f"expected={expected}:cell={cell.id}"
            )
        expected = int(cell.upper) + 1
    if deadline > 0 and expected != deadline:
        raise PCSSCUnresolved(
            f"DEADLINE_CANONICAL_SWITCH_PARTITION_INCOMPLETE:{target.name}:theta={theta}:"
            f"covered_end={expected}:deadline={deadline}"
        )
    return {
        "obligation_id": f"DEADLINE_CANONICAL_SWITCH_PARTITION::{target.name},theta={theta}",
        "status": "PASS",
        "canonical_deadline": deadline,
        "switch_cells": [cell.id for cell in cells],
        "lo_switch_integer_domain": [0, max(-1, deadline - 1)],
        "complete": True,
        "disjoint": True,
        "constructed_once_at_deadline": True,
    }


def _deadline_canonical_case_domain(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
) -> tuple[tuple[CaseKey, ...], tuple[dict[str, Any], ...], str]:
    deadline = int(target.deadline)
    rows: list[CaseKey] = []
    partition_receipts: list[dict[str, Any]] = []
    for theta in controller_phase_residues(target.period, model.agent_period):
        controller_times = candidate_controller_times(theta, model.agent_period, deadline)
        switch_cells = _switch_cells(deadline, controller_times, hp_tasks)
        partition_receipts.append(
            _canonical_switch_partition_receipt(target, int(theta), switch_cells)
        )
        profiles = (SwitchCell("PRE_HI"), SwitchCell("LO_NO_SWITCH"), *switch_cells)
        for switch in profiles:
            for classification in _valid_target_classes(target, switch):
                rows.append(CaseKey(
                    theta=int(theta),
                    switch=switch,
                    target_classification=str(classification),
                    canonical_deadline=deadline,
                ))
    rows.sort(key=lambda case: (
        int(case.theta),
        case.switch.kind,
        -1 if case.switch.lower is None else int(case.switch.lower),
        -1 if case.switch.upper is None else int(case.switch.upper),
        case.target_classification,
    ))
    if not rows:
        raise PCSSCUnresolved(f"CASE_CONSISTENT_CLASS_DOMAIN_EMPTY:{target.name}")
    ids = [case.id for case in rows]
    if len(ids) != len(set(ids)):
        raise PCSSCUnresolved(f"CASE_CONSISTENT_CLASS_DOMAIN_DUPLICATE:{target.name}")
    payload = [case.as_dict() for case in rows]
    domain_hash = sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return tuple(rows), tuple(partition_receipts), domain_hash


def _pointwise_postfix_search(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    protected: set[str],
    protected_response_by_task: dict[str, int],
) -> tuple[int | None, list[dict[str, Any]], list[dict[str, Any]], str | None]:
    tested: list[dict[str, Any]] = []
    proof_receipts: list[dict[str, Any]] = []
    deadline = int(target.deadline)
    R = _initial_horizon(target, hp_tasks, protected)
    seen: set[int] = set()
    while True:
        if R in seen:
            return None, tested, proof_receipts, f"POINTWISE_CANDIDATE_CYCLE:{target.name}:R={R}"
        seen.add(R)
        try:
            prefix_receipt = _controller_prefix_coverage_receipt(model, target, path, R)
            workload, argmax, case_count = _max_workload_at_horizon(
                model, target, path, protected, protected_response_by_task, R
            )
        except PCSSCUnresolved as exc:
            return None, tested, proof_receipts, str(exc)
        tested.append({
            "terminal": "POINTWISE",
            "R": int(R),
            "W": int(workload),
            "postfixed": bool(workload <= R),
            "maximizing_case": argmax,
            "joint_case_count": int(case_count),
        })
        proof_receipts.append(prefix_receipt)
        if workload <= R:
            proof_receipts.extend((
                {"obligation_id": f"WORKLOAD_DOMINANCE::{target.name}", "status": "PASS", "horizon": int(R)},
                {"obligation_id": f"POSTFIX_RESPONSE_CERTIFICATE::{target.name},R={R}", "status": "PASS", "W": int(workload), "R": int(R)},
                {"obligation_id": f"HI_TARGET_SAFE::{target.name}", "status": "PASS", "route": "PCSSC_POINTWISE"},
            ))
            return int(R), tested, proof_receipts, None
        if R == deadline:
            return None, tested, proof_receipts, "POLICY_SINGLE_SWITCH_POINTWISE_POSTFIX_UNRESOLVED"
        next_R = max(int(R) + 1, int(workload))
        R = deadline if next_R > deadline else next_R


def _case_postfix_search(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    protected: set[str],
    protected_response_by_task: dict[str, int],
    case: CaseKey,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    deadline = int(target.deadline)
    R = max(1, min(deadline, int(_target_cap(target, case.target_classification))))
    path_rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    while True:
        if R in seen:
            return None, path_rows, f"CASE_CANDIDATE_CYCLE:{case.id}:R={R}"
        seen.add(R)
        try:
            _controller_prefix_coverage_receipt(model, target, path, R)
            workload, details = _workload_case(
                model, target, hp_tasks, path, protected, protected_response_by_task,
                horizon=R, theta=case.theta, switch=case.switch,
                classification=case.target_classification,
            )
        except PCSSCUnresolved as exc:
            return None, path_rows, str(exc)
        path_rows.append({"R": int(R), "W": int(workload), "postfixed": bool(workload <= R)})
        if workload <= R:
            # Normative direct re-check.  The iteration above is only a candidate generator.
            try:
                prefix_receipt = _controller_prefix_coverage_receipt(model, target, path, R)
                final_workload, final_details = _workload_case(
                    model, target, hp_tasks, path, protected, protected_response_by_task,
                    horizon=R, theta=case.theta, switch=case.switch,
                    classification=case.target_classification,
                )
            except PCSSCUnresolved as exc:
                return None, path_rows, str(exc)
            if final_workload > R:
                return None, path_rows, f"CASE_DIRECT_POSTFIX_RECHECK_FAILED:{case.id}:R={R}:W={final_workload}"
            return {
                **case.as_dict(),
                "status": "PASS",
                "R": int(R),
                "W": int(final_workload),
                "postfixed": True,
                "candidate_path": path_rows,
                "controller_prefix_receipt": prefix_receipt,
                "final_case": final_details,
            }, path_rows, None
        if R == deadline:
            return None, path_rows, f"CASE_POSTFIX_NOT_FOUND_BY_DEADLINE:{case.id}:W={workload}:D={deadline}"
        next_R = max(int(R) + 1, int(workload))
        R = deadline if next_R > deadline else next_R


def _case_conditioned_postfix_search_v10_13(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    protected: set[str],
    protected_response_by_task: dict[str, int],
    case: CaseKey,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    if case.switch.kind not in {"LO_NO_SWITCH", "LO_SWITCH"}:
        return None, [], f"V10_13_CASE_CONDITIONED_CARRY_NOT_APPLICABLE:{case.id}"
    deadline = int(target.deadline)
    R = max(1, min(deadline, int(_target_cap(target, case.target_classification))))
    path_rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    while True:
        if R in seen:
            return None, path_rows, f"V10_13_CASE_CANDIDATE_CYCLE:{case.id}:R={R}"
        seen.add(R)
        try:
            _controller_prefix_coverage_receipt(model, target, path, R)
            workload, details = _workload_case_conditioned_v10_13(
                model, target, hp_tasks, path, protected, protected_response_by_task,
                horizon=R, theta=case.theta, switch=case.switch,
                classification=case.target_classification,
            )
        except PCSSCUnresolved as exc:
            return None, path_rows, str(exc)
        path_rows.append({
            "R": int(R),
            "W": int(workload),
            "postfixed": bool(workload <= R),
            "selected_hp_bound": details.get("selected_hp_bound"),
        })
        if workload <= R:
            try:
                prefix_receipt = _controller_prefix_coverage_receipt(model, target, path, R)
                final_workload, final_details = _workload_case_conditioned_v10_13(
                    model, target, hp_tasks, path, protected, protected_response_by_task,
                    horizon=R, theta=case.theta, switch=case.switch,
                    classification=case.target_classification,
                )
            except PCSSCUnresolved as exc:
                return None, path_rows, str(exc)
            if final_workload > R:
                return None, path_rows, (
                    f"V10_13_CASE_DIRECT_POSTFIX_RECHECK_FAILED:{case.id}:"
                    f"R={R}:W={final_workload}"
                )
            return {
                **case.as_dict(),
                "status": "PASS",
                "R": int(R),
                "W": int(final_workload),
                "postfixed": True,
                "candidate_path": path_rows,
                "controller_prefix_receipt": prefix_receipt,
                "final_case": final_details,
                "case_theorem_basis": "V10_13_CASE_CONDITIONED_CARRY_FUTURE",
            }, path_rows, None
        if R == deadline:
            return None, path_rows, (
                f"V10_13_CASE_POSTFIX_NOT_FOUND_BY_DEADLINE:{case.id}:"
                f"W={workload}:D={deadline}"
            )
        next_R = max(int(R) + 1, int(workload))
        R = deadline if next_R > deadline else next_R



def _pre_hi_phase_consistent_postfix_search_v10_14(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    protected_response_by_task: dict[str, int],
    case: CaseKey,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    """V10.14 PRE_HI refinement with one fixed joint release phase per recurrence.

    V10.12 already uses one target-release index to couple all HP phases at a
    *single* horizon, but then maximizes that index again when the response
    candidate changes.  A concrete target job has one immutable release index,
    so V10.14 enumerates the finite joint phase orbit and gives every phase
    subcase its own directly checked postfix before aggregating completion
    bounds.  The enumeration is streamed; no full orbit/history table is kept.
    """
    if case.switch.kind != "PRE_HI":
        return None, [], f"V10_14_PRE_HI_PHASE_CONSISTENT_NOT_APPLICABLE:{case.id}"

    specs = _carry_task_specs(hp_tasks, path)
    completion_bounds: tuple[int, ...] | None = None
    if all(task.name in protected_response_by_task for task in hp_tasks):
        candidate = tuple(
            int(protected_response_by_task[task.name]) for task in hp_tasks
        )
        if all(
            0 < int(bound) <= int(task.period)
            for task, bound in zip(hp_tasks, candidate)
        ):
            completion_bounds = candidate

    try:
        n0, q_step, cycle = target_release_joint_phase_parameters(
            int(target.period), int(model.agent_period), int(case.theta), specs
        )
    except CarryInEnvelopeUnresolved as exc:
        return None, [], str(exc)
    if int(cycle) <= 0:
        return None, [], f"V10_14_PRE_HI_PHASE_DOMAIN_EMPTY:{case.id}"

    deadline = int(target.deadline)
    target_work = int(_target_cap(target, case.target_classification))
    maximum_response = 0
    worst_q = -1
    worst_phases: tuple[int, ...] = ()
    worst_final_workload = 0
    worst_path: list[dict[str, Any]] = []
    max_candidate_steps = 0
    digest = sha256()

    for q in range(int(cycle)):
        phases = target_release_joint_phases_at_q(
            int(target.period), specs, n0=int(n0), q_step=int(q_step), q=int(q)
        )
        try:
            fixed_carry, carry_details = fixed_phase_pre_hi_carry(
                specs, phases, completion_bounds
            )
        except CarryInEnvelopeUnresolved as exc:
            return None, [], str(exc)

        R = max(1, min(deadline, target_work))
        seen: set[int] = set()
        q_path: list[dict[str, Any]] = []
        final_workload = -1

        while True:
            if R in seen:
                return None, q_path, (
                    f"V10_14_PRE_HI_PHASE_CANDIDATE_CYCLE:{case.id}:q={q}:R={R}"
                )
            seen.add(int(R))
            future = fixed_phase_post_switch_future_work(specs, phases, int(R))
            workload = int(target_work) + int(fixed_carry) + int(future)
            q_path.append({
                "R": int(R),
                "W": int(workload),
                "postfixed": bool(workload <= R),
                "carry_in": int(fixed_carry),
                "future": int(future),
            })

            if workload <= R:
                # Direct recheck of the same fixed-q formula.  The recurrence is
                # only a candidate generator; this equality is the certificate
                # check that supports PASS.
                final_future = fixed_phase_post_switch_future_work(
                    specs, phases, int(R)
                )
                final_workload = (
                    int(target_work) + int(fixed_carry) + int(final_future)
                )
                if final_workload > R:
                    return None, q_path, (
                        f"V10_14_PRE_HI_PHASE_DIRECT_POSTFIX_RECHECK_FAILED:"
                        f"{case.id}:q={q}:R={R}:W={final_workload}"
                    )
                break

            if R == deadline:
                return None, q_path, (
                    f"V10_14_PRE_HI_PHASE_POSTFIX_NOT_FOUND_BY_DEADLINE:"
                    f"{case.id}:q={q}:W={workload}:D={deadline}:phases={list(phases)}"
                )
            next_R = max(int(R) + 1, int(workload))
            R = deadline if next_R > deadline else next_R

        max_candidate_steps = max(int(max_candidate_steps), len(q_path))
        digest.update(json.dumps({
            "q": int(q),
            "phases": list(int(value) for value in phases),
            "R": int(R),
            "W": int(final_workload),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        if (
            int(R) > int(maximum_response)
            or (int(R) == int(maximum_response) and int(q) > int(worst_q))
        ):
            maximum_response = int(R)
            worst_q = int(q)
            worst_phases = tuple(int(value) for value in phases)
            worst_final_workload = int(final_workload)
            worst_path = list(q_path)

    if maximum_response <= 0 or maximum_response > deadline:
        return None, worst_path, (
            f"V10_14_PRE_HI_PHASE_RCERT_INVALID:{case.id}:R={maximum_response}:D={deadline}"
        )

    try:
        prefix_receipt = _controller_prefix_coverage_receipt(
            model, target, path, int(maximum_response)
        )
    except PCSSCUnresolved as exc:
        return None, worst_path, str(exc)

    return {
        **case.as_dict(),
        "status": "PASS",
        "R": int(maximum_response),
        "W": int(worst_final_workload),
        "W_scope": "WORST_RESPONSE_PHASE_SUBCASE_AT_ITS_OWN_POSTFIX",
        "postfixed": True,
        "candidate_path": worst_path,
        "controller_prefix_receipt": prefix_receipt,
        "case_theorem_basis": "V10_14_PRE_HI_PHASE_CONSISTENT",
        "phase_subcase_count": int(cycle),
        "phase_subcase_covered_count": int(cycle),
        "phase_subcase_digest_sha256": digest.hexdigest(),
        "phase_subcase_max_candidate_steps": int(max_candidate_steps),
        "worst_phase_q": int(worst_q),
        "worst_phase_vector": list(worst_phases),
        "worst_phase_final_workload": int(worst_final_workload),
        "uniform_R_is_common_postfix": False,
        "completion_prefix_used": completion_bounds is not None,
    }, worst_path, None

def _case_consistent_postfix_search(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    protected: set[str],
    protected_response_by_task: dict[str, int],
) -> tuple[int | None, list[dict[str, Any]], list[dict[str, Any]], str | None]:
    try:
        domain, partition_receipts, domain_hash = _deadline_canonical_case_domain(
            model, target, hp_tasks
        )
    except PCSSCUnresolved as exc:
        return None, [], [], str(exc)

    receipts: list[dict[str, Any]] = list(partition_receipts)
    receipts.extend((
        {
            "obligation_id": f"CASE_CONSISTENT_CLASS_DOMAIN::{target.name}",
            "status": "PASS",
            "canonical_deadline": int(target.deadline),
            "canonical_case_count": len(domain),
            "canonical_case_domain_hash": domain_hash,
            "case_ids": [case.id for case in domain],
        },
        {
            "obligation_id": f"CASE_HORIZON_STABILITY::{target.name}",
            "status": "PASS",
            "fixed_class_fields": ["theta", "switch_cell_at_deadline", "target_classification"],
            "horizon_indexed_envelope": "controller prefix and workload are recomputed at R without changing the CaseKey",
            "switch_partition_rebuilt_during_case_recurrence": False,
        },
    ))

    case_certificates: list[dict[str, Any]] = []
    tested: list[dict[str, Any]] = []
    for case in domain:
        cert, path_rows, failure = _case_postfix_search(
            model, target, hp_tasks, path, protected, protected_response_by_task, case
        )
        if cert is None:
            legacy_failure = failure
            if not str(legacy_failure or "").startswith("CASE_POSTFIX_NOT_FOUND_BY_DEADLINE:"):
                tested.append({
                    "terminal": "CASE_CONSISTENT",
                    **case.as_dict(),
                    "status": "UNRESOLVED",
                    "candidate_path": path_rows,
                    "failure": legacy_failure,
                    "v10_13_refinement_attempted": False,
                })
                return None, tested, receipts, (
                    f"CASE_CONSISTENT_PCSSC_UNRESOLVED:{case.id}:{legacy_failure}"
                )
            if case.switch.kind == "PRE_HI":
                refined, refined_rows, refined_failure = (
                    _pre_hi_phase_consistent_postfix_search_v10_14(
                        model, target, hp_tasks, path, protected_response_by_task, case
                    )
                )
                refinement_version = "V10_14"
            else:
                refined, refined_rows, refined_failure = _case_conditioned_postfix_search_v10_13(
                    model, target, hp_tasks, path, protected, protected_response_by_task, case
                )
                refinement_version = "V10_13"
            if refined is None:
                tested.append({
                    "terminal": "CASE_CONSISTENT",
                    **case.as_dict(),
                    "status": "UNRESOLVED",
                    "candidate_path": path_rows,
                    "v10_12_failure": legacy_failure,
                    "refinement_version": refinement_version,
                    "refinement_candidate_path": refined_rows,
                    "failure": refined_failure,
                })
                return None, tested, receipts, (
                    f"CASE_CONSISTENT_PCSSC_UNRESOLVED:{case.id}:"
                    f"V10_12={legacy_failure}:{refinement_version}={refined_failure}"
                )
            cert = refined
            path_rows = refined_rows
            if refinement_version == "V10_14":
                receipts.append({
                    "obligation_id": f"PRE_HI_PHASE_CONSISTENT_REFINEMENT::{target.name},{case.id}",
                    "status": "PASS",
                    "case_id": case.id,
                    "legacy_failure": legacy_failure,
                    "theorem_basis": "V10_14_PRE_HI_FIXED_TARGET_RELEASE_PHASE",
                    "phase_subcase_count": int(cert["phase_subcase_count"]),
                    "phase_subcase_covered_count": int(cert["phase_subcase_covered_count"]),
                    "phase_subcase_digest_sha256": cert["phase_subcase_digest_sha256"],
                    "uniform_R_is_common_postfix": False,
                    "event_graph_used": False,
                })
            else:
                receipts.append({
                    "obligation_id": f"CASE_CONDITIONED_CARRY_FUTURE_REFINEMENT::{target.name},{case.id}",
                    "status": "PASS",
                    "case_id": case.id,
                    "legacy_failure": legacy_failure,
                    "theorem_basis": "V10_13_CASE_CONDITIONED_JOINT_TARGET_RELEASE_PHASE",
                    "same_target_release_index_for_carry_and_future": True,
                    "event_graph_used": False,
                })
        case_certificates.append(cert)
        theorem_basis = cert.get("case_theorem_basis", "V10_12_CASE_CONSISTENT")
        tested_row = {
            "terminal": "CASE_CONSISTENT",
            **case.as_dict(),
            "status": "PASS",
            "R": int(cert["R"]),
            "W": int(cert["W"]),
            "candidate_path": cert["candidate_path"],
            "case_theorem_basis": theorem_basis,
        }
        if theorem_basis == "V10_14_PRE_HI_PHASE_CONSISTENT":
            tested_row.update({
                "phase_subcase_count": int(cert["phase_subcase_count"]),
                "phase_subcase_covered_count": int(cert["phase_subcase_covered_count"]),
                "phase_subcase_digest_sha256": cert["phase_subcase_digest_sha256"],
                "worst_phase_q": int(cert["worst_phase_q"]),
                "worst_phase_vector": list(cert["worst_phase_vector"]),
                "uniform_R_is_common_postfix": False,
            })
        tested.append(tested_row)
        prefix_receipt = dict(cert["controller_prefix_receipt"])
        prefix_receipt["case_id"] = case.id
        receipts.append(prefix_receipt)
        if theorem_basis == "V10_14_PRE_HI_PHASE_CONSISTENT":
            receipts.extend((
                {
                    "obligation_id": f"PRE_HI_PHASE_SUBCASE_DOMAIN_COVERAGE::{target.name},{case.id}",
                    "status": "PASS",
                    "case_id": case.id,
                    "phase_subcase_count": int(cert["phase_subcase_count"]),
                    "covered_phase_subcase_count": int(cert["phase_subcase_covered_count"]),
                    "coverage_digest_sha256": cert["phase_subcase_digest_sha256"],
                    "streamed_without_materializing_full_history_table": True,
                },
                {
                    "obligation_id": f"ALL_PRE_HI_PHASE_SUBCASES_POSTFIX::{target.name},{case.id}",
                    "status": "PASS",
                    "case_id": case.id,
                    "R_case": int(cert["R"]),
                    "worst_phase_q": int(cert["worst_phase_q"]),
                    "worst_phase_final_workload": int(cert["worst_phase_final_workload"]),
                    "uniform_R_case_is_common_postfix": False,
                },
                {
                    "obligation_id": f"PRE_HI_PHASE_CONSISTENT_RESPONSE_CERTIFICATE::{target.name},{case.id},R={cert['R']}",
                    "status": "PASS",
                    "case_id": case.id,
                    "R": int(cert["R"]),
                    "R_le_D": int(cert["R"]) <= int(target.deadline),
                    "derivation": "max of exhaustively checked fixed-joint-phase PRE_HI completion bounds",
                    "not_a_common_phase_postfix": True,
                },
            ))
        else:
            receipts.extend((
                {
                    "obligation_id": f"CASE_WORKLOAD_DOMINANCE::{target.name},{case.id},R={cert['R']}",
                    "status": "PASS",
                    "case_id": case.id,
                    "R": int(cert["R"]),
                    "W": int(cert["W"]),
                    "direct_recheck": True,
                    "case_theorem_basis": theorem_basis,
                },
                {
                    "obligation_id": f"CASE_POSTFIX_RESPONSE_CERTIFICATE::{target.name},{case.id},R={cert['R']}",
                    "status": "PASS",
                    "case_id": case.id,
                    "W": int(cert["W"]),
                    "R": int(cert["R"]),
                    "W_le_R_le_D": int(cert["W"]) <= int(cert["R"]) <= int(target.deadline),
                },
            ))

    domain_ids = [case.id for case in domain]
    covered_ids = [str(cert["case_id"]) for cert in case_certificates]
    missing = sorted(set(domain_ids) - set(covered_ids))
    duplicates = sorted({case_id for case_id in covered_ids if covered_ids.count(case_id) > 1})
    if missing or duplicates or len(covered_ids) != len(domain_ids):
        return None, tested, receipts, (
            f"CASE_DOMAIN_COVERAGE_FAILED:{target.name}:missing={missing}:duplicates={duplicates}"
        )
    Rcert = max(int(cert["R"]) for cert in case_certificates)
    if Rcert > int(target.deadline):
        return None, tested, receipts, f"CASE_CONSISTENT_RCERT_EXCEEDS_DEADLINE:{target.name}:R={Rcert}"
    max_case = max(case_certificates, key=lambda cert: (int(cert["R"]), str(cert["case_id"])))
    v10_13_cases = [
        str(cert["case_id"]) for cert in case_certificates
        if cert.get("case_theorem_basis") == "V10_13_CASE_CONDITIONED_CARRY_FUTURE"
    ]
    v10_14_cases = [
        str(cert["case_id"]) for cert in case_certificates
        if cert.get("case_theorem_basis") == "V10_14_PRE_HI_PHASE_CONSISTENT"
    ]
    receipts.extend((
        {
            "obligation_id": f"CASE_DOMAIN_COVERAGE::{target.name}",
            "status": "PASS",
            "canonical_case_count": len(domain_ids),
            "covered_case_count": len(covered_ids),
            "missing_case_ids": missing,
            "duplicate_case_ids": duplicates,
            "canonical_case_domain_hash": domain_hash,
        },
        {
            "obligation_id": f"ALL_CASES_POSTFIX_COVERED::{target.name}",
            "status": "PASS",
            "canonical_case_count": len(domain_ids),
            "covered_case_count": len(covered_ids),
            "Rcert": int(Rcert),
            "max_case_id": str(max_case["case_id"]),
            "uniform_Rcert_is_common_postfix": False,
            "v10_13_conditioned_case_count": len(v10_13_cases),
            "v10_13_conditioned_case_ids": v10_13_cases,
            "v10_14_pre_hi_phase_case_count": len(v10_14_cases),
            "v10_14_pre_hi_phase_case_ids": v10_14_cases,
        },
        {
            "obligation_id": f"CASE_CONSISTENT_RESPONSE_CERTIFICATE::{target.name},Rcert={Rcert}",
            "status": "PASS",
            "Rcert": int(Rcert),
            "deadline": int(target.deadline),
            "derivation": "max of per-case completion bounds after exhaustive canonical case coverage",
            "not_a_global_postfix": True,
        },
        {
            "obligation_id": (
                f"PCSSC_REFINED_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_14::{target.name}"
                if v10_14_cases else
                f"PCSSC_CASE_CONDITIONED_SAFE_PREFIX_COMPLETION_EXPORT_V10_13::{target.name}"
                if v10_13_cases else
                f"PCSSC_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_12::{target.name}"
            ),
            "status": "PASS",
            "response_bound": int(Rcert),
            "premise": f"ALL_CASES_POSTFIX_COVERED::{target.name}",
            "safe_prefix_completion_contract": True,
            "v10_13_conditioned_cases": v10_13_cases,
            "v10_14_pre_hi_phase_cases": v10_14_cases,
        },
        {
            "obligation_id": f"HI_TARGET_SAFE::{target.name}",
            "status": "PASS",
            "route": (
                "PCSSC_REFINED_CASES_V10_14" if v10_14_cases
                else "PCSSC_CASE_CONDITIONED_CARRY_V10_13" if v10_13_cases
                else "PCSSC_CASE_CONSISTENT"
            ),
            "response_bound": int(Rcert),
        },
    ))
    return int(Rcert), tested, receipts, None


def _initial_horizon(target: TaskBound, hp_tasks: tuple[TaskBound, ...], protected: set[str]) -> int:
    # Candidate generation only.  Soundness never depends on this lower bound:
    # every terminal PASS is checked by W#(R)<=R at the concrete tested R.
    base = _hi_high_cap(target)
    for task in hp_tasks:
        if task.criticality == "LO" and task.name not in protected:
            continue
        base += int(task.actual_demand_upper)
    return max(1, min(int(target.deadline), int(base)))


def prove_target_pcssc(
    model: BoundModel,
    target_name: str,
    path: ControllerMacroPath,
    *,
    priority_assignment_hash: str,
    tie_break_hash: str,
    certified_completion_by_task: Mapping[str, CertifiedCompletionBound] | None = None,
) -> TargetCertificate:
    target = model.task_by_name[target_name]
    if target.criticality != "HI":
        raise ValueError("PCSSC_TARGET_MUST_BE_HI")
    receipts: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = list(path.conservatism_ledger)

    if target.deadline > target.period:
        return TargetCertificate(target.name, "UNRESOLVED", None,
                                 "CONSTRAINED_DEADLINE_D_LE_T_FAILED", (), (), ())
    receipts.extend((
        {"obligation_id": "CONSTRAINED_DEADLINE_D_LE_T", "status": "PASS"},
        {"obligation_id": f"TARGET_SELF_CARRY_IN_ZERO::{target.name}", "status": "PASS"},
        {"obligation_id": f"TARGET_NEXT_RELEASE_EXCLUDED_ON_POSTFIX_HORIZON::{target.name}", "status": "PASS"},
    ))

    priorities = [int(task.priority) for task in model.tasks]
    if len(set(priorities)) != len(priorities) or priorities != sorted(priorities):
        return TargetCertificate(target.name, "UNRESOLVED", None,
                                 "FP_DELAY_ACCOUNTING_UNRESOLVED", tuple(receipts), (), ())
    target_index = next(index for index, task in enumerate(model.tasks) if task.name == target.name)
    hp_tasks = tuple(model.tasks[:target_index])
    receipts.extend((
        {"obligation_id": "TOTAL_PRIORITY_OR_TIEBREAK_BINDING", "status": "PASS",
         "kind": "unique canonical priority_index total order",
         "priority_assignment_hash": priority_assignment_hash,
         "tie_break_hash": tie_break_hash},
        {"obligation_id": f"EFFECTIVE_AHEAD_SET_SOUND::{target.name}", "status": "PASS",
         "hp_eff": [task.name for task in hp_tasks]},
        {"obligation_id": "FIXED_PRIORITY_PREEMPTIVE_WORK_CONSERVING", "status": "PASS",
         "basis": "frozen P0 dispatch/runtime conformance"},
        {"obligation_id": "NO_SELF_SUSPENSION", "status": "PASS"},
        {"obligation_id": "NO_UNMODELED_BLOCKING", "status": "PASS", "blocking_bound": 0},
        {"obligation_id": f"FIXED_PRIORITY_TARGET_DELAY_ACCOUNTING::{target.name}", "status": "PASS",
         "blocking_bound": 0, "ahead_tasks": [task.name for task in hp_tasks],
         "priority_assignment_hash": priority_assignment_hash,
         "tie_break_hash": tie_break_hash},
    ))

    prefix = derive_protected_priority_prefix(model)
    universal_completion = prefix.response_by_task
    certified_completion = dict(certified_completion_by_task or {})
    protected_response_by_task = dict(universal_completion)
    completion_source_by_task: dict[str, list[str]] = {
        name: ["UNIVERSAL_RAW_SERVICE_RESPONSE_PREFIX"]
        for name in universal_completion
    }
    for name, certificate in certified_completion.items():
        task = model.task_by_name.get(name)
        if task is None or certificate.task != name:
            return TargetCertificate(
                target.name, "UNRESOLVED", None,
                f"CERTIFIED_COMPLETION_BINDING_MISMATCH:{name}",
                tuple(receipts), tuple(ledger), (),
            )
        if int(task.priority) >= int(target.priority):
            return TargetCertificate(
                target.name, "UNRESOLVED", None,
                f"PRIORITY_ORDERED_CERTIFICATE_DAG_VIOLATION:{name}->{target.name}",
                tuple(receipts), tuple(ledger), (),
            )
        if (
            certificate.priority != int(task.priority)
            or certificate.deadline != int(task.deadline)
            or certificate.period != int(task.period)
            or certificate.response_bound <= 0
            or certificate.response_bound > int(task.deadline)
            or int(task.deadline) > int(task.period)
        ):
            return TargetCertificate(
                target.name, "UNRESOLVED", None,
                f"CERTIFIED_COMPLETION_CERTIFICATE_INVALID:{name}",
                tuple(receipts), tuple(ledger), (),
            )
        old = protected_response_by_task.get(name)
        if old is None or int(certificate.response_bound) < int(old):
            protected_response_by_task[name] = int(certificate.response_bound)
        completion_source_by_task.setdefault(name, []).append(certificate.source)
    protected = set(protected_response_by_task)
    hp_names = {task.name for task in hp_tasks}
    certified_reused = {
        name: int(certified_completion[name].response_bound)
        for name in sorted(hp_names & set(certified_completion))
    }
    base_reused = {
        name: certified_reused[name]
        for name in certified_reused
        if certified_completion[name].source == BASE_COMPLETION_SOURCE
    }
    pcssc_reused = {
        name: certified_reused[name]
        for name in certified_reused
        if certified_completion[name].source == PCSSC_COMPLETION_SOURCE
    }
    unprotected_hp_lo = [task.name for task in hp_tasks
                         if task.criticality == "LO" and task.name not in protected]
    receipts.append({
        "obligation_id": f"CERTIFIED_COMPLETION_PREFIX_SOUND::{target.name}",
        "status": "PASS",
        "strict_higher_priority_only": True,
        "target_priority": int(target.priority),
        "certificates": {
            name: certified_completion[name].as_dict()
            for name in sorted(certified_reused, key=lambda item: model.task_by_name[item].priority)
        },
    })
    receipts.append({
        "obligation_id": f"REACHABLE_CARRY_IN::{target.name}",
        "status": "PASS",
        "route": "R7_SINGLE_SWITCH_AGGREGATE_BACKLOG_ENVELOPE",
        "universal_protected_priority_prefix": list(prefix.task_names),
        "base_section4_1_completion_envelopes_reused": base_reused,
        "pcssc_completion_envelopes_reused": pcssc_reused,
        "certified_completion_envelopes_reused": certified_reused,
        "effective_completion_envelopes": {
            task.name: int(protected_response_by_task[task.name])
            for task in hp_tasks if task.name in protected_response_by_task
        },
        "completion_envelope_sources": {
            task.name: completion_source_by_task[task.name]
            for task in hp_tasks if task.name in completion_source_by_task
        },
        "lo_tasks_without_single_job_completion_envelope": unprotected_hp_lo,
        "handling_of_unprotected_lo": (
            "included in aggregate work-conserving backlog; no per-task R<=T premise required"
        ),
    })
    if base_reused:
        ledger.append({
            "kind": "BASE_SECTION4_1_COMPLETION_ENVELOPE_REUSE",
            "target": target.name,
            "tasks": base_reused,
            "soundness_basis": (
                "DYNAMIC_TO_BASE_C_AMC_SEM_TRACE_REFINEMENT plus successful fixed-priority "
                "Section 4.1 prefix certificate; max(R_LO,R_HI) bounds completion from release"
            ),
            "soundness_direction": "TIGHTENS_CARRY_IN_WITH_ALREADY_PROVED_COMPLETION_BOUNDS",
        })
    if pcssc_reused:
        ledger.append({
            "kind": "CROSS_TARGET_PCSSC_COMPLETION_PROPAGATION",
            "target": target.name,
            "tasks": pcssc_reused,
            "theorem_basis_by_task": {
                name: certified_completion[name].theorem_basis for name in pcssc_reused
            },
            "soundness_basis": (
                "each reused PCSSC certificate carries an explicit pointwise-V10.11 or "
                "case-consistent-V10.12 safe-prefix completion theorem basis; strict "
                "fixed-priority forward induction permits only fully proved HP Rcert<=D<=T"
            ),
            "soundness_direction": "TIGHTENS_CARRY_IN_AND_PROTECTED_PRE_HI_WITH_PROVED_HP_BOUNDS",
        })
    for name in sorted(certified_reused, key=lambda item: model.task_by_name[item].priority):
        certificate = certified_completion[name]
        receipts.append({
            "obligation_id": f"CROSS_TARGET_COMPLETION_BOUND_REUSE::{name}::{target.name}",
            "status": "PASS",
            "source_task": name,
            "target_task": target.name,
            "response_bound": int(certificate.response_bound),
            "source": certificate.source,
            "theorem_basis": certificate.theorem_basis,
            "source_priority": int(certificate.priority),
            "target_priority": int(target.priority),
            "acyclic": int(certificate.priority) < int(target.priority),
        })
    ledger.append({
        "kind": "R7_SINGLE_SWITCH_AGGREGATE_CARRY_IN",
        "target": target.name,
        "effect": (
            "replace independent per-task carry maxima by one work-conserving aggregate "
            "backlog over exact-periodic releases and at most one LO-to-HI switch"
        ),
        "start_budget_basis": "BOOT_REACHABLE_BUDGET_INVARIANT",
        "soundness_direction": "SOUND_REACHABLE_BACKLOG_REFINEMENT",
    })
    theta = controller_phase_residues(target.period, model.agent_period)
    read_features = required_policy_read_features(model)
    max_depth = int(ceil(int(target.deadline) / int(model.agent_period)))
    try:
        feature_bases = {
            feature: _require_feature_basis(path, feature) for feature in read_features
        }
    except PCSSCUnresolved as exc:
        return TargetCertificate(
            target.name, "UNRESOLVED", None, str(exc), tuple(receipts), tuple(ledger), ()
        )
    for feature in read_features:
        receipts.append({
            "obligation_id": f"FLOW_START_FEATURE_TRANSFER_SOUND::{target.name}::{feature}",
            "status": "PASS",
            "basis": feature_bases[feature],
            "transfer": "arbitrary first-bad SafePrefix start -> full legal policy-read feature envelope",
        })
    for depth in range(max_depth):
        for feature in read_features:
            receipts.append({
                "obligation_id": (
                    f"INTER_EPOCH_FEATURE_TRANSFER_SOUND::{target.name},"
                    f"k={depth}::{feature}"
                ),
                "status": "PASS",
                "basis": feature_bases[feature],
                "transfer": (
                    "budget invariant between candidates; exact FP64 history basis and "
                    "source-bound clipped/fixed-point projection remain inside envelope"
                ),
            })
        receipts.append({
            "obligation_id": f"INTER_EPOCH_FEATURE_TRANSFER_COVERAGE::{target.name},k={depth}",
            "status": "PASS",
            "required_features": list(read_features),
            "basis_ids": [feature_bases[feature] for feature in read_features],
        })
    receipts.extend((
        {"obligation_id": f"CERTIFICATE_START_COVERAGE::{target.name}", "status": "PASS",
         "release_entry_modes": ["HI", "LO"], "target_classes": ["NORMAL", "ABNORMAL"]},
        {"obligation_id": f"CONTROLLER_CANDIDATE_SCHEDULE_COVERAGE::{target.name}", "status": "PASS",
         "trigger": f"strict periodic mod {model.agent_period}", "theta_cells": list(theta)},
        {"obligation_id": f"SINGLE_SWITCH_PROFILE_COVERAGE::{target.name}", "status": "PASS",
         "profiles": ["PRE_HI", "LO_NO_SWITCH", "LO_SWITCH(Sigma)"],
         "lo_endpoint": "u=s uses primary", "hi_endpoint": "u=s may use C_HI"},
        {"obligation_id": f"EXACT_PERIODIC_RELEASE_PROFILE_COVERAGE::{target.name}", "status": "PASS",
         "release_model": "EXACT_PERIODIC_PHASE_ZERO",
         "phase_relation": "hp release phase is conditioned on the same target-relative controller theta",
         "protected_pre_hi_joint_phase": True,
         "other_profiles_phase_relaxation": "only adds compatible phase combinations"},
    ))
    ledger.append({
        "kind": "CROSS_TASK_PERIODIC_PHASE_RELAXATION_OUTSIDE_PROTECTED_PRE_HI",
        "target": target.name,
        "effect": (
            "non-PRE_HI and unprotected-PRE_HI aggregate envelopes may choose "
            "different task phases from the same theta-compatible sets"
        ),
        "soundness_direction": "ONLY_ADDS_CROSS_TASK_PHASE_COMBINATIONS",
    })
    ledger.append({
        "kind": "PROTECTED_PRE_HI_JOINT_PHASE_REFINEMENT",
        "target": target.name,
        "effect": (
            "when every hp task has a single-job completion envelope, one target-release "
            "index jointly determines all hp carry and future release phases"
        ),
        "soundness_direction": "SOUND_TIGHTENING_FROM_EXACT_PERIODIC_AND_COMPLETION_BINDINGS",
    })
    ledger.append({
        "kind": "POLICY_ARRIVAL_CORRELATION_RELAXATION",
        "target": target.name,
        "soundness_direction": "PER_TASK_PERIODIC_PHASE_MAX_NOT_REQUIRED_TO_REPRODUCE_FEATURE_HISTORY",
    })
    ledger.append({
        "kind": "SWITCH_CELL_RELAXATION",
        "target": target.name,
        "soundness_direction": "AMBIGUOUS_SIGMA_CELL_USES_MAX_OF_BOTH_SWITCH_SIDES",
    })
    ledger.append({
        "kind": "FULL_LEGAL_HISTORY_FEATURE_ENVELOPE",
        "target": target.name,
        "soundness_direction": "CONTROLLER_HISTORY_FEATURES_WIDENED_WITHOUT_FUTURE_SAFETY_FEEDBACK",
    })

    point_bound, point_tested, point_receipts, point_failure = _pointwise_postfix_search(
        model, target, hp_tasks, path, protected, protected_response_by_task
    )
    receipts.extend(point_receipts)
    if point_bound is not None:
        return TargetCertificate(
            target.name, "PASS", int(point_bound), None,
            tuple(receipts), tuple(ledger), tuple(point_tested),
            terminal_route=TARGET_PROVED_PCSSC,
            completion_theorem_basis=PCSSC_POINTWISE_COMPLETION_THEOREM,
        )

    receipts.append({
        "obligation_id": f"POINTWISE_PCSSC_FALLBACK_TRIGGER::{target.name}",
        "status": "PASS",
        "pointwise_result": "UNRESOLVED",
        "pointwise_failure": point_failure,
        "completion_prefix_mutated": False,
        "self_completion_exported": False,
        "controller_macro_rebuilt": False,
        "r1_rebuilt": False,
    })
    case_bound, case_tested, case_receipts, case_failure = _case_consistent_postfix_search(
        model, target, hp_tasks, path, protected, protected_response_by_task
    )
    receipts.extend(case_receipts)
    tested = [*point_tested, *case_tested]
    if case_bound is not None:
        used_v10_14 = any(
            str(row.get("obligation_id", "")).startswith(
                f"PCSSC_REFINED_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_14::{target.name}"
            )
            for row in case_receipts
        )
        used_v10_13 = any(
            str(row.get("obligation_id", "")).startswith(
                f"PCSSC_CASE_CONDITIONED_SAFE_PREFIX_COMPLETION_EXPORT_V10_13::{target.name}"
            )
            for row in case_receipts
        )
        if used_v10_13:
            ledger.append({
                "kind": "V10_13_CASE_CONDITIONED_CARRY_FUTURE_COUPLING",
                "target": target.name,
                "effect": (
                    "for V10.12-unresolved LO-entry canonical cases, maximize carry-in "
                    "and future interference over one shared exact-periodic target-release "
                    "index instead of independently maximizing their phase witnesses"
                ),
                "scope": "generic LO-entry LO_NO_SWITCH/LO_SWITCH cases",
                "event_graph_required": False,
                "soundness_direction": "REMOVES_ONLY_IMPOSSIBLE_CROSS_PHASE_COMBINATIONS",
            })
        if used_v10_14:
            ledger.append({
                "kind": "V10_14_PRE_HI_PHASE_CONSISTENT_POSTFIX",
                "target": target.name,
                "effect": (
                    "for a V10.12-unresolved PRE_HI canonical case, keep one exact "
                    "target-release joint HP phase fixed across each response recurrence, "
                    "prove every finite phase subcase separately, then aggregate R bounds"
                ),
                "scope": "generic PRE_HI canonical cases",
                "event_graph_required": False,
                "soundness_direction": "REMOVES_ONLY_CROSS_HORIZON_PHASE_WITNESS_SWITCHING",
            })
        return TargetCertificate(
            target.name, "PASS", int(case_bound), None,
            tuple(receipts), tuple(ledger), tuple(tested),
            terminal_route=(
                TARGET_PROVED_PCSSC_REFINED_CASES_V10_14
                if used_v10_14 else
                TARGET_PROVED_PCSSC_CASE_CONDITIONED_CARRY
                if used_v10_13 else TARGET_PROVED_PCSSC_CASE_CONSISTENT
            ),
            completion_theorem_basis=(
                PCSSC_REFINED_CASE_COMPLETION_THEOREM_V10_14
                if used_v10_14 else
                PCSSC_CONDITIONED_CARRY_COMPLETION_THEOREM
                if used_v10_13 else PCSSC_CASE_COMPLETION_THEOREM
            ),
        )

    return TargetCertificate(
        target.name, "UNRESOLVED", None,
        case_failure or point_failure or "CASE_CONSISTENT_PCSSC_UNRESOLVED",
        tuple(receipts), tuple(ledger), tuple(tested),
    )


__all__ = [
    "CaseKey", "MacroCell", "PCSSCUnresolved", "SwitchCell", "TargetCertificate",
    "prove_target_pcssc",
]
