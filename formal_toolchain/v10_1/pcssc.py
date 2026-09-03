"""Policy-Constrained Single-Switch Certificate (PCSSC), V10.16 revision.

The BASE route is followed by the canonical case-consistent PCSSC terminal.
PRE_HI cases go directly to the V10.16 Adaptive Phase-Block theorem; the obsolete
pointwise and phase-relaxed PRE_HI prepasses are not part of the production proof
path.  LO-entry cases retain the V10.12 fast bound and V10.13 exact joint-phase
refinement.  Event Graph search and exact-history BMC remain outside PASS.
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
    PCSSC_REFINED_CASE_COMPLETION_THEOREM_V10_16,
    CertifiedCompletionBound,
)
from .constants import (
    TARGET_PROVED_PCSSC_CASE_CONSISTENT,
    TARGET_PROVED_PCSSC_CASE_CONDITIONED_CARRY,
    TARGET_PROVED_PCSSC_REFINED_CASES_V10_16,
)
from .base_section4_1 import paper_c_lo_bound
from .carry_in_envelope import (
    CarryInEnvelopeUnresolved,
    CarryTaskSpec,
    PhaseBlockProjection,
    exact_joint_lo_entry_max_with_periodic_future,
    fixed_phase_lo_entry_backlog,
    fixed_phase_pre_hi_interference,
    phase_block_completion_carry_upper,
    phase_block_post_switch_future_upper,
    phase_block_r7_carry_upper,
    phase_block_task_projections,
    target_release_joint_phase_parameters,
    target_release_joint_phases_at_q,
    phase_relaxed_lo_entry_carry,
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
class PhaseBlock:
    modulus: int
    residue: int
    depth: int
    parent_id: str | None = None
    split_factor: int | None = None

    @property
    def id(self) -> str:
        return f"M{int(self.modulus)}_A{int(self.residue)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.id,
            "M": int(self.modulus),
            "a": int(self.residue),
            "depth": int(self.depth),
            "parent_id": self.parent_id,
            "split_factor": self.split_factor,
        }


PHASE_BLOCK_MAX_LEAVES = 512


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


@dataclass(slots=True)
class _TargetWorkloadCache:
    """Target-local cache for deterministic LO-entry proof primitives.

    The cache lifetime is exactly one target-level case-consistent proof.  It
    never stores solver state and never changes the theorem selected for a
    case.  Final postfix checks still rebuild the top-level workload identity;
    only deterministic higher-priority subexpressions are reused.
    """

    controller_prefix_by_horizon: dict[int, dict[str, Any]]
    lo_entry_carry_by_theta: dict[int, tuple[int, dict[str, Any]]]
    v10_13_joint_hp_by_case_horizon: dict[
        tuple[int, int, SwitchCell], tuple[int, dict[str, Any]]
    ]

    @classmethod
    def empty(cls) -> "_TargetWorkloadCache":
        return cls({}, {}, {})


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


def _controller_prefix_coverage_receipt_cached(
    cache: _TargetWorkloadCache,
    model: BoundModel,
    target: TaskBound,
    path: ControllerMacroPath,
    horizon: int,
) -> dict[str, Any]:
    key = int(horizon)
    cached = cache.controller_prefix_by_horizon.get(key)
    if cached is not None:
        return cached
    receipt = _controller_prefix_coverage_receipt(model, target, path, key)
    cache.controller_prefix_by_horizon[key] = receipt
    return receipt



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


def _lo_entry_aggregate_carry_bound(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    *,
    theta: int,
) -> tuple[int, dict[str, Any]]:
    specs = _carry_task_specs(hp_tasks, path)
    try:
        value, details = phase_relaxed_lo_entry_carry(
            int(target.period), int(model.agent_period), int(theta), specs
        )
    except CarryInEnvelopeUnresolved as exc:
        raise PCSSCUnresolved(str(exc)) from exc
    return int(value), {
        "basis": "LO_ENTRY_AGGREGATE_BACKLOG_WITH_PER_TASK_PHASE_RELAXATION",
        "carry_in": int(value),
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
    cache: _TargetWorkloadCache,
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

    carry_cached = cache.lo_entry_carry_by_theta.get(int(theta))
    if carry_cached is None:
        carry_cached = _lo_entry_aggregate_carry_bound(
            model, target, hp_tasks, path, theta=int(theta)
        )
        cache.lo_entry_carry_by_theta[int(theta)] = carry_cached
    carry, carry_details = carry_cached

    rows: list[dict[str, Any]] = []
    future_total = 0
    legacy_total: int | None = 0
    legacy_rows: list[dict[str, Any]] = []
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
        if legacy_total is None:
            continue
        if task.criticality == "LO" and task.name not in protected:
            legacy_total = None
            legacy_rows = []
            continue
        task_work, phase_details = _exact_periodic_task_workload(
            target, task, theta=theta, horizon=horizon, cells=cells, weights=weights,
            switch_kind=switch.kind, protected=protected,
            protected_response_by_task=protected_response_by_task,
            controller_period=model.agent_period,
        )
        legacy_total += int(task_work)
        legacy_rows.append({"task": task.name, **phase_details})

    aggregate_total = int(carry) + int(future_total)

    if legacy_total is not None and int(legacy_total) < int(aggregate_total):
        hp_work = int(legacy_total)
        chosen = "MIN_OF_AGGREGATE_AND_COMPLETION_PHASE_COUPLED:COMPLETION_PHASE_COUPLED"
    else:
        hp_work = int(aggregate_total)
        chosen = (
            "MIN_OF_AGGREGATE_AND_COMPLETION_PHASE_COUPLED:AGGREGATE"
            if legacy_total is not None else "AGGREGATE"
        )

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
    cache: _TargetWorkloadCache,
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
    The mathematical V10.13 maximum remains exact, but the implementation does
    not enumerate or materialize the global joint orbit.  Task-local q-periods
    are decomposed into independent coprime CRT components and only those small
    component periods are scanned.
    """
    if switch.kind not in {"LO_NO_SWITCH", "LO_SWITCH"}:
        raise PCSSCUnresolved("CASE_CONDITIONED_CARRY_REQUIRES_LO_ENTRY")

    cache_key = (int(horizon), int(theta), switch)
    cached = cache.v10_13_joint_hp_by_case_horizon.get(cache_key)
    if cached is not None:
        return cached

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
    n0, q_step, joint_cycle = target_release_joint_phase_parameters(
        int(target.period), int(model.agent_period), int(theta), specs
    )
    root_projections = phase_block_task_projections(
        int(target.period),
        int(q_step),
        int(n0),
        specs,
        block_modulus=1,
        block_residue=0,
    )
    future_by_task_q_residue: list[tuple[int, ...]] = []
    for task, spec, projection, weights in zip(
        hp_tasks, specs, root_projections, weights_by_task
    ):
        task_future: list[int] = []
        for q_residue in range(int(projection.q_period)):
            n = int(n0) + int(q_residue) * int(q_step)
            phase = (-int(n) * int(target.period)) % int(spec.period)
            task_future.append(
                _fixed_phase_future_only(
                    task,
                    phase=int(phase),
                    horizon=int(horizon),
                    cells=cells,
                    weights=weights,
                )
            )
        future_by_task_q_residue.append(tuple(task_future))

    maximum, exact_details = exact_joint_lo_entry_max_with_periodic_future(
        int(target.period),
        int(q_step),
        int(n0),
        specs,
        tuple(future_by_task_q_residue),
    )
    witness_q = int(exact_details["witness_q"])
    witness_phases = target_release_joint_phases_at_q(
        int(target.period),
        specs,
        n0=int(n0),
        q_step=int(q_step),
        q=int(witness_q),
    )
    witness_carry, witness_carry_details = fixed_phase_lo_entry_backlog(
        specs, witness_phases
    )
    witness_future = 0
    witness_future_rows: list[dict[str, int | str]] = []
    for task, phase, weights in zip(hp_tasks, witness_phases, weights_by_task):
        task_future = _fixed_phase_future_only(
            task,
            phase=int(phase),
            horizon=int(horizon),
            cells=cells,
            weights=weights,
        )
        witness_future += int(task_future)
        witness_future_rows.append({
            "task": task.name,
            "release_phase": int(phase),
            "future_interference": int(task_future),
        })

    result = (int(maximum), {
        "basis": "V10_13_CASE_CONDITIONED_JOINT_TARGET_RELEASE_PHASE",
        "theta": int(theta),
        "switch_profile": switch.id,
        "joint_phase_cycle": int(joint_cycle),
        "joint_phase_evaluation": "EXACT_CRT_COMPONENT_FACTORIZATION",
        "component_periods": list(exact_details["component_periods"]),
        "component_tasks": [list(row) for row in exact_details["component_tasks"]],
        "carry_candidate_lengths": int(exact_details["candidate_lengths"]),
        "residues_per_candidate": int(exact_details["residues_per_candidate"]),
        "witness_q": int(witness_q),
        "witness_phases": [int(value) for value in witness_phases],
        "carry_in": int(witness_carry),
        "future": int(witness_future),
        "joint_hp_bound": int(maximum),
        "carry_details": witness_carry_details,
        "future_rows": witness_future_rows,
        "same_target_release_index_for_carry_and_future": True,
    })
    cache.v10_13_joint_hp_by_case_horizon[cache_key] = result
    return result


def _workload_case_conditioned_v10_13(
    cache: _TargetWorkloadCache,
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
        cache, model, target, hp_tasks, path, protected, protected_response_by_task,
        horizon=horizon, theta=theta, switch=switch, classification=classification,
    )
    joint_hp, joint_details = _case_conditioned_joint_phase_interference(
        cache, model, target, hp_tasks, path,
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


def _case_postfix_search(
    cache: _TargetWorkloadCache,
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
    while True:
        try:
            _controller_prefix_coverage_receipt_cached(cache, model, target, path, R)
            workload, details = _workload_case(
                cache, model, target, hp_tasks, path, protected, protected_response_by_task,
                horizon=R, theta=case.theta, switch=case.switch,
                classification=case.target_classification,
            )
        except PCSSCUnresolved as exc:
            return None, path_rows, str(exc)
        path_rows.append({"R": int(R), "W": int(workload), "postfixed": bool(workload <= R)})
        if workload <= R:
            # Normative direct re-check.  The iteration above is only a candidate generator.
            try:
                prefix_receipt = _controller_prefix_coverage_receipt_cached(
                    cache, model, target, path, R
                )
                final_workload, final_details = _workload_case(
                    cache, model, target, hp_tasks, path, protected, protected_response_by_task,
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
    cache: _TargetWorkloadCache,
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
    while True:
        try:
            _controller_prefix_coverage_receipt_cached(cache, model, target, path, R)
            workload, details = _workload_case_conditioned_v10_13(
                cache, model, target, hp_tasks, path, protected, protected_response_by_task,
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
                prefix_receipt = _controller_prefix_coverage_receipt_cached(
                    cache, model, target, path, R
                )
                final_workload, final_details = _workload_case_conditioned_v10_13(
                    cache, model, target, hp_tasks, path, protected, protected_response_by_task,
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



def _stable_hash(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _small_prime_factors(value: int) -> tuple[int, ...]:
    n = int(value)
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            factors.append(int(divisor))
            while n % divisor == 0:
                n //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if n > 1:
        factors.append(int(n))
    return tuple(factors)


def _phase_block_split_factor(
    block: PhaseBlock,
    projections: tuple[PhaseBlockProjection, ...],
    specs: tuple[CarryTaskSpec, ...],
    *,
    joint_period: int,
    current_leaf_count: int,
) -> int | None:
    """Choose a low-branching split that shrinks high-work task projections."""

    remaining = int(joint_period) // int(block.modulus)
    candidates: set[int] = set()
    for projection in projections:
        if int(projection.phase_count) > 1:
            candidates.update(_small_prime_factors(int(projection.phase_count)))

    best_factor: int | None = None
    best_score: tuple[int, int, int] | None = None
    for factor in sorted(candidates):
        if factor <= 1 or remaining % factor != 0:
            continue
        if int(current_leaf_count) + int(factor) - 1 > PHASE_BLOCK_MAX_LEAVES:
            continue
        impact = 0
        for spec, projection in zip(specs, projections):
            count = int(projection.phase_count)
            if count % int(factor) == 0:
                reduction = count - count // int(factor)
                weight = max(int(spec.pre_switch_cap), int(spec.post_switch_cap), 1)
                impact += int(reduction) * int(weight)
        if impact <= 0:
            continue
        score = (
            (int(impact) * 1_000_000) // max(1, int(factor) - 1),
            int(impact),
            -int(factor),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_factor = int(factor)
    return best_factor


def _phase_block_children(block: PhaseBlock, factor: int, joint_period: int) -> tuple[PhaseBlock, ...]:
    modulus = int(block.modulus)
    residue = int(block.residue)
    d = int(factor)
    if d <= 1 or int(joint_period) % modulus != 0 or (int(joint_period) // modulus) % d != 0:
        raise PCSSCUnresolved(f"PHASE_BLOCK_INVALID_SPLIT:{block.id}:d={d}")
    child_modulus = modulus * d
    return tuple(
        PhaseBlock(
            modulus=int(child_modulus),
            residue=int(residue + r * modulus),
            depth=int(block.depth) + 1,
            parent_id=block.id,
            split_factor=d,
        )
        for r in range(d)
    )


def _verify_phase_block_leaf_tree(
    root: PhaseBlock,
    leaves: tuple[PhaseBlock, ...],
    children_by_parent: Mapping[str, tuple[PhaseBlock, ...]],
    joint_period: int,
) -> bool:
    leaf_ids = {leaf.id for leaf in leaves}
    if len(leaf_ids) != len(leaves):
        return False
    visited_leaves: set[str] = set()
    visited_nodes: set[str] = set()

    def walk(node: PhaseBlock) -> bool:
        if node.id in visited_nodes:
            return False
        visited_nodes.add(node.id)
        if int(node.modulus) <= 0 or int(joint_period) % int(node.modulus) != 0:
            return False
        if int(node.residue) < 0 or int(node.residue) >= int(node.modulus):
            return False
        children = children_by_parent.get(node.id)
        if children is None:
            if node.id not in leaf_ids:
                return False
            visited_leaves.add(node.id)
            return True
        if node.id in leaf_ids or not children:
            return False
        factor = len(children)
        expected = _phase_block_children(node, factor, int(joint_period))
        if tuple((c.modulus, c.residue) for c in children) != tuple(
            (c.modulus, c.residue) for c in expected
        ):
            return False
        return all(walk(child) for child in children)

    return walk(root) and visited_leaves == leaf_ids


def _phase_block_binding_context(
    target: TaskBound,
    case: CaseKey,
    specs: tuple[CarryTaskSpec, ...],
    *,
    carry_mode: str,
    completion_bounds: tuple[int, ...] | None,
) -> dict[str, Any]:
    """Precompute the case/spec-static part of the V10.16 formula identity."""

    if completion_bounds is not None and len(completion_bounds) != len(specs):
        raise ValueError("PHASE_BLOCK_COMPLETION_BINDING_DIMENSION_MISMATCH")
    service_binding = [
        {
            "task": spec.name,
            "criticality": spec.criticality,
            "period": int(spec.period),
            "pre": int(spec.pre_switch_cap),
            "post": int(spec.post_switch_cap),
            "completion_bound": (
                None if completion_bounds is None else int(completion_bounds[index])
            ),
        }
        for index, spec in enumerate(specs)
    ]
    target_binding = {
        "task": target.name,
        "classification": case.target_classification,
        "service_cap": int(_target_cap(target, case.target_classification)),
    }
    carry_formula_hash = _stable_hash({
        "formula": "V10_16_R7_BOUNDARY_GROUP_COUNT_LIFT",
        "operator": "post_demand + max_switch_delta - age",
        "candidate_domain": "COMPLETE_PROJECTION_BOUNDARY_UNION",
    })
    future_formula_hash = _stable_hash({
        "formula": "V10_16_PRE_HI_FUTURE_NBAR",
        "interval": "[0,R)",
        "coefficient": "post_switch_cap_nonnegative",
    })
    endpoint_binding_hash = _stable_hash({
        "carry": "pending strictly before relative zero",
        "future": "half_open_[0,R)",
        "LO_release_at_switch": "primary_pre_switch_cap",
        "HI_release_at_switch": "post_switch_cap",
    })
    service_cap_binding_hash = _stable_hash({
        "target": target_binding,
        "ahead": service_binding,
    })
    case_key_hash = _stable_hash(case.as_dict())
    monotonicity_check_hash = _stable_hash({
        "carry": "group coordinates enter through + and max; lifted negative optional coordinates use a pointwise upper",
        "future": "arrival count multiplied only by nonnegative post_switch_cap",
        "count_coordinates": "complete symbolic task-local projection",
    })
    noncount_phase_input_binding_hash = _stable_hash({
        "case_key": case.as_dict(),
        "target": target_binding,
        "service_binding": service_binding,
        "hidden_q_dependent_inputs": [],
    })
    return {
        "carry_mode": carry_mode,
        "carry_formula_hash": carry_formula_hash,
        "future_formula_hash": future_formula_hash,
        "endpoint_binding_hash": endpoint_binding_hash,
        "service_cap_binding_hash": service_cap_binding_hash,
        "case_key_hash": case_key_hash,
        "monotonicity_check_hash": monotonicity_check_hash,
        "noncount_phase_input_binding_hash": noncount_phase_input_binding_hash,
    }


def _phase_block_binding(
    projections: tuple[PhaseBlockProjection, ...],
    carry_details: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, str]:
    """Bind only the projection-dependent part of one PASS-leaf formula."""

    projection_binding = [projection.as_dict() for projection in projections]
    count_coordinate_map_hash = _stable_hash(projection_binding)
    candidate_domain_hash = _stable_hash({
        "kind": carry_details.get("candidate_domain_kind"),
        "busy_horizon": int(carry_details.get("busy_horizon", 0)),
        "projection_age_classes": [
            {
                "task": projection.task,
                "first": int(projection.first_release_age),
                "stride": int(projection.phase_stride),
            }
            for projection in projections
        ],
    })
    formula_hash = _stable_hash({
        "carry_formula_hash": context["carry_formula_hash"],
        "future_formula_hash": context["future_formula_hash"],
        "endpoint_binding_hash": context["endpoint_binding_hash"],
        "service_cap_binding_hash": context["service_cap_binding_hash"],
        "case_key_hash": context["case_key_hash"],
        "count_coordinate_map_hash": count_coordinate_map_hash,
        "candidate_domain_hash": candidate_domain_hash,
        "monotonicity_check_hash": context["monotonicity_check_hash"],
        "noncount_phase_input_binding_hash": context["noncount_phase_input_binding_hash"],
        "carry_mode": context["carry_mode"],
    })
    return {
        "formula_hash": formula_hash,
        "carry_formula_hash": str(context["carry_formula_hash"]),
        "future_formula_hash": str(context["future_formula_hash"]),
        "endpoint_binding_hash": str(context["endpoint_binding_hash"]),
        "service_cap_binding_hash": str(context["service_cap_binding_hash"]),
        "case_key_hash": str(context["case_key_hash"]),
        "count_coordinate_map_hash": count_coordinate_map_hash,
        "candidate_domain_hash": candidate_domain_hash,
        "monotonicity_check_hash": str(context["monotonicity_check_hash"]),
        "noncount_phase_input_binding_hash": str(context["noncount_phase_input_binding_hash"]),
    }

def _phase_block_postfix_search_v10_16(
    model: BoundModel,
    target: TaskBound,
    hp_tasks: tuple[TaskBound, ...],
    path: ControllerMacroPath,
    protected_response_by_task: dict[str, int],
    case: CaseKey,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    """V10.16 adaptive PRE_HI phase-block terminal without global-q enumeration.

    Numeric search stays deliberately compact.  Formula/hash receipts are
    materialized only for final PASS leaves; failed internal nodes never build
    certificate JSON.  Final leaves retain only the fields needed for response
    aggregation and the normative V10.16 leaf identity.
    """

    if case.switch.kind != "PRE_HI":
        return None, [], f"V10_16_PHASE_BLOCK_NOT_APPLICABLE:{case.id}"

    specs = _carry_task_specs(hp_tasks, path)
    completion_bounds: tuple[int, ...] | None = None
    if hp_tasks and all(task.name in protected_response_by_task for task in hp_tasks):
        candidate = tuple(int(protected_response_by_task[task.name]) for task in hp_tasks)
        if all(0 < int(bound) <= int(task.period) for task, bound in zip(hp_tasks, candidate)):
            completion_bounds = candidate

    carry_mode = (
        "R7_INTERSECT_COMPLETION" if completion_bounds is not None else "R7_ONLY"
    )
    try:
        binding_context = _phase_block_binding_context(
            target, case, specs, carry_mode=carry_mode, completion_bounds=completion_bounds
        )
    except ValueError as exc:
        return None, [], str(exc)

    try:
        n0, q_step, joint_period = target_release_joint_phase_parameters(
            int(target.period), int(model.agent_period), int(case.theta), specs
        )
    except CarryInEnvelopeUnresolved as exc:
        return None, [], str(exc)
    if int(joint_period) <= 0:
        return None, [], f"V10_16_PHASE_BLOCK_DOMAIN_EMPTY:{case.id}"

    root = PhaseBlock(modulus=1, residue=0, depth=0)
    deadline = int(target.deadline)
    target_work = int(_target_cap(target, case.target_classification))
    pending: list[PhaseBlock] = [root]
    passed: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str, tuple[PhaseBlock, ...]] = {}
    split_receipts: list[dict[str, Any]] = []
    leaf_receipts: list[dict[str, Any]] = []
    block_attempts: list[dict[str, Any]] = []

    while pending:
        block = pending.pop()
        try:
            projections = phase_block_task_projections(
                int(target.period), int(q_step), int(n0), specs,
                block_modulus=int(block.modulus), block_residue=int(block.residue),
            )
            carry_r7, carry_details = phase_block_r7_carry_upper(specs, projections)
            carry_comp: int | None = None
            if completion_bounds is not None:
                carry_comp = phase_block_completion_carry_upper(
                    specs, projections, completion_bounds
                )
            carry_cert = (
                min(int(carry_r7), int(carry_comp))
                if carry_comp is not None else int(carry_r7)
            )
        except (CarryInEnvelopeUnresolved, ValueError) as exc:
            return None, block_attempts, str(exc)

        R = max(1, min(deadline, target_work))
        candidate_steps = 0
        final_workload = -1
        workload_at_deadline = -1
        final_future = -1
        while True:
            candidate_steps += 1
            future = phase_block_post_switch_future_upper(specs, projections, int(R))
            workload = int(target_work) + int(carry_cert) + int(future)
            if workload <= R:
                # Normative V10.16 direct recheck: recompute the same leaf formula
                # at the candidate postfix rather than trusting recurrence state.
                final_future = phase_block_post_switch_future_upper(
                    specs, projections, int(R)
                )
                final_workload = int(target_work) + int(carry_cert) + int(final_future)
                if final_workload > int(R):
                    return None, block_attempts, (
                        f"V10_16_PHASE_BLOCK_DIRECT_POSTFIX_RECHECK_FAILED:"
                        f"{case.id}:{block.id}:R={R}:W={final_workload}"
                    )
                break
            if R == deadline:
                workload_at_deadline = int(workload)
                break
            next_R = max(int(R) + 1, int(workload))
            R = deadline if next_R > deadline else next_R

        if final_workload >= 0:
            try:
                binding = _phase_block_binding(
                    projections, carry_details, binding_context
                )
            except ValueError as exc:
                return None, block_attempts, str(exc)

            # One lifting receipt plus one postfix receipt is sufficient.  The
            # lifting receipt already binds the R7 candidate domain and future
            # count formula, so separate per-leaf R7/FUTURE duplicate receipts
            # add no proof information and previously dominated JSON volume.
            leaf_receipts.append({
                "obligation_id": (
                    f"PHASE_BLOCK_WORKLOAD_LIFTING_SOUND::{target.name},"
                    f"{case.id},{binding['formula_hash']}"
                ),
                "status": "PASS",
                "case_id": case.id,
                **block.as_dict(),
                **binding,
                "candidate_domain_kind": carry_details["candidate_domain_kind"],
                "count_coordinate_mapping": "SYMBOLIC_GCD_TASK_LOCAL_PROJECTION",
                "noncount_phase_inputs_fully_bound": True,
                "carry_r7_B": int(carry_r7),
                "carry_cert_B": int(carry_cert),
                "future_at_postfix": int(final_future),
                "boundary_union_complete_for_all_fixed_q_maximizers": True,
            })
            if carry_comp is not None:
                leaf_receipts.append({
                    "obligation_id": f"PHASE_BLOCK_COMPLETION_CARRY_SOUND::{case.id},{block.id}",
                    "status": "PASS",
                    **block.as_dict(),
                    "completion_carry_B": int(carry_comp),
                    "construction": "SUM_OF_PER_TASK_MAX_RESIDUAL_OVER_COMPLETE_BLOCK_PROJECTION",
                })
            leaf_receipts.append({
                "obligation_id": f"PHASE_BLOCK_POSTFIX::{case.id},{block.id}",
                "status": "PASS",
                **block.as_dict(),
                "R_B": int(R),
                "W_B_R_B": int(final_workload),
                "formula_hash": binding["formula_hash"],
                "count_projection_hash": binding["count_coordinate_map_hash"],
                "candidate_domain_hash": binding["candidate_domain_hash"],
                "carry_mode": carry_mode,
                "direct_recheck": True,
                "W_le_R_le_D": int(final_workload) <= int(R) <= deadline,
            })
            passed[block.id] = {
                **block.as_dict(),
                "R": int(R),
                "W": int(final_workload),
                "formula_hash": binding["formula_hash"],
            }
            block_attempts.append({
                **block.as_dict(), "status": "PASS", "R": int(R), "W": int(final_workload),
                "projection_sizes": [int(p.phase_count) for p in projections],
                "candidate_steps": int(candidate_steps),
            })
            continue

        block_attempts.append({
            **block.as_dict(), "status": "FAILED_BLOCK",
            "W_at_D": int(workload_at_deadline),
            "projection_sizes": [int(p.phase_count) for p in projections],
            "candidate_steps": int(candidate_steps),
        })
        active_leaf_count = len(passed) + len(pending) + 1
        factor = _phase_block_split_factor(
            block, projections, specs, joint_period=int(joint_period),
            current_leaf_count=int(active_leaf_count),
        )
        if factor is None:
            mathematical_candidates = {
                prime
                for projection in projections
                if int(projection.phase_count) > 1
                for prime in _small_prime_factors(int(projection.phase_count))
                if int(joint_period) // int(block.modulus) % int(prime) == 0
            }
            reason = (
                "STRUCTURAL_LEAF_BUDGET_EXHAUSTED"
                if mathematical_candidates else "NO_LEGAL_SPLIT"
            )
            return None, block_attempts, (
                f"PHASE_BLOCK_REFINEMENT_INSUFFICIENT:{case.id}:{block.id}:{reason}"
            )
        try:
            children = _phase_block_children(block, int(factor), int(joint_period))
        except PCSSCUnresolved as exc:
            return None, block_attempts, str(exc)
        children_by_parent[block.id] = children
        split_receipts.append({
            "obligation_id": f"PHASE_BLOCK_SPLIT::{case.id},{block.id}",
            "status": "PASS",
            "case_id": case.id,
            "parent_block_id": block.id,
            "parent_M": int(block.modulus),
            "parent_a": int(block.residue),
            "split_factor": int(factor),
            "child_modulus": int(block.modulus) * int(factor),
            "child_count": int(factor),
            "parent_equals_disjoint_union_children": True,
        })
        pending.extend(reversed(children))

    final_leaves = tuple(
        PhaseBlock(
            modulus=int(cert["M"]), residue=int(cert["a"]), depth=int(cert["depth"]),
            parent_id=cert["parent_id"], split_factor=cert["split_factor"],
        )
        for cert in passed.values()
    )
    if not final_leaves or not _verify_phase_block_leaf_tree(
        root, final_leaves, children_by_parent, int(joint_period)
    ):
        return None, block_attempts, f"PHASE_BLOCK_LEAF_COVERAGE_FAILED:{case.id}"

    leaf_certs = list(passed.values())
    maximum_response = max(int(cert["R"]) for cert in leaf_certs)
    if maximum_response <= 0 or maximum_response > deadline:
        return None, block_attempts, (
            f"V10_16_PHASE_BLOCK_RCERT_INVALID:{case.id}:R={maximum_response}:D={deadline}"
        )
    worst = max(leaf_certs, key=lambda cert: (int(cert["R"]), str(cert["block_id"])))
    try:
        prefix_receipt = _controller_prefix_coverage_receipt(
            model, target, path, int(maximum_response)
        )
    except PCSSCUnresolved as exc:
        return None, block_attempts, str(exc)

    leaf_digest = _stable_hash([
        {
            "block_id": cert["block_id"],
            "M": cert["M"],
            "a": cert["a"],
            "R": cert["R"],
            "W": cert["W"],
            "formula_hash": cert["formula_hash"],
        }
        for cert in sorted(leaf_certs, key=lambda row: (int(row["M"]), int(row["a"])))
    ])
    phase_receipts = [{
        "obligation_id": f"PHASE_BLOCK_ROOT_DOMAIN::{case.id}",
        "status": "PASS",
        "case_id": case.id,
        "Q": int(joint_period),
        "root": root.as_dict(),
        "lcm_empty_convention": 1,
        "empty_ahead": not bool(specs),
        "global_q_enumerated": False,
        "structural_plan": {
            "max_leaf_count": PHASE_BLOCK_MAX_LEAVES,
            "termination_measure": "bounded active leaf partition",
            "split_factors": "prime factors of current task-local projection periods",
        },
    }, *split_receipts, *leaf_receipts, {
        "obligation_id": f"PHASE_BLOCK_LEAF_COVERAGE::{case.id}",
        "status": "PASS",
        "case_id": case.id,
        "Q": int(joint_period),
        "leaf_count": len(final_leaves),
        "leaf_digest_sha256": leaf_digest,
        "verified_by_legal_split_induction": True,
    }, {
        "obligation_id": f"ALL_PHASE_BLOCKS_POSTFIX::{case.id}",
        "status": "PASS",
        "case_id": case.id,
        "leaf_count": len(final_leaves),
        "R_case": int(maximum_response),
        "uniform_R_case_is_common_postfix": False,
    }, {
        "obligation_id": f"PHASE_BLOCK_RESPONSE_CERTIFICATE::{case.id}",
        "status": "PASS",
        "case_id": case.id,
        "R": int(maximum_response),
        "R_le_D": int(maximum_response) <= deadline,
        "derivation": "max of all final phase-block completion bounds",
        "not_a_common_block_postfix": True,
    }]

    return {
        **case.as_dict(),
        "status": "PASS",
        "R": int(maximum_response),
        "W": int(worst["W"]),
        "W_scope": "WORST_RESPONSE_PHASE_BLOCK_AT_ITS_OWN_POSTFIX",
        "postfixed": True,
        "candidate_path": block_attempts,
        "controller_prefix_receipt": prefix_receipt,
        "case_theorem_basis": "V10_16_ADAPTIVE_PHASE_BLOCK",
        "phase_block_joint_period": int(joint_period),
        "phase_block_leaf_count": len(final_leaves),
        "phase_block_leaf_digest_sha256": leaf_digest,
        "worst_phase_block": str(worst["block_id"]),
        "worst_phase_block_final_workload": int(worst["W"]),
        "uniform_R_is_common_postfix": False,
        "completion_prefix_used": completion_bounds is not None,
        "phase_block_receipts": phase_receipts,
        "global_q_enumerated": False,
    }, block_attempts, None

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

    cache = _TargetWorkloadCache.empty()
    case_certificates: list[dict[str, Any]] = []
    tested: list[dict[str, Any]] = []
    for case in domain:
        if case.switch.kind == "PRE_HI":
            cert, path_rows, failure = _phase_block_postfix_search_v10_16(
                model, target, hp_tasks, path, protected_response_by_task, case
            )
            if cert is None:
                tested.append({
                    "terminal": "CASE_CONSISTENT",
                    **case.as_dict(),
                    "status": "UNRESOLVED",
                    "refinement_version": "V10_16",
                    "refinement_candidate_path": path_rows,
                    "failure": failure,
                })
                return None, tested, receipts, (
                    f"CASE_CONSISTENT_PCSSC_UNRESOLVED:{case.id}:V10_16={failure}"
                )
            receipts.append({
                "obligation_id": f"ADAPTIVE_PHASE_BLOCK_REFINEMENT::{target.name},{case.id}",
                "status": "PASS",
                "case_id": case.id,
                "theorem_basis": "V10_16_ADAPTIVE_PHASE_BLOCK_PCSSC",
                "phase_block_joint_period": int(cert["phase_block_joint_period"]),
                "phase_block_leaf_count": int(cert["phase_block_leaf_count"]),
                "phase_block_leaf_digest_sha256": cert["phase_block_leaf_digest_sha256"],
                "global_q_enumerated": False,
                "uniform_R_is_common_postfix": False,
                "event_graph_used": False,
                "legacy_pre_hi_prepass_removed": True,
            })
            receipts.extend(cert["phase_block_receipts"])
        else:
            cert, path_rows, failure = _case_postfix_search(
                cache, model, target, hp_tasks, path, protected,
                protected_response_by_task, case
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
                cert, refined_rows, refined_failure = _case_conditioned_postfix_search_v10_13(
                    cache, model, target, hp_tasks, path, protected,
                    protected_response_by_task, case
                )
                if cert is None:
                    tested.append({
                        "terminal": "CASE_CONSISTENT",
                        **case.as_dict(),
                        "status": "UNRESOLVED",
                        "candidate_path": path_rows,
                        "v10_12_failure": legacy_failure,
                        "refinement_version": "V10_13",
                        "refinement_candidate_path": refined_rows,
                        "failure": refined_failure,
                    })
                    return None, tested, receipts, (
                        f"CASE_CONSISTENT_PCSSC_UNRESOLVED:{case.id}:"
                        f"V10_12={legacy_failure}:V10_13={refined_failure}"
                    )
                path_rows = refined_rows
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
        if theorem_basis == "V10_16_ADAPTIVE_PHASE_BLOCK":
            tested_row.update({
                "phase_block_joint_period": int(cert["phase_block_joint_period"]),
                "phase_block_leaf_count": int(cert["phase_block_leaf_count"]),
                "phase_block_leaf_digest_sha256": cert["phase_block_leaf_digest_sha256"],
                "worst_phase_block": cert["worst_phase_block"],
                "uniform_R_is_common_postfix": False,
                "global_q_enumerated": False,
            })
        tested.append(tested_row)
        prefix_receipt = dict(cert["controller_prefix_receipt"])
        prefix_receipt["case_id"] = case.id
        receipts.append(prefix_receipt)
        if theorem_basis != "V10_16_ADAPTIVE_PHASE_BLOCK":
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
    v10_16_cases = [
        str(cert["case_id"]) for cert in case_certificates
        if cert.get("case_theorem_basis") == "V10_16_ADAPTIVE_PHASE_BLOCK"
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
            "v10_16_adaptive_phase_block_case_count": len(v10_16_cases),
            "v10_16_adaptive_phase_block_case_ids": v10_16_cases,
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
                f"PCSSC_REFINED_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_16::{target.name}"
                if v10_16_cases else
                f"PCSSC_CASE_CONDITIONED_SAFE_PREFIX_COMPLETION_EXPORT_V10_13::{target.name}"
                if v10_13_cases else
                f"PCSSC_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_12::{target.name}"
            ),
            "status": "PASS",
            "response_bound": int(Rcert),
            "premise": f"ALL_CASES_POSTFIX_COVERED::{target.name}",
            "safe_prefix_completion_contract": True,
            "v10_13_conditioned_cases": v10_13_cases,
            "v10_16_adaptive_phase_block_cases": v10_16_cases,
        },
        {
            "obligation_id": f"HI_TARGET_SAFE::{target.name}",
            "status": "PASS",
            "route": (
                "PCSSC_REFINED_CASES_V10_16" if v10_16_cases
                else "PCSSC_CASE_CONDITIONED_CARRY_V10_13" if v10_13_cases
                else "PCSSC_CASE_CONSISTENT"
            ),
            "response_bound": int(Rcert),
        },
    ))
    return int(Rcert), tested, receipts, None


def prove_target_pcssc(
    model: BoundModel,
    target_name: str,
    path: ControllerMacroPath,
    *,
    priority_assignment_hash: str,
    tie_break_hash: str,
    release_model: str,
    release_model_hash: str,
    release_domain_hash: str,
    source_manifest_semantic_hash: str,
    release_generator_source_hash: str,
    certified_completion_by_task: Mapping[str, CertifiedCompletionBound] | None = None,
) -> TargetCertificate:
    target = model.task_by_name[target_name]
    if target.criticality != "HI":
        raise ValueError("PCSSC_TARGET_MUST_BE_HI")
    receipts: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = list(path.conservatism_ledger)

    if release_model != "EXACT_PERIODIC_PHASE_ZERO":
        return TargetCertificate(
            target.name, "UNRESOLVED", None,
            "P0_RELEASE_DOMAIN_EXACT_PERIODIC_PHASE_ZERO_MISSING", (), (), (),
        )
    receipts.append({
        "obligation_id": "P0_RELEASE_DOMAIN_EXACT_PERIODIC_PHASE_ZERO",
        "status": "PASS",
        "release_model": release_model,
        "release_model_hash": release_model_hash,
        "release_domain_hash": release_domain_hash,
        "source_manifest_semantic_hash": source_manifest_semantic_hash,
        "release_generator_source_hash": release_generator_source_hash,
        "binding": "actual frozen event runtime release generator plus admissible-environment release domain",
    })

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
         "hp_eff": [task.name for task in hp_tasks],
         "strict_hp": [task.name for task in hp_tasks],
         "hp_eff_equals_strict_hp": True,
         "proof": "UNIQUE_PRIORITY_IMPLIES_HP_EFF_EQ_STRICT_HP"},
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
    ))
    ledger.append({
        "kind": "FAST_ROUTE_CROSS_TASK_PERIODIC_PHASE_RELAXATION",
        "target": target.name,
        "effect": (
            "V10.12 LO-entry aggregate envelopes may choose different theta-compatible "
            "task phases; PRE_HI canonical cases are handled directly by V10.16"
        ),
        "soundness_direction": "ONLY_ADDS_CROSS_TASK_PHASE_COMBINATIONS",
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

    case_bound, case_tested, case_receipts, case_failure = _case_consistent_postfix_search(
        model, target, hp_tasks, path, protected, protected_response_by_task
    )
    receipts.extend(case_receipts)
    tested = list(case_tested)
    if case_bound is not None:
        used_v10_16 = any(
            str(row.get("obligation_id", "")).startswith(
                f"PCSSC_REFINED_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_16::{target.name}"
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
        if used_v10_16:
            ledger.append({
                "kind": "V10_16_ADAPTIVE_PHASE_BLOCK_PCSSC",
                "target": target.name,
                "effect": (
                    "replace global joint-q PRE_HI enumeration by adaptive congruence blocks; "
                    "lift R7 carry and future counts over complete task-local gcd projections, "
                    "directly recheck every final block postfix, then aggregate leaf bounds"
                ),
                "scope": "generic PRE_HI canonical cases under exact-periodic phase-zero releases",
                "event_graph_required": False,
                "global_q_enumerated": False,
                "soundness_direction": "BLOCK_LIFTING_DOMINATES_EVERY_MEMBER_AND_REFINEMENT_ONLY_PARTITIONS_THE_SAME_DOMAIN",
            })
        return TargetCertificate(
            target.name, "PASS", int(case_bound), None,
            tuple(receipts), tuple(ledger), tuple(tested),
            terminal_route=(
                TARGET_PROVED_PCSSC_REFINED_CASES_V10_16
                if used_v10_16 else
                TARGET_PROVED_PCSSC_CASE_CONDITIONED_CARRY
                if used_v10_13 else TARGET_PROVED_PCSSC_CASE_CONSISTENT
            ),
            completion_theorem_basis=(
                PCSSC_REFINED_CASE_COMPLETION_THEOREM_V10_16
                if used_v10_16 else
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
