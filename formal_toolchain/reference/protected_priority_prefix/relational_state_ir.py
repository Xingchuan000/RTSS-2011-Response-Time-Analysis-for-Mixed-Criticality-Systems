from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

from .executable_transition_ir import BoolExpr, IntExpr

JobKey = tuple[str, int]
Phase = str


@dataclass(frozen=True, slots=True)
class SetExpr:
    kind: str
    elements: tuple[str, ...] = ()

    def to_smt(self) -> str:
        if self.kind == "finite_set":
            if not self.elements:
                return "empty_set"
            parts = " ".join(self.elements)
            return f"(set {parts})"
        return "empty_set"


@dataclass(frozen=True, slots=True)
class ProtectedJobIR:
    job_key: tuple[str, int]
    task_name: str
    criticality: str
    release_time: int
    absolute_deadline: int
    priority_index: int
    actual_demand: int
    hi_class: str | None
    executed_service: int
    active: bool
    ready: bool
    running: bool
    completed: bool
    missed: bool


@dataclass(frozen=True, slots=True)
class PendingReleaseIR:
    job_key: tuple[str, int]
    task_name: str
    criticality: str
    release_time: int
    absolute_deadline: int
    priority_index: int
    actual_demand: int
    hi_class: str | None


@dataclass(frozen=True, slots=True)
class ProtectedRelationalStateIR:
    time: IntExpr
    protected_jobs: Mapping[JobKey, ProtectedJobIR]
    protected_pending_releases: Mapping[JobKey, PendingReleaseIR]
    protected_running: Optional[JobKey]
    protected_miss_ledger: SetExpr
    phase: Phase
    ddl_cursor_full: IntExpr
    ddl_cursor_prefix: IntExpr
    arr_cursor_full: IntExpr
    arr_cursor_prefix: IntExpr


EXCLUDED_RELATIONAL_FIELDS = frozenset({
    "global_mode",
    "protected_lo_primary_degraded_label",
    "effective_release_mode",
    "tail_jobs",
    "tail_only_event_identity",
    "switch_trigger_identity",
})


def _job_key_equal(a: tuple[str, int] | None, b: tuple[str, int] | None) -> BoolExpr:
    if a is None and b is None:
        return BoolExpr(kind="atomic", left="true")
    if a is None or b is None:
        return BoolExpr(kind="atomic", left="false")
    a_str = f'"{a[0]}"{a[1]}'
    b_str = f'"{b[0]}"{b[1]}'
    return BoolExpr(kind="cmp", op="=", left=a_str, right=b_str)


def _job_map_eq(full_jobs: Mapping[JobKey, ProtectedJobIR],
                prefix_jobs: Mapping[JobKey, ProtectedJobIR]) -> list[BoolExpr]:
    constraints: list[BoolExpr] = []
    full_keys = set(full_jobs)
    prefix_keys = set(prefix_jobs)
    if full_keys != prefix_keys:
        constraints.append(BoolExpr(kind="atomic", left="false"))
        return constraints
    for key in sorted(full_keys):
        fj = full_jobs[key]
        pj = prefix_jobs[key]
        for attr in ("task_name", "criticality", "release_time", "absolute_deadline",
                     "priority_index", "actual_demand", "hi_class",
                     "executed_service", "active", "ready", "running",
                     "completed", "missed"):
            fv = getattr(fj, attr)
            pv = getattr(pj, attr)
            if isinstance(fv, str):
                constraints.append(
                    BoolExpr(kind="cmp", op="=", left=f'"{fv}"', right=f'"{pv}"')
                )
            elif isinstance(fv, bool):
                constraints.append(
                    BoolExpr(kind="atomic", left="true" if fv == pv else "false")
                )
            else:
                constraints.append(
                    BoolExpr(kind="cmp", op="=", left=str(fv), right=str(pv))
                )
    return constraints


def _pending_map_eq(full_pending: Mapping[JobKey, PendingReleaseIR],
                    prefix_pending: Mapping[JobKey, PendingReleaseIR]) -> list[BoolExpr]:
    constraints: list[BoolExpr] = []
    full_keys = set(full_pending)
    prefix_keys = set(prefix_pending)
    if full_keys != prefix_keys:
        constraints.append(BoolExpr(kind="atomic", left="false"))
        return constraints
    for key in sorted(full_keys):
        fp = full_pending[key]
        pp = prefix_pending[key]
        for attr in ("task_name", "criticality", "release_time", "absolute_deadline",
                     "priority_index", "actual_demand", "hi_class"):
            fv = getattr(fp, attr)
            pv = getattr(pp, attr)
            if isinstance(fv, str):
                constraints.append(
                    BoolExpr(kind="cmp", op="=", left=f'"{fv}"', right=f'"{pv}"')
                )
            elif fv is None and pv is None:
                constraints.append(BoolExpr(kind="atomic", left="true"))
            else:
                constraints.append(
                    BoolExpr(kind="cmp", op="=", left=str(fv), right=str(pv))
                )
    return constraints


def _set_expr_eq(full_set: SetExpr, prefix_set: SetExpr) -> BoolExpr:
    if full_set.kind != prefix_set.kind:
        return BoolExpr(kind="atomic", left="false")
    if sorted(full_set.elements) == sorted(prefix_set.elements):
        return BoolExpr(kind="atomic", left="true")
    return BoolExpr(kind="atomic", left="false")


def rel_pp_close(full: ProtectedRelationalStateIR,
                 prefix: ProtectedRelationalStateIR) -> BoolExpr:
    constraints: list[BoolExpr] = []

    constraints.append(
        BoolExpr(kind="cmp", op="=",
                 left=full.time.to_smt(), right=prefix.time.to_smt())
    )

    constraints.append(_job_key_equal(full.protected_running, prefix.protected_running))

    constraints.extend(_job_map_eq(full.protected_jobs, prefix.protected_jobs))

    constraints.extend(_pending_map_eq(full.protected_pending_releases,
                                        prefix.protected_pending_releases))

    constraints.append(_set_expr_eq(full.protected_miss_ledger,
                                     prefix.protected_miss_ledger))

    return BoolExpr(kind="and", children=tuple(constraints))


def rel_pp_phase(full: ProtectedRelationalStateIR,
                 prefix: ProtectedRelationalStateIR,
                 phase: Phase,
                 cursor_full: int = 0,
                 cursor_prefix: int = 0) -> BoolExpr:
    constraints: list[BoolExpr] = []

    constraints.append(
        BoolExpr(kind="cmp", op="=",
                 left=full.time.to_smt(), right=prefix.time.to_smt())
    )

    constraints.append(
        BoolExpr(kind="cmp", op="=",
                 left=f'"{full.phase}"', right=f'"{prefix.phase}"')
    )

    if phase in ("DDLCursor",):
        constraints.append(
            BoolExpr(kind="cmp", op="=",
                     left=full.ddl_cursor_full.to_smt(),
                     right=str(cursor_full))
        )
        constraints.append(
            BoolExpr(kind="cmp", op="=",
                     left=prefix.ddl_cursor_prefix.to_smt(),
                     right=str(cursor_prefix))
        )

    if phase in ("ARRCursor",):
        constraints.append(
            BoolExpr(kind="cmp", op="=",
                     left=full.arr_cursor_full.to_smt(),
                     right=str(cursor_full))
        )
        constraints.append(
            BoolExpr(kind="cmp", op="=",
                     left=prefix.arr_cursor_prefix.to_smt(),
                     right=str(cursor_prefix))
        )

    constraints.append(_job_key_equal(full.protected_running, prefix.protected_running))

    constraints.extend(_job_map_eq(full.protected_jobs, prefix.protected_jobs))

    constraints.extend(_pending_map_eq(full.protected_pending_releases,
                                        prefix.protected_pending_releases))

    constraints.append(_set_expr_eq(full.protected_miss_ledger,
                                     prefix.protected_miss_ledger))

    return BoolExpr(kind="and", children=tuple(constraints))
