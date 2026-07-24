"""PP0 Transition Intermediate Representation.

Defines hand-maintained but **人工审计 (manually audited)** guard/update
adapter equations for the nine primitive cases.

These equations are manually written and audited against the actual
executable semantics code.  The ``binding_kind`` field is set to
``"HAND_WRITTEN_SCHEMA_ONLY"`` to mark them as the manually audited but non-code-bound
transition schema for PP0 relational SMT queries.

Per the V10 Codex plan, the adapter equations are source-bound by
function name and AST hash from ``pp_transition_binding.py``, not by
compiler extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_file, sha256_object

ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = {
    "executable_semantics": "formal_toolchain/reference/executable_semantics.py",
    "p0_transition_contract": "formal_toolchain/reference/p0_transition_contract.py",
    "p0_projection": "formal_toolchain/reference/p0_projection.py",
    "logical_events": "formal_toolchain/bridge/logical_events.py",
}


@dataclass(frozen=True, slots=True)
class Equation:
    lhs: str
    rhs: str
    kind: str  # "state", "frame", "time", "guard"


@dataclass(frozen=True, slots=True)
class PP0TransitionIR:
    case_id: str
    phase_before: str
    phase_after: str
    guard_formula: str
    state_equations: tuple[Equation, ...]
    frame_equations: tuple[Equation, ...]
    time_equation: Equation
    source_function: str
    source_binding: str
    binding_kind: str = "HAND_WRITTEN_SCHEMA_ONLY"

    @property
    def ir_hash(self) -> str:
        payload = {
            "case_id": self.case_id,
            "phase_before": self.phase_before,
            "phase_after": self.phase_after,
            "guard_formula": self.guard_formula,
            "state_equations": [(e.lhs, e.rhs, e.kind) for e in self.state_equations],
            "frame_equations": [(e.lhs, e.rhs, e.kind) for e in self.frame_equations],
            "time_equation": (self.time_equation.lhs, self.time_equation.rhs, self.time_equation.kind),
            "source_function": self.source_function,
            "binding_kind": self.binding_kind,
        }
        return sha256_object(payload)


def _eq(kind: str, lhs: str, rhs: str) -> Equation:
    return Equation(lhs=lhs, rhs=rhs, kind=kind)


def _source_hash(short_name: str) -> str:
    return sha256_file(ROOT / SOURCE_FILES[short_name])


def build_pp0_transition_ir() -> tuple[PP0TransitionIR, ...]:
    """Build all nine schema-level PP0 transition records.

    A source-file hash is provenance only.  It does not make the handwritten
    equations equivalent to the Python transition function.
    """
    exe_hash = _source_hash("executable_semantics")
    p0_hash = _source_hash("p0_transition_contract")
    events_hash = _source_hash("logical_events")
    proj_hash = _source_hash("p0_projection")

    return (
        PP0TransitionIR(
            case_id="REM_COMPLETION",
            phase_before="AfterSvc",
            phase_after="AfterREM",
            guard_formula="(>= service fixed_demand)",
            state_equations=(
                _eq("state", "active_post", "(- active_pre 1)"),
                _eq("state", "ready_post", "(- ready_pre 1)"),
                _eq("state", "completed_post", "1"),
            ),
            frame_equations=(
                _eq("frame", "release_time_post", "release_time_pre"),
                _eq("frame", "absolute_deadline_post", "absolute_deadline_pre"),
                _eq("frame", "criticality_post", "criticality_pre"),
                _eq("frame", "fixed_demand_post", "fixed_demand_pre"),
                _eq("frame", "priority_index_post", "priority_index_pre"),
                _eq("frame", "hi_class_post", "hi_class_pre"),
            ),
            time_equation=_eq("time", "time_post", "time_pre"),
            source_function="apply_removal",
            source_binding=exe_hash,
        ),
        PP0TransitionIR(
            case_id="RECOVERY",
            phase_before="AfterREM",
            phase_after="AfterREC",
            guard_formula="(and (= mode HI) (= active_job_count 0) (= running_present 0) (= pending_release_count 0))",
            state_equations=(
                _eq("state", "mode_post", "LO"),
            ),
            frame_equations=(
                _eq("frame", "release_time_post", "release_time_pre"),
                _eq("frame", "absolute_deadline_post", "absolute_deadline_pre"),
                _eq("frame", "criticality_post", "criticality_pre"),
                _eq("frame", "fixed_demand_post", "fixed_demand_pre"),
                _eq("frame", "service_post", "service_pre"),
                _eq("frame", "priority_index_post", "priority_index_pre"),
                _eq("frame", "active_post", "active_pre"),
                _eq("frame", "ready_post", "ready_pre"),
            ),
            time_equation=_eq("time", "time_post", "time_pre"),
            source_function="apply_recovery",
            source_binding=exe_hash,
        ),
        PP0TransitionIR(
            case_id="DDL_OBSERVE",
            phase_before="AfterREC",
            phase_after="DDLCursor",
            guard_formula="(= event_kind DEADLINE)",
            state_equations=(
                _eq("state", "miss_post", "(ite (not completed_pre) (+ miss_pre 1) miss_pre)"),
            ),
            frame_equations=(
                _eq("frame", "release_time_post", "release_time_pre"),
                _eq("frame", "criticality_post", "criticality_pre"),
                _eq("frame", "fixed_demand_post", "fixed_demand_pre"),
                _eq("frame", "service_post", "service_pre"),
                _eq("frame", "active_post", "active_pre"),
                _eq("frame", "ready_post", "ready_pre"),
                _eq("frame", "priority_index_post", "priority_index_pre"),
            ),
            time_equation=_eq("time", "time_post", "time_pre"),
            source_function="apply_deadline_observation",
            source_binding=exe_hash,
        ),
        PP0TransitionIR(
            case_id="ARRIVAL_BATCH_OPEN",
            phase_before="DDLCursor",
            phase_after="ARRCursor",
            guard_formula="(= event_kind ARR_BATCH)",
            state_equations=(
                _eq("state", "pending_releases_post", "(+ pending_releases_pre batch_size)"),
                _eq("state", "active_post", "active_pre"),
                _eq("state", "ready_post", "ready_pre"),
            ),
            frame_equations=(
                _eq("frame", "job_key_post", "job_key_pre"),
                _eq("frame", "release_time_post", "release_time_pre"),
                _eq("frame", "criticality_post", "criticality_pre"),
            ),
            time_equation=_eq("time", "time_post", "time_pre"),
            source_function="apply_arrival_batch",
            source_binding=exe_hash,
        ),
        PP0TransitionIR(
            case_id="MODE_SWITCH",
            phase_before="ARRCursor",
            phase_after="PreDisp",
            guard_formula="(and (= event_kind SW) (= mode LO) (= pending_abnormal_trigger 1))",
            state_equations=(
                _eq("state", "mode_post", "HI"),
            ),
            frame_equations=(
                _eq("frame", "release_time_post", "release_time_pre"),
                _eq("frame", "absolute_deadline_post", "absolute_deadline_pre"),
                _eq("frame", "criticality_post", "criticality_pre"),
                _eq("frame", "fixed_demand_post", "fixed_demand_pre"),
                _eq("frame", "service_post", "service_pre"),
                _eq("frame", "active_post", "active_pre"),
                _eq("frame", "ready_post", "ready_pre"),
                _eq("frame", "priority_index_post", "priority_index_pre"),
            ),
            time_equation=_eq("time", "time_post", "time_pre"),
            source_function="apply_mode_switch",
            source_binding=exe_hash,
        ),
        PP0TransitionIR(
            case_id="RELEASE",
            phase_before="PreDisp",
            phase_after="PreDisp",
            guard_formula="(= event_kind RELEASE)",
            state_equations=(
                _eq("state", "active_post", "(+ active_pre 1)"),
                _eq("state", "ready_post", "(+ ready_pre 1)"),
            ),
            frame_equations=(
                _eq("frame", "job_key_post", "job_key_pre"),
                _eq("frame", "release_time_post", "release_time_pre"),
                _eq("frame", "criticality_post", "criticality_pre"),
                _eq("frame", "priority_index_post", "priority_index_pre"),
            ),
            time_equation=_eq("time", "time_post", "time_pre"),
            source_function="apply_release",
            source_binding=exe_hash,
        ),
        PP0TransitionIR(
            case_id="FINAL_DISPATCH",
            phase_before="PreDisp",
            phase_after="Close",
            guard_formula="(> active_job_count_pre 0)",
            state_equations=(
                _eq("state", "running_post", "1"),
                _eq("state", "running_job_key_post",
                    "(select_min_priority_index active_job_set priority_index release_time job_key)"),
            ),
            frame_equations=(
                _eq("frame", "release_time_post", "release_time_pre"),
                _eq("frame", "absolute_deadline_post", "absolute_deadline_pre"),
                _eq("frame", "criticality_post", "criticality_pre"),
                _eq("frame", "fixed_demand_post", "fixed_demand_pre"),
                _eq("frame", "service_post", "service_pre"),
                _eq("frame", "priority_index_post", "priority_index_pre"),
            ),
            time_equation=_eq("time", "time_post", "time_pre"),
            source_function="_normalize_dispatch",
            source_binding=exe_hash,
        ),
        PP0TransitionIR(
            case_id="SERVICE_UNIT",
            phase_before="Close",
            phase_after="AfterSvc",
            guard_formula="(> running_pre 0)",
            state_equations=(
                _eq("state", "service_post", "(+ service_pre 1)"),
                _eq("state", "time_post", "(+ time_pre 1)"),
            ),
            frame_equations=(
                _eq("frame", "release_time_post", "release_time_pre"),
                _eq("frame", "absolute_deadline_post", "absolute_deadline_pre"),
                _eq("frame", "criticality_post", "criticality_pre"),
                _eq("frame", "priority_index_post", "priority_index_pre"),
                _eq("frame", "active_post", "active_pre"),
                _eq("frame", "ready_post", "ready_pre"),
            ),
            time_equation=_eq("time", "time_post", "(+ time_pre 1)"),
            source_function="apply_service_tick",
            source_binding=exe_hash,
        ),
        PP0TransitionIR(
            case_id="TAIL_ONLY_SERVICE",
            phase_before="Close",
            phase_after="AfterSvc",
            guard_formula="(and (= protected_ready_pre 0) (> tail_ready_pre 0))",
            state_equations=(
                _eq("state", "tail_service_post", "(+ tail_service_pre 1)"),
                _eq("state", "time_post", "(+ time_pre 1)"),
            ),
            frame_equations=(
                _eq("frame", "release_time_post", "release_time_pre"),
                _eq("frame", "absolute_deadline_post", "absolute_deadline_pre"),
                _eq("frame", "criticality_post", "criticality_pre"),
                _eq("frame", "fixed_demand_post", "fixed_demand_pre"),
                _eq("frame", "service_post", "service_pre"),
                _eq("frame", "active_post", "active_pre"),
                _eq("frame", "ready_post", "ready_pre"),
                _eq("frame", "priority_index_post", "priority_index_pre"),
            ),
            time_equation=_eq("time", "time_post", "(+ time_pre 1)"),
            source_function="apply_service_tick",
            source_binding=exe_hash,
        ),
    )


def ir_for_case(case_id: str) -> PP0TransitionIR | None:
    for ir in build_pp0_transition_ir():
        if ir.case_id == case_id:
            return ir
    return None


def transition_ir_map() -> dict[str, PP0TransitionIR]:
    return {ir.case_id: ir for ir in build_pp0_transition_ir()}
