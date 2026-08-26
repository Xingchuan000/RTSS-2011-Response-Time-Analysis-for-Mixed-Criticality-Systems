"""P0 运行时语义证据的共享构造入口。

这个模块只做一次正常程序运行内的事实归集，不会自己发明新的语义结论。
各个 checker 需要的字段都从这里统一取，避免 semantic builder 各自重跑一遍
运行时并引入互相不一致的 witness。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True)
class P0RuntimeEvidence:
    """一次正常 P0 运行中收集到的共享运行时证据。"""

    status: str
    route: str | None
    code: str | None

    initial_state: Mapping[str, Any]
    boot: Mapping[str, Any]
    micro_scenarios: Mapping[str, Any]

    event_binding: Mapping[str, Any]
    removal_binding: Mapping[str, Any]
    recovery_binding: Mapping[str, Any]
    controller_binding: Mapping[str, Any]

    phase_edges: Mapping[str, list[str]]
    controller_fields: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p0_runtime_evidence_v1",
            "status": self.status,
            "route": self.route,
            "code": self.code,
            "initial_state": dict(self.initial_state),
            "boot": dict(self.boot),
            "micro_scenarios": dict(self.micro_scenarios),
            "event_binding": dict(self.event_binding),
            "removal_binding": dict(self.removal_binding),
            "recovery_binding": dict(self.recovery_binding),
            "controller_binding": dict(self.controller_binding),
            "phase_edges": {str(key): list(value) for key, value in self.phase_edges.items()},
            "controller_fields": dict(self.controller_fields),
        }


def _initial_state_from_adapter(adapter: Any) -> Mapping[str, Any]:
    if not callable(getattr(adapter, "export_initial_state_contract", None)):
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "code": "INITIAL_STATE_CONTRACT_MISSING",
        }
    return adapter.export_initial_state_contract()


def _boot_from_adapter(adapter: Any) -> Mapping[str, Any]:
    if not callable(getattr(adapter, "export_boot_transition_contract", None)):
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "code": "BOOT_TRANSITION_CONTRACT_MISSING",
        }
    return adapter.export_boot_transition_contract()


def derive_controller_fields(
    *,
    target: Any,
    controller_binding: Mapping[str, Any],
    event_binding: Mapping[str, Any],
    recovery_binding: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    boot: Mapping[str, Any],
) -> dict[str, Any]:
    """把不同 binding 里已经证明过的事实整理成 controller contract 需要的字段。"""

    initial_budgets = dict(initial_state.get("runtime_budgets", {}))
    boot_snapshot = dict(boot.get("initial_runtime_budget_snapshot", {}))
    budget_immutable = (
        controller_binding.get("status") == "PASS"
        and initial_budgets == boot_snapshot
    )
    noop = controller_binding.get("explicit_noop_runtime_binding", {})
    noop_stutter = (
        isinstance(noop, Mapping)
        and noop.get("status") == "PASS"
        and noop.get("timing_projection") == "STUTTER"
    )
    selected = controller_binding.get("selected_action_runtime_binding", {})
    selected_stutter_if_preclosed = (
        isinstance(selected, Mapping)
        and selected.get("status") == "PASS"
        and selected.get("timing_projection") == "STUTTER_IF_PRECLOSED"
        and selected.get("requires_preclosed_boundary") is True
    )
    selected_active_unchanged = (
        selected_stutter_if_preclosed and selected.get("active_jobs_unchanged") is True
    )
    selected_ready_unchanged = (
        selected_stutter_if_preclosed and selected.get("ready_jobs_unchanged") is True
    )
    selected_running_unchanged_if_preclosed = (
        selected_stutter_if_preclosed
        and selected.get("running_job_unchanged_if_preclosed") is True
    )
    selected_service_unchanged = (
        selected_stutter_if_preclosed and selected.get("service_unchanged") is True
    )
    selected_mode_unchanged = (
        selected_stutter_if_preclosed and selected.get("mode_unchanged") is True
    )
    selected_frontier_unchanged_if_preclosed = (
        selected_stutter_if_preclosed
        and selected.get("effective_event_frontier_unchanged_if_preclosed") is True
    )
    return {
        "witnesses": [controller_binding, event_binding, recovery_binding, initial_state, boot],
        "binding": controller_binding,
        "binding_hash": sha256_object(controller_binding),
        "sequence_allocation_deterministic": (
            event_binding.get("status") == "PASS"
            and event_binding.get("fifo_sequence") == "EventQueue._counter"
        ),
        "finite_token_height": all(
            int(task.period) > 0 and int(task.deadline) > 0
            for task in target.ordered_tasks
        ),
        "ready_nonempty_advances_tick": controller_binding.get("status") == "PASS",
        "ready_empty_jumps_next_event": event_binding.get("status") == "PASS",
        "zero_time_stutter_forbidden": (
            controller_binding.get("status") == "PASS"
            and event_binding.get("status") == "PASS"
        ),
        "active_release_budget_immutable": budget_immutable,
        # V7 A13-A16 are not inferred from generic controller status.  They
        # are consumed from the production AmcBudgetEnv.step noop branch
        # binding, whose proof boundary ends before the later plant run_until.
        "explicit_noop_budget_identity": noop_stutter and noop.get("budget_identity") is True,
        "explicit_noop_macro_stutter": noop_stutter
        and noop.get("running_job_unchanged") is True
        and noop.get("mode_unchanged") is True
        and noop.get("controller_time_unchanged") is True,
        "explicit_noop_effective_frontier_stutter": noop_stutter
        and noop.get("effective_event_frontier_unchanged") is True,
        "explicit_noop_released_jobs_immutable": noop_stutter
        and noop.get("released_job_fields_unchanged") is True,
        "explicit_noop_fallback_equivalent": noop_stutter
        and noop.get("explicit_and_fallback_same_timing_semantics") is True,
        "explicit_noop_plant_progress_separated": noop_stutter
        and noop.get("plant_progress_separated") is True,
        "selected_active_unchanged": selected_active_unchanged,
        "selected_ready_unchanged": selected_ready_unchanged,
        "selected_requires_preclosed_boundary": selected_stutter_if_preclosed,
        "selected_running_unchanged_if_preclosed": selected_running_unchanged_if_preclosed,
        "selected_service_unchanged": selected_service_unchanged,
        "selected_mode_unchanged": selected_mode_unchanged,
        "selected_released_job_fields_unchanged": (
            selected_stutter_if_preclosed and selected.get("released_job_fields_unchanged") is True
        ),
        "selected_released_job_snapshot_unchanged": (
            selected_stutter_if_preclosed and selected.get("released_job_snapshot_unchanged") is True
        ),
        "selected_released_job_service_unchanged": (
            selected_stutter_if_preclosed and selected.get("released_job_service_unchanged") is True
        ),
        "selected_released_job_demand_unchanged": (
            selected_stutter_if_preclosed and selected.get("released_job_demand_unchanged") is True
        ),
        "selected_released_job_classification_unchanged": (
            selected_stutter_if_preclosed and selected.get("released_job_classification_unchanged") is True
        ),
        "selected_completion_miss_unchanged": (
            selected_stutter_if_preclosed and selected.get("completion_miss_unchanged") is True
        ),
        "selected_effective_event_frontier_unchanged_if_preclosed": (
            selected_frontier_unchanged_if_preclosed
        ),
        "selected_plant_progression_separated": (
            selected_stutter_if_preclosed and selected.get("plant_progression_separated") is True
        ),
        "selected_timing_stutter_if_preclosed": selected_stutter_if_preclosed,
        "changes_active": not selected_active_unchanged,
        "changes_ready": not selected_ready_unchanged,
        "changes_running_if_preclosed": not selected_running_unchanged_if_preclosed,
        "changes_current_service": not selected_service_unchanged,
        "changes_mode": not selected_mode_unchanged,
        "changes_service": not selected_service_unchanged,
    }


def build_p0_runtime_evidence(*, target: Any, source_root: Path) -> P0RuntimeEvidence:
    """统一生成一次正常运行里的 P0 runtime 证据。"""

    from formal_toolchain.binding.controller_binding import bind_controller_runtime
    from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
    from formal_toolchain.binding.recovery_binding import bind_recovery_runtime
    from formal_toolchain.binding.removal_binding import bind_removal_runtime
    from formal_toolchain.conformance.boot_controller import derive_phase_edges
    from formal_toolchain.conformance.micro_scenarios import run_p0_micro_scenarios

    adapter = getattr(target, "runtime_adapter", None)
    if adapter is None:
        return P0RuntimeEvidence(
            status="UNRESOLVED",
            route="UNRESOLVED",
            code="FORMAL_RUNTIME_ADAPTER_MISSING",
            initial_state={},
            boot={},
            micro_scenarios={},
            event_binding={},
            removal_binding={},
            recovery_binding={},
            controller_binding={},
            phase_edges={},
            controller_fields={},
        )

    initial = _initial_state_from_adapter(adapter)
    boot = _boot_from_adapter(adapter)
    micro = run_p0_micro_scenarios(target_available=True)
    event = bind_event_runtime(Path(source_root))
    removal = bind_removal_runtime(Path(source_root))
    recovery = bind_recovery_runtime(Path(source_root))
    controller = bind_controller_runtime(Path(source_root))

    if event.get("status") == "PASS":
        try:
            phase_edges = derive_phase_edges(event)
        except ValueError:
            phase_edges = {}
    else:
        phase_edges = {}

    controller_fields = derive_controller_fields(
        target=target,
        controller_binding=controller,
        event_binding=event,
        recovery_binding=recovery,
        initial_state=initial,
        boot=boot,
    )

    components = {
        "initial_state": initial,
        "boot": boot,
        "micro_scenarios": micro,
        "event_binding": event,
        "removal_binding": removal,
        "recovery_binding": recovery,
        "controller_binding": controller,
    }
    statuses = {
        str(value.get("status", "UNRESOLVED"))
        for value in components.values()
        if isinstance(value, Mapping)
    }
    if "FAIL" in statuses:
        status = "FAIL"
        route = "MODEL_CONFORMANCE_FAILED"
        code = "P0_RUNTIME_EVIDENCE_FAILED"
    elif "UNRESOLVED" in statuses:
        status = "UNRESOLVED"
        route = "UNRESOLVED"
        code = "P0_RUNTIME_EVIDENCE_INCOMPLETE"
    else:
        status = "PASS"
        route = None
        code = None

    return P0RuntimeEvidence(
        status=status,
        route=route,
        code=code,
        initial_state=initial,
        boot=boot,
        micro_scenarios=micro,
        event_binding=event,
        removal_binding=removal,
        recovery_binding=recovery,
        controller_binding=controller,
        phase_edges=phase_edges,
        controller_fields=controller_fields,
    )
