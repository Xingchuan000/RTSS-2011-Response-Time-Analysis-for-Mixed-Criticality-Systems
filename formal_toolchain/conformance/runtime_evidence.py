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

    names = [str(task.name) for task in target.ordered_tasks]
    initial_budgets = dict(initial_state.get("runtime_budgets", {}))
    boot_snapshot = dict(boot.get("initial_runtime_budget_snapshot", {}))
    budget_immutable = (
        controller_binding.get("status") == "PASS"
        and initial_budgets == boot_snapshot
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
        "changes_active": False,
        "changes_ready": False,
        "changes_running": False,
        "changes_current_service": False,
        "changes_mode": False,
        "changes_service": False,
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
