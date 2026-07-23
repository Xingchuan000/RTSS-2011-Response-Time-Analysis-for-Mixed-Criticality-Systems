from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.bridge.model_bounds import P0ModelBounds
from formal_toolchain.bridge.state_relation import p0_smt_relation_fields
from formal_toolchain.core.hashing import sha256_object


REFERENCE_P0_CASE_IDS = (
    "BOOT_TO_PRECLOSED_0",
    "ARRIVAL_BATCH_NO_SWITCH",
    "ARRIVAL_BATCH_SWITCH_S0",
    "PRIMARY_LO_RELEASE",
    "DEGRADED_LO_RELEASE",
    "HI_RELEASE",
    "RESCHEDULE_KEEP_SAME",
    "RESCHEDULE_TO_IDLE",
    "PREEMPTION_DISPATCH",
    "ONE_SERVICE_TICK",
    "NORMAL_COMPLETION",
    "PRIMARY_LO_CANCELLATION",
    "DEGRADED_COMPLETION",
    "HI_COMPLETION",
    "DEADLINE_OBSERVATION_NO_MISS",
    "DEADLINE_OBSERVATION_FIRST_HI_MISS",
    "IDLE_RECOVERY",
    "CONTROLLER_NO_ACTION",
    "CONTROLLER_SELECTED_ACTION",
    "JUMP_TO_NEXT_EVENT",
)


@dataclass(frozen=True, slots=True)
class ReferenceP0Environment:
    expected_demand: int = 0

    release_budget: int = 0
    release_job_key: int = 0
    release_priority: int = 0
    release_time: int = 0
    release_deadline: int = 0
    release_category: int = 0
    release_mode: int = 0
    is_degraded: int = 0
    task_criticality: int = 0

    selected_job_key: int = 0
    event_job_key: int = 0
    running_job_key: int = 0

    elapsed: int = 0
    next_event_time: int = 0

    release_slot: int = 0
    affected_job_slot: int = 0
    update_target_slot: int = 0
    update_arity: int = 0


REFERENCE_P0_ENVIRONMENT_FIELDS = (
    "expected_demand",
    "release_budget",
    "release_job_key",
    "release_priority",
    "release_time",
    "release_deadline",
    "release_category",
    "release_mode",
    "is_degraded",
    "task_criticality",
    "selected_job_key",
    "event_job_key",
    "running_job_key",
    "elapsed",
    "next_event_time",
    "release_slot",
    "affected_job_slot",
    "update_target_slot",
    "update_arity",
)


def reference_transition_identity_fields(
    bounds: P0ModelBounds,
) -> tuple[str, ...]:
    scalar_fields = ("time", "miss", "mode")
    job_fields = tuple(
        f"job_{slot}_{field}"
        for slot in range(bounds.job_slots)
        for field in (
            "present", "key", "active", "ready", "running", "priority",
            "release", "deadline", "category", "criticality", "mode",
            "released_mode", "is_degraded", "budget", "demand", "service",
            "completion_token", "overrun_token", "hi_complete", "hi_miss",
        )
    )
    task_fields = tuple(
        f"task_{slot}_{field}"
        for slot in range(bounds.task_slots)
        for field in ("present", "key", "criticality", "future_budget")
    )
    fields = scalar_fields + job_fields + task_fields
    relation_fields = set(p0_smt_relation_fields(bounds))
    missing = sorted(set(fields) - relation_fields)
    if missing:
        raise ValueError(
            "REFERENCE_IDENTITY_FIELD_OUTSIDE_RELATION:" + ",".join(missing)
        )
    return fields


def _smt_int(value: int) -> str:
    value = int(value)
    return f"(- {-value})" if value < 0 else str(value)


def build_reference_p0_identity_query(
    *,
    case_id: str,
    before: Mapping[str, int],
    actual_after: Mapping[str, int],
    environment: ReferenceP0Environment,
    bounds: P0ModelBounds,
) -> str:
    fields = p0_smt_relation_fields(bounds)
    identity_fields = reference_transition_identity_fields(bounds)
    declarations = []
    for field in fields:
        declarations.extend((f"(declare-const r_{field} Int)",
                             f"(declare-const r_{field}_post Int)"))
    for field in REFERENCE_P0_ENVIRONMENT_FIELDS:
        declarations.append(f"(declare-const {field} Int)")
    assertions = [f"(assert {render_reference_p0_delta(case_id, bounds)})"]
    for field in fields:
        assertions.append(f"(assert (= r_{field} {_smt_int(before[field])}))")
    for field in REFERENCE_P0_ENVIRONMENT_FIELDS:
        assertions.append(
            f"(assert (= {field} {_smt_int(getattr(environment, field))}))"
        )
    for field in identity_fields:
        assertions.append(
            f"(assert (= r_{field}_post {_smt_int(actual_after[field])}))"
        )
    return "\n".join(("(set-logic QF_LIA)", *declarations, *assertions))


def _slot_ite(variable: str, slot: int, when_true: str, when_false: str) -> str:
    return f"(ite (= {variable} {slot}) {when_true} {when_false})"


def _all_post(prefix: str, bounds: P0ModelBounds, overrides: Mapping[str, str] | None = None) -> str:
    overrides = overrides or {}
    fields = p0_smt_relation_fields(bounds)
    equations: list[str] = []
    for field in fields:
        value = overrides.get(field, f"{prefix}_{field}")
        equations.append(f"(= {prefix}_{field}_post {value})")
    return "(and " + " ".join(equations) + ")"


def _release_job_overrides(prefix: str, bounds: P0ModelBounds) -> dict[str, str]:
    result: dict[str, str] = {}
    for slot in range(bounds.job_slots):
        base = f"{prefix}_job_{slot}_"
        values = {
            "present": "1", "active": "1", "ready": "1", "running": "0",
            "key": "release_job_key", "priority": "release_priority",
            "release": "release_time", "deadline": "release_deadline",
            "category": "release_category", "criticality": "task_criticality",
            "mode": "release_mode", "released_mode": "release_mode",
            "is_degraded": "is_degraded", "budget": "release_budget",
            "demand": "expected_demand", "service": "0", "token": "0",
            "completion_token": "0", "overrun_token": "0", "hi_complete": "0", "hi_miss": "0",
        }
        for field, value in values.items():
            result[f"job_{slot}_{field}"] = _slot_ite(
                "release_slot", slot, value, f"{base}{field}")
    return result


def _remove_job_overrides(prefix: str, bounds: P0ModelBounds) -> dict[str, str]:
    result: dict[str, str] = {}
    for slot in range(bounds.job_slots):
        base = f"{prefix}_job_{slot}_"
        selected = f"(= {base}key event_job_key)"
        for field, value in {"present": "0", "active": "0", "ready": "0",
                             "running": "0", "completion_token": "0", "overrun_token": "0",
                             "hi_complete": "0", "hi_miss": "0"}.items():
            result[f"job_{slot}_{field}"] = f"(ite {selected} {value} {base}{field})"
    return result


def _dispatch_job_overrides(prefix: str, bounds: P0ModelBounds) -> dict[str, str]:
    result: dict[str, str] = {}
    for slot in range(bounds.job_slots):
        base = f"{prefix}_job_{slot}_"
        selected = (
            f"(and (= {base}present 1) "
            f"(= {base}key selected_job_key))"
        )
        result[f"job_{slot}_running"] = f"(ite {selected} 1 0)"
    return result


def _task_budget_overrides(prefix: str, bounds: P0ModelBounds) -> dict[str, str]:
    return {f"task_{slot}_future_budget": _slot_ite(
        "update_target_slot", slot, "release_budget", f"{prefix}_task_{slot}_future_budget")
        for slot in range(bounds.task_slots)}


def render_reference_p0_delta(
    case_id: str,
    bounds: P0ModelBounds,
) -> str:
    c: dict[str, str] = {"time": "r_time", "service": "r_service", "remaining": "r_remaining"}
    if case_id == "ONE_SERVICE_TICK":
        c.update(time="(+ r_time elapsed)", service="(+ r_service elapsed)",
                 remaining="(- r_remaining elapsed)",
                 affected_job_service="(+ r_affected_job_service elapsed)")
        c.update({f"job_{slot}_service":
                  f"(ite (and (= r_job_{slot}_present 1) (= r_job_{slot}_running 1) "
                  f"(= r_job_{slot}_key running_job_key)) "
                  f"(+ r_job_{slot}_service elapsed) r_job_{slot}_service)"
                  for slot in range(bounds.job_slots)})
    elif case_id == "JUMP_TO_NEXT_EVENT":
        c["time"] = "next_event_time"
    elif case_id == "DEADLINE_OBSERVATION_FIRST_HI_MISS":
        c.update(miss="1", affected_job_hi_miss="1")
        c.update({
            f"job_{slot}_hi_miss": _slot_ite(
                "affected_job_slot",
                slot,
                "1",
                f"r_job_{slot}_hi_miss",
            )
            for slot in range(bounds.job_slots)
        })
    elif case_id in {"PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE"}:
        c.update(active="(+ r_active 1)", ready="(+ r_ready 1)", service="0",
                  remaining="expected_demand", budget="release_budget", priority="release_priority",
                 release="release_time", deadline="release_deadline", category="release_category",
                 job_key="release_job_key", affected_job_key="release_job_key",
                 affected_job_active="1", affected_job_ready="1",
                 affected_job_running="0", affected_job_priority="release_priority",
                 affected_job_release="release_time", affected_job_deadline="release_deadline",
                 affected_job_category="release_category", affected_job_budget="release_budget",
                 affected_job_demand="expected_demand", affected_job_service="0")
        c.update(_release_job_overrides("r", bounds))
        c.update({"event_job_key": "release_job_key"})
    elif case_id in {"NORMAL_COMPLETION", "DEGRADED_COMPLETION", "PRIMARY_LO_CANCELLATION"}:
        c.update(active="(- r_active 1)", ready="(- r_ready 1)", running="0",
                 affected_job_active="0", affected_job_ready="0", affected_job_running="0")
        c.update(_remove_job_overrides("r", bounds))
    elif case_id == "HI_COMPLETION":
        c.update(active="(- r_active 1)", ready="(- r_ready 1)", running="0",
                 affected_job_active="0", affected_job_ready="0", affected_job_running="0",
                 affected_job_hi_complete="r_affected_job_hi_complete")
        c.update(_remove_job_overrides("r", bounds))
    elif case_id == "ARRIVAL_BATCH_SWITCH_S0":
        c["mode"] = "1"
    elif case_id == "IDLE_RECOVERY":
        c["mode"] = "0"
    elif case_id == "RESCHEDULE_KEEP_SAME":
        pass
    elif case_id == "RESCHEDULE_TO_IDLE":
        c.update(running="0", running_job_key="0", affected_job_running="0",
                 queue_token_epoch="(+ r_queue_token_epoch 1)")
        c.update({f"job_{slot}_running": "0" for slot in range(bounds.job_slots)})
    elif case_id == "PREEMPTION_DISPATCH":
        c.update(running="1", running_job_key="selected_job_key", affected_job_key="selected_job_key",
                 affected_job_running="1",
                 queue_token_epoch="(+ r_queue_token_epoch 1)",
                 queue_event_count="(+ r_queue_event_count 2)")
        c.update(_dispatch_job_overrides("r", bounds))
    elif case_id == "CONTROLLER_SELECTED_ACTION":
        c.update(future_budget="release_budget", affected_task_budget="release_budget")
        c.update(_task_budget_overrides("r", bounds))
    elif case_id in {"BOOT_TO_PRECLOSED_0", "ARRIVAL_BATCH_NO_SWITCH",
                     "CONTROLLER_NO_ACTION", "DEADLINE_OBSERVATION_NO_MISS"}:
        pass
    else:
        raise ValueError(f"REFERENCE_P0_UNKNOWN_CASE_ID:{case_id}")
    body = _all_post("r", bounds, c)
    if case_id in {"PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE"}:
        return (
            "(and "
            "(> expected_demand 0) "
            "(<= expected_demand release_budget) "
            + body[5:]
        )
    return body


def legacy_reference_p0_numeric_delta(
    *,
    case_id: str,
    before: Mapping[str, int],
    environment: ReferenceP0Environment,
    bounds: P0ModelBounds,
) -> dict[str, int]:
    after: dict[str, int] = {}
    fields = p0_smt_relation_fields(bounds)
    for field in fields:
        after[field] = before[field]
    r_time = before["time"]
    r_service = before["service"]
    r_remaining = before["remaining"]
    env = environment
    if case_id == "ONE_SERVICE_TICK":
        after["time"] = r_time + env.elapsed
        after["service"] = r_service + env.elapsed
        after["remaining"] = r_remaining - env.elapsed
        after["affected_job_service"] = before["affected_job_service"] + env.elapsed
        for slot in range(bounds.job_slots):
            key = f"job_{slot}_key"
            present = f"job_{slot}_present"
            running_flag = f"job_{slot}_running"
            svc = f"job_{slot}_service"
            if before[present] == 1 and before[running_flag] == 1 and before[key] == before["running_job_key"]:
                after[svc] = before[svc] + env.elapsed
                if after[svc] >= before[f"job_{slot}_budget"]:
                    after[f"job_{slot}_hi_complete"] = int(
                        before[f"job_{slot}_criticality"] == 1
                    )
    elif case_id == "JUMP_TO_NEXT_EVENT":
        after["time"] = env.next_event_time
    elif case_id == "DEADLINE_OBSERVATION_FIRST_HI_MISS":
        after["miss"] = 1
        after["affected_job_hi_miss"] = 1
        if 0 <= env.affected_job_slot < bounds.job_slots:
            after[f"job_{env.affected_job_slot}_hi_miss"] = 1
    elif case_id in {"PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE"}:
        after["active"] = before["active"] + 1
        after["ready"] = before["ready"] + 1
        after["service"] = 0
        after["remaining"] = env.expected_demand
        after["budget"] = env.release_budget
        after["priority"] = env.release_priority
        after["release"] = env.release_time
        after["deadline"] = env.release_deadline
        after["category"] = env.release_category
        after["job_key"] = env.release_job_key
        after["affected_job_key"] = env.release_job_key
        after["affected_job_active"] = 1
        after["affected_job_ready"] = 0
        after["affected_job_running"] = int(
            env.selected_job_key == env.release_job_key
        )
        after["affected_job_priority"] = env.release_priority
        after["affected_job_release"] = env.release_time
        after["affected_job_deadline"] = env.release_deadline
        after["affected_job_category"] = env.release_category
        after["affected_job_budget"] = env.release_budget
        after["affected_job_demand"] = env.expected_demand
        after["affected_job_service"] = 0
        after["running"] = 1
        after["running_job_key"] = env.selected_job_key
        after["selected_job_key"] = env.selected_job_key
        after["event_job_key"] = env.event_job_key
        for slot in range(bounds.job_slots):
            if slot == env.release_slot:
                after[f"job_{slot}_present"] = 1
                after[f"job_{slot}_active"] = 1
                after[f"job_{slot}_ready"] = 1
                after[f"job_{slot}_running"] = int(
                    env.selected_job_key == env.release_job_key
                )
                after[f"job_{slot}_key"] = env.release_job_key
                after[f"job_{slot}_priority"] = env.release_priority
                after[f"job_{slot}_release"] = env.release_time
                after[f"job_{slot}_deadline"] = env.release_deadline
                after[f"job_{slot}_category"] = env.release_category
                after[f"job_{slot}_criticality"] = env.task_criticality
                after[f"job_{slot}_mode"] = env.release_mode
                after[f"job_{slot}_released_mode"] = env.release_mode
                after[f"job_{slot}_is_degraded"] = env.is_degraded
                after[f"job_{slot}_budget"] = env.release_budget
                after[f"job_{slot}_demand"] = env.expected_demand
                after[f"job_{slot}_service"] = 0
                after[f"job_{slot}_completion_token"] = 0
                after[f"job_{slot}_overrun_token"] = 0
                after[f"job_{slot}_hi_complete"] = 0
                after[f"job_{slot}_hi_miss"] = 0
        after["ready_empty"] = int(after["ready"] == 0)
        for slot in range(bounds.job_slots):
            key = f"job_{slot}_key"
            if after[key] == env.selected_job_key:
                after[f"job_{slot}_running"] = 1
            elif after[f"job_{slot}_present"]:
                after[f"job_{slot}_running"] = 0
    elif case_id in {"NORMAL_COMPLETION", "DEGRADED_COMPLETION", "PRIMARY_LO_CANCELLATION"}:
        after["active"] = before["active"] - 1
        after["ready"] = before["ready"] - 1
        after["running"] = 0
        after["affected_job_active"] = 0
        after["affected_job_ready"] = 1
        after["affected_job_running"] = 0
        for slot in range(bounds.job_slots):
            key = f"job_{slot}_key"
            if before[key] == before.get("event_job_key", before.get("running_job_key", 0)):
                after[f"job_{slot}_present"] = 0
                after[f"job_{slot}_active"] = 0
                after[f"job_{slot}_ready"] = 0
                after[f"job_{slot}_running"] = 0
                after[f"job_{slot}_completion_token"] = 0
                after[f"job_{slot}_overrun_token"] = 0
                after[f"job_{slot}_hi_complete"] = 0
                after[f"job_{slot}_hi_miss"] = 0
    elif case_id == "HI_COMPLETION":
        after["active"] = before["active"] - 1
        after["ready"] = before["ready"] - 1
        after["running"] = 0
        after["affected_job_active"] = 0
        after["affected_job_ready"] = 0
        after["affected_job_running"] = 0
        after["affected_job_hi_complete"] = before["affected_job_hi_complete"]
        for slot in range(bounds.job_slots):
            key = f"job_{slot}_key"
            if before[key] == before.get("event_job_key", before.get("running_job_key", 0)):
                after[f"job_{slot}_present"] = 0
                after[f"job_{slot}_active"] = 0
                after[f"job_{slot}_ready"] = 0
                after[f"job_{slot}_running"] = 0
                after[f"job_{slot}_completion_token"] = 0
                after[f"job_{slot}_overrun_token"] = 0
                after[f"job_{slot}_hi_complete"] = 0
                after[f"job_{slot}_hi_miss"] = 0
    elif case_id == "ARRIVAL_BATCH_SWITCH_S0":
        after["mode"] = 1
    elif case_id == "IDLE_RECOVERY":
        after["mode"] = 0
    elif case_id == "RESCHEDULE_KEEP_SAME":
        pass
    elif case_id == "RESCHEDULE_TO_IDLE":
        after["running"] = 0
        after["running_job_key"] = 0
        after["affected_job_running"] = 0
        for slot in range(bounds.job_slots):
            after[f"job_{slot}_running"] = 0
    elif case_id == "PREEMPTION_DISPATCH":
        # ``running`` is the Boolean occupancy projection; the selected job
        # identity is carried separately by ``running_job_key``.
        after["running"] = 1
        after["running_job_key"] = env.selected_job_key
        after["selected_job_key"] = env.selected_job_key
        after["affected_job_key"] = env.selected_job_key
        after["affected_job_active"] = 1
        after["affected_job_ready"] = 1
        after["affected_job_running"] = 1
        for slot in range(bounds.job_slots):
            key = f"job_{slot}_key"
            if before[key] == env.selected_job_key:
                after["budget"] = before[f"job_{slot}_budget"]
                after["release"] = before[f"job_{slot}_release"]
                after["deadline"] = before[f"job_{slot}_deadline"]
                after["job_key"] = env.selected_job_key
                after["affected_job_priority"] = before[f"job_{slot}_priority"]
                after["affected_job_deadline"] = before[f"job_{slot}_deadline"]
                after["affected_job_budget"] = before[f"job_{slot}_budget"]
                after["affected_job_demand"] = before[f"job_{slot}_demand"]
                after[f"job_{slot}_running"] = 1
            else:
                after[f"job_{slot}_running"] = 0
    elif case_id == "CONTROLLER_SELECTED_ACTION":
        after["future_budget"] = env.release_budget
        after["affected_task_budget"] = env.release_budget
        for slot in range(bounds.task_slots):
            if slot == env.update_target_slot:
                after[f"task_{slot}_future_budget"] = env.release_budget
    elif case_id in {"BOOT_TO_PRECLOSED_0", "ARRIVAL_BATCH_NO_SWITCH",
                     "CONTROLLER_NO_ACTION", "DEADLINE_OBSERVATION_NO_MISS"}:
        pass
    else:
        raise ValueError(f"REFERENCE_P0_UNKNOWN_CASE_ID:{case_id}")
    return after


def validate_reference_p0_contract(
    bounds: P0ModelBounds,
) -> dict[str, Any]:
    symbolic_hashes = {}

    for case_id in REFERENCE_P0_CASE_IDS:
        rendered = render_reference_p0_delta(
            case_id,
            bounds,
        )

        for field in p0_smt_relation_fields(
            bounds
        ):
            if f"r_{field}_post" not in rendered:
                raise ValueError(
                    "REFERENCE_P0_FIELD_NOT_TOTAL:"
                    f"{case_id}:{field}"
                )

        symbolic_hashes[case_id] = sha256_object(
            rendered
        )

    return {
        "status": "PASS",
        "schema_version":
            "reference_p0_transition_contract_v3",
        "transition_system_id":
            "FIXED_EXECUTABLE_REFERENCE_P0_V3",
        "case_ids": list(REFERENCE_P0_CASE_IDS),
        "symbolic_delta_hashes": symbolic_hashes,
        "state_relation_fields": list(
            p0_smt_relation_fields(
                bounds
            )
        ),
        "model_bounds_hash": bounds.fingerprint,
    }
