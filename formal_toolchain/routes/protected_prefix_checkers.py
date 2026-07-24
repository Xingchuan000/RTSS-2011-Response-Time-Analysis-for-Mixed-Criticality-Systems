"""Fresh-verifier checks for the executable prefix scaffold."""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.reference.protected_priority_prefix.certificates import verify_construction_witness


def _finish(status: str, code: str | None = None, witness: Mapping[str, Any] | None = None):
    return {"status": status, "route": None if status == "PASS" else "UNRESOLVED",
            "code": code, "witness": dict(witness or {})}


def _route_state(kwargs: Mapping[str, Any]):
    ctx = kwargs.get("context")
    return getattr(ctx, "fresh_state", None) or kwargs.get("fresh_state")


def check_partition(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.prepared_route is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    result = state.prepared_route.construction_witnesses.get("build_result")
    certificate = state.route_construction_certificates.get(
        "PROTECTED_PRIORITY_PREFIX_PARTITION", {})
    witness = certificate.get("witness", {})
    structural_flags = (
        witness.get("tail_all_lo") is True
        and witness.get("all_hi_protected") is True
        and witness.get("partition_complete") is True
        and witness.get("order_preserved") is True
    )
    ok = (result is not None
          and verify_construction_witness(result, witness)
          and structural_flags)
    return _finish("PASS" if ok else "FAIL",
                   None if ok else "PROTECTED_PREFIX_PARTITION_INVALID",
                   witness)


def check_saturation(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.prepared_route is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    result = state.prepared_route.construction_witnesses.get("build_result")
    certificate = state.route_construction_certificates.get(
        "SATURATED_PROTECTED_PREFIX_REFERENCE", {})
    witness = certificate.get("witness", {})
    expected = result.saturation_witness if result is not None else {}
    equalities = expected.get("lo_saturation_equalities", [])
    equalities_hold = all(
        row.get("C_pp_LO") == row.get("C_pp_HI") == row.get("C_ref_LO")
        for row in equalities
        if isinstance(row, Mapping)
    )
    ok = (witness == expected
          and expected.get("hi_fields_equal") is True
          and expected.get("timing_fields_equal") is True
          and equalities_hold)
    return _finish("PASS" if ok else "FAIL",
                   None if ok else "PROTECTED_PREFIX_SATURATION_MISMATCH",
                   witness)


def check_parameter_preservation(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.prepared_route is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    result = state.prepared_route.construction_witnesses.get("build_result")
    if result is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    full = state.full_reference_taskset
    prefix = state.analysis_taskset
    protected_count = len(result.protected_task_names)
    if len(prefix.tasks) != protected_count or protected_count != result.cutoff_priority_index + 1:
        return _finish("FAIL", "PROTECTED_PREFIX_PARAMETER_PRESERVATION_FAILED")

    fields_equal = True
    for full_task, prefix_task, expected_name in zip(
            full.tasks[:protected_count], prefix.tasks, result.protected_task_names):
        fields_equal = fields_equal and (
            full_task.name == prefix_task.name == expected_name
            and full_task.period == prefix_task.period
            and full_task.deadline == prefix_task.deadline
            and full_task.offset == prefix_task.offset
            and full_task.priority_index == prefix_task.priority_index
            and full_task.criticality == prefix_task.criticality
            and full_task.code_c_lo == prefix_task.code_c_lo
            and full_task.code_c_hi == prefix_task.code_c_hi
            and full_task.degraded_cost == prefix_task.degraded_cost
            and full_task.c_lo == prefix_task.c_lo
            and (full_task.criticality != "HI" or full_task.c_hi == prefix_task.c_hi)
        )
    ok = (fields_equal
          and tuple(t.name for t in full.tasks[:protected_count])
          == tuple(result.protected_task_names)
          and tuple(t.name for t in full.tasks[protected_count:])
          == tuple(result.tail_task_names))
    return _finish("PASS" if ok else "FAIL",
                   None if ok else "PROTECTED_PREFIX_PARAMETER_PRESERVATION_FAILED")


def check_lo_saturation(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.prepared_route is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    result = state.prepared_route.construction_witnesses.get("build_result")
    if result is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    full_by_name = {task.name: task for task in state.full_reference_taskset.tasks}
    ok = all(
        task.c_hi == task.c_lo == full_by_name[task.name].c_lo
        for task in result.prefix_taskset.tasks
        if task.criticality == "LO"
    )
    return _finish("PASS" if ok else "FAIL",
                   None if ok else "PROTECTED_PREFIX_SATURATION_MISMATCH")


def check_prefix_rta(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.selected_rta_obligation_id != "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC":
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_RTA_STATE_MISSING")
    replay = state.fresh_rta_replay
    witness = replay.get("witness", replay.get("replay", {}).get("witness", {}))
    ok = (replay.get("status") == "PASS"
          and (witness.get("all_tasks_covered") is True
               or replay.get("replay", {}).get("all_tasks_covered") is True))
    return _finish("PASS" if ok else "UNRESOLVED",
                   None if ok else "PROTECTED_PREFIX_RTA_REPLAY_UNRESOLVED",
                   replay)


def check_mathematical_conformance(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"PROTECTED_PREFIX_PARAMETER_PRESERVATION", "PROTECTED_PREFIX_LO_SATURATION",
                "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC", "REFERENCE_MODEL_CONFORMANCE",
                "THEORY_LIBRARY_VERSION"}
    if set(predecessors) != required:
        return _finish("UNRESOLVED", "PREDECESSOR_SET_MISMATCH",
                       {"expected": sorted(required), "actual": sorted(predecessors)})
    if any(item.get("obligation_status") != "PASS" for item in predecessors.values()):
        return _finish("UNRESOLVED", "PREDECESSOR_NOT_PASS")
    return _finish("PASS", witness={"conclusion": "ALL_PROTECTED_PREFIX_TASKS_MEET_DEADLINES"})


def check_selected_safety(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    predecessors = kwargs.get("verified_predecessors", {})
    expected = ("REFERENCE_HI_SUBSET_SAFETY"
                if state and state.selected_route_id == "strict_full"
                else "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX")
    if set(predecessors) != {expected}:
        return _finish("UNRESOLVED", "PREDECESSOR_SET_MISMATCH",
                       {"expected": [expected], "actual": sorted(predecessors)})
    if predecessors[expected].get("obligation_status") != "PASS":
        return _finish("UNRESOLVED", "PREDECESSOR_NOT_PASS")
    full = state.full_reference_taskset.to_dict()["fingerprint"] if state else None
    return _finish("PASS", witness={"route_id": state.selected_route_id,
                                      "source_safety_obligation": expected,
                                      "reference_taskset_fingerprint": full,
                                      "reference_transition_system_id": "FIXED_EXECUTABLE_REFERENCE_P0_V3",
                                      "conclusion": "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES"})
