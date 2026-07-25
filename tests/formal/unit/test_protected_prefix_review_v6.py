from __future__ import annotations

from formal_toolchain.reference.protected_priority_prefix.bad_prefix_reflection import (
    derive_hi_bad_prefix_reflection,
)
from formal_toolchain.reference.protected_priority_prefix.phase_relation import check_phase_relation
from formal_toolchain.reference.protected_priority_prefix.pp0_checker import check_pp0_transition_queries
from formal_toolchain.reference.protected_priority_prefix.pp0_transition_ir import transition_ir_map
from formal_toolchain.routes.registry import resolve_registry


def _observable(service: int = 0):
    return {
        "time": 4,
        "jobs": ({
            "job_key": ("H1", 0), "task_name": "H1", "criticality": "HI",
            "release_time": 0, "absolute_deadline": 10, "priority_index": 0,
            "actual_demand": 3, "hi_class": "ABNORMAL",
            "executed_service": service, "active": True, "ready": False,
            "running": True, "completed": False, "missed": False,
        },),
        "running_job_key": ("H1", 0),
        "miss_job_keys": (),
    }


def test_phase_relation_compares_nested_job_contents():
    result = check_phase_relation(_observable(1), _observable(2), "Close")
    assert result["status"] == "FAIL"
    assert any("executed_service" in name for name in result["failed_fields"])


def test_phase_relation_rejects_excluded_state_leakage():
    full = _observable(1)
    full["global_mode"] = "HI"
    result = check_phase_relation(full, _observable(1), "Close")
    assert result["status"] == "FAIL"
    assert "excluded_global_mode" in result["failed_fields"]


def test_pp0_handwritten_ir_is_not_code_bound():
    ir = transition_ir_map()["RECOVERY"]
    assert ir.guard_formula.startswith("(and (= mode HI)")
    assert ir.state_equations[0].rhs == "LO"
    assert ir.binding_kind == "HAND_WRITTEN_SCHEMA_ONLY"
    assert ir.binding_kind != "EXECUTABLE_TRANSITION_COMPILER"
    report = check_pp0_transition_queries()
    # The legacy hand-written IR remains non-authoritative.  The PASS report is
    # produced by the separate path-sensitive executable compiler and direct
    # PP0 encoder, never by transition_ir_map().
    assert report["status"] == "PASS"
    assert report["code_bound_query_count"] == 12
    assert report["pass_count"] == 12
    assert all(row["direct_executable_encoding"] is True for row in report["receipt_results"])


def test_simulation_domain_depends_on_demand_receptiveness():
    resolved = resolve_registry("protected_prefix")
    by_id = {entry["id"]: entry for entry in resolved.entries}
    deps = set(by_id["FULL_TO_PREFIX_SIMULATION_DOMAIN"]["depends_on"])
    assert "PROTECTED_INPUT_DEMAND_RECEPTIVENESS" in deps


def test_narrative_reflection_receipts_do_not_prove_theorem():
    receipt = {"status": "PASS"}
    result = derive_hi_bad_prefix_reflection(
        simulation_receipt=receipt,
        observable_schema_receipt=receipt,
        deadline_batch_receipt=receipt,
    )
    assert result["status"] == "UNRESOLVED"
