from __future__ import annotations

from types import SimpleNamespace

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.reference.protected_priority_prefix.construction import (
    build_saturated_protected_prefix,
)
from formal_toolchain.reference.protected_priority_prefix.projected_oracle_theorem import (
    PROJECTION_QUANTIFIER,
    build_symbolic_projection_theorem,
    build_symbolic_demand_receptiveness_theorem,
)
from formal_toolchain.reference.protected_priority_prefix.full_execution_input_view import (
    build_symbolic_full_execution_input_theorem,
)
from formal_toolchain.reference.protected_priority_prefix.batch_cursor import (
    construct_batch_cursor,
)
from formal_toolchain.reference.protected_priority_prefix.macro_step import (
    prove_protected_service_correspondence,
    prove_completion_removal_correspondence,
)
from formal_toolchain.routes.protected_prefix import ROUTE
from formal_toolchain.routes.protected_prefix_checkers import check_lo_saturation
from formal_toolchain.routes.registry import resolve_registry
from formal_toolchain.theory.backends.protected_prefix_idle_jump import (
    verify_idle_jump_expansion_receipt,
)


def _full_and_construction():
    full = ReferenceTaskset((
        ReferenceTask("lo", 20, 20, 4, 2, "LO", 0, 4, 4, 2, 0),
        ReferenceTask("hi", 25, 25, 2, 5, "HI", 1, 2, 5, None, 0),
        ReferenceTask("tail", 40, 40, 3, 1, "LO", 2, 3, 3, 1, 0),
    ), "a" * 64)
    construction = build_saturated_protected_prefix(
        full, source_context_hash="a" * 64,
    )
    return full, construction


def _full_input_contract(full):
    payload = {
        "theorem_id": "FULL_REFERENCE_RECURRING_INPUT_ORACLE",
        "status": "PASS",
        "reference_taskset_fingerprint": full.to_dict()["fingerprint"],
        "forall_full_reference_executions": True,
        "unique_release_record_for_every_job_key": True,
        "release_fixed_actual_demand": True,
        "infinite_recurring_domain": True,
        "demand_not_regenerated_from_wcet": True,
        "demand_contract_complete": True,
        "lo_demand_le_reference_c_lo": True,
        "normal_hi_demand_le_reference_c_lo": True,
        "abnormal_hi_demand_in_lo_hi_interval": True,
        "finite_instance_data_used": False,
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload


def test_lo_saturation_checker_exports_equality_witness():
    full, construction = _full_and_construction()
    prepared = ROUTE.prepare_analysis(
        full_reference_taskset=full,
        reference_context_hash="a" * 64,
    )
    state = SimpleNamespace(
        prepared_route=prepared,
        full_reference_taskset=full,
    )
    result = check_lo_saturation(fresh_state=state)
    assert result["status"] == "PASS"
    assert result["witness"] == construction.saturation_witness
    assert result["witness"]["lo_saturation_equalities"]


def test_projection_has_only_stream_quantifier_not_weak_simulation_quantifier():
    full, construction = _full_and_construction()
    receipt = build_symbolic_projection_theorem(
        full_theorem=_full_input_contract(full),
        protected_task_names=frozenset(construction.protected_task_names),
        full_taskset=full,
        prefix_taskset=construction.prefix_taskset,
        saturation_witness=construction.saturation_witness,
        saturation_certificate_hash="b" * 64,
    )
    assert receipt["status"] == "PASS"
    assert receipt["quantifier_scope"] == PROJECTION_QUANTIFIER
    assert "quantifier_order" not in receipt


def test_demand_receptiveness_requires_actual_lo_saturation_equalities():
    full, construction = _full_and_construction()
    projection = build_symbolic_projection_theorem(
        full_theorem=_full_input_contract(full),
        protected_task_names=frozenset(construction.protected_task_names),
        full_taskset=full,
        prefix_taskset=construction.prefix_taskset,
        saturation_witness=construction.saturation_witness,
        saturation_certificate_hash="b" * 64,
    )
    missing = build_symbolic_demand_receptiveness_theorem(
        full_taskset=full,
        prefix_taskset=construction.prefix_taskset,
        projection_theorem=projection,
        saturation_witness={},
    )
    assert missing["status"] == "UNRESOLVED"
    proven = build_symbolic_demand_receptiveness_theorem(
        full_taskset=full,
        prefix_taskset=construction.prefix_taskset,
        projection_theorem=projection,
        saturation_witness=construction.saturation_witness,
    )
    assert proven["status"] == "PASS"
    assert proven["lo_saturation_equalities_verified"] is True


def test_full_input_theorem_needs_prefix_extension_and_standard_semantics():
    full, _ = _full_and_construction()
    base_rows = [
        {"condition_id": "FINITE_INDEPENDENT_PERIODIC_SUBLANGUAGE", "passed": True},
        {"condition_id": "RELEASE_FIXED_DEMAND_DOMINATION", "passed": True},
    ]
    missing = build_symbolic_full_execution_input_theorem(
        full_taskset=full, conformance_witness={"condition_results": base_rows},
    )
    assert missing["status"] == "UNRESOLVED"
    rows = base_rows + [
        {"condition_id": "REFERENCE_PREFIX_EXTENSIBILITY", "passed": True},
        {"condition_id": "STANDARD_EMPTY_LO_INITIALIZATION", "passed": True},
        {"condition_id": "REFERENCE_TRANSITION_SYSTEM_IDENTITY", "passed": True},
    ]
    complete = build_symbolic_full_execution_input_theorem(
        full_taskset=full, conformance_witness={"condition_results": rows},
    )
    assert complete["status"] == "PASS"
    assert complete["demand_contract_complete"] is True


def test_prefix_model_conformance_is_model_level_not_simulation_witness_dependent():
    by_id = {item["id"]: item for item in resolve_registry("protected_prefix").entries}
    deps = set(by_id["PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE"]["depends_on"])
    assert "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION" in deps
    assert "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS" not in deps


def test_legacy_batch_cursor_keyword_no_longer_raises_type_error():
    cursor, proof = construct_batch_cursor([], [], frozenset({"hi"}), "DDLCursor",
                                           proof_kernel_receipt={"status": "PASS"})
    assert cursor.measure == 0
    assert proof.base_case_valid is True


def test_idle_jump_backend_rejects_self_asserted_booleans():
    receipt = {
        "theorem_id": "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        "status": "PASS",
        "proof_scope": "ALL_LEGAL_CLOSED_IDLE_JUMPS",
        "parameterized": True,
        "source_bound_transition_relation": True,
        "all_integer_times_observable": True,
        "time_indexed_closed_observation_defined": True,
        "protected_observable_stutters_on_expanded_idle_ticks": True,
        "independent_of_complete_execution_witness": True,
        "receipt_hash": "self-declared",
    }
    checked = verify_idle_jump_expansion_receipt(receipt)
    assert checked["status"] == "UNRESOLVED"
    assert checked["code"] == "IDLE_JUMP_SOURCE_BOUND_RECEIPT_REQUIRED"


def test_local_pp0_receipt_alone_does_not_prove_relational_l3_or_l4():
    _, construction = _full_and_construction()
    pp0 = {
        "SERVICE_UNIT": {"status": "PASS"},
        "TAIL_ONLY_SERVICE": {"status": "PASS"},
        "REM_COMPLETION": {"status": "PASS"},
    }
    idle = {
        "status": "PASS", "parameterized": True,
        "independent_of_complete_execution_witness": True,
    }
    l3 = prove_protected_service_correspondence(
        construction=construction, pp0_receipts=pp0, idle_jump_receipt=idle,
    )
    l4 = prove_completion_removal_correspondence(
        construction=construction, pp0_receipts=pp0,
    )
    assert l3["status"] == "UNRESOLVED"
    assert l3["relational_kernel_consumed"] is False
    assert l4["status"] == "UNRESOLVED"
    assert l4["relational_kernel_consumed"] is False


def test_phase_relation_compares_pending_protected_release_payload():
    from formal_toolchain.reference.protected_priority_prefix.phase_relation import (
        check_phase_relation,
    )
    base = {
        "time": 5,
        "jobs": [],
        "running_job_key": None,
        "miss_job_keys": [],
        "pending_releases": [{
            "job_key": ("hi", 1),
            "task_name": "hi",
            "criticality": "HI",
            "release_time": 5,
            "absolute_deadline": 30,
            "priority_index": 1,
            "actual_demand": 5,
            "hi_class": "ABNORMAL",
        }],
    }
    changed = {
        **base,
        "pending_releases": [{**base["pending_releases"][0], "actual_demand": 4}],
    }
    receipt = check_phase_relation(base, changed, "ARRCursor")
    assert receipt["status"] == "FAIL"
    assert any("pending" in field for field in receipt["failed_fields"])
