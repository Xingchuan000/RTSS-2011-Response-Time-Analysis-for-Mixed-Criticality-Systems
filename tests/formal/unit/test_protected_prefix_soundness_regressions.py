from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from formal_toolchain.reference.protected_priority_prefix.construction import build_saturated_protected_prefix
from formal_toolchain.reference.protected_priority_prefix.observable import _hi_class
from formal_toolchain.reference.protected_priority_prefix.batch_cursor import (
    BatchCursor, BatchCursorProof, construct_batch_cursor, verify_batch_cursor_proof,
)
from formal_toolchain.reference.protected_priority_prefix.transition_schema import (
    CANONICAL_CASES, PrimitiveTransitionSchema, canonical_case_ids,
)
from formal_toolchain.reference.protected_priority_prefix.model_conformance import (
    derive_prefix_model_conformance, PrefixModelConformanceWitness,
)
from formal_toolchain.reference.protected_priority_prefix.runtime_schema import (
    build_runtime_schema_certificate,
)
from formal_toolchain.reference.rta_production import all_task_protected_prefix_rta
from formal_toolchain.reference.rta_replay import replay_all_task_rta
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.routes.protected_prefix_checkers import (
    check_runtime_schema_conformance, check_weak_forward_simulation,
    check_hi_bad_prefix_reflection, check_mathematical_conformance,
    check_prefix_rta,
)
from formal_toolchain.routes.registry import resolve_registry
from formal_toolchain.core.hashing import sha256_object


def _pass_predecessor():
    return {"obligation_status": "PASS"}


def test_theorem_receipts_cannot_close_quantified_pp_obligations():
    runtime = check_runtime_schema_conformance(
        verified_predecessors={"SATURATED_PROTECTED_PREFIX_REFERENCE": _pass_predecessor()}
    )
    assert runtime["status"] == "UNRESOLVED"
    assert runtime["code"] == "PROTECTED_PREFIX_RUNTIME_SCHEMA_PARAMETRIC_PROOF_MISSING"

    weak = check_weak_forward_simulation(
        verified_predecessors={"FULL_TO_PREFIX_SIMULATION_DOMAIN": _pass_predecessor()}
    )
    assert weak["status"] == "UNRESOLVED"

    reflection = check_hi_bad_prefix_reflection(
        verified_predecessors={"PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED": _pass_predecessor()}
    )
    assert reflection["status"] == "UNRESOLVED"


def test_rta_replay_does_not_pass_an_unschedulable_prefix():
    taskset = ReferenceTaskset((
        ReferenceTask("hi", 10, 1, 2, 3, "HI", 0, 2, 3, None, 0),
    ), "a" * 64)
    production = all_task_protected_prefix_rta(
        taskset, certificate_context_hash="b" * 64
    )
    assert production["status"] == "FAIL"
    replay = replay_all_task_rta(
        taskset, production,
        expected_obligation_id="PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        expected_route_id="protected_prefix",
    )
    assert replay["status"] == "FAIL"
    assert replay["code"] == "ALL_TASK_SUFFICIENT_TEST_FAILED"
    assert replay["witness"]["all_deadlines_met"] is False


def test_protected_observable_forgets_trigger_identity_but_keeps_hi_class():
    assert _hi_class("HI", "HI_ABNORMAL_SWITCH_TRIGGER") == "ABNORMAL"
    assert _hi_class("HI", "HI_ABNORMAL") == "ABNORMAL"
    assert _hi_class("HI", "HI_NORMAL") == "NORMAL"
    assert _hi_class("LO", "LO_PRIMARY_NORMAL") is None


# ---------------------------------------------------------------------------
# Section 14.1 必测反例
# ---------------------------------------------------------------------------

def test_case_1_two_abnormal_hi_same_arrival_both_keep_abnormal_class():
    assert _hi_class("HI", "HI_ABNORMAL_SWITCH_TRIGGER") == "ABNORMAL"
    assert _hi_class("HI", "HI_ABNORMAL") == "ABNORMAL"


def test_case_2_normal_hi_in_hi_mode_still_uses_c_lo_demand():
    taskset = ReferenceTaskset((
        ReferenceTask("hi_task", 20, 3, 5, 10, "HI", 0, 1, 2, None, 0),
    ), "c" * 64)
    result = build_saturated_protected_prefix(taskset, source_context_hash="d" * 64)
    for task in result.prefix_taskset.tasks:
        if task.criticality == "HI":
            assert task.c_lo == taskset.tasks[0].c_lo


def test_case_3_two_abnormal_hi_same_batch_one_trigger_label():
    t1 = ReferenceTask("t1", 10, 1, 2, 3, "HI", 0, 1, 2, None, 0)
    t2 = ReferenceTask("t2", 10, 1, 2, 3, "HI", 1, 1, 2, None, 0)
    taskset = ReferenceTaskset((t1, t2), "0" * 64)
    result = build_saturated_protected_prefix(taskset, source_context_hash="0" * 64)
    assert len(result.protected_task_names) >= 2


def test_case_4_tail_service_exclusion_priority_order():
    t_hi = ReferenceTask("hi", 10, 1, 2, 3, "HI", 0, 1, 2, None, 0)
    t_lo1 = ReferenceTask("lo1", 20, 5, 10, 10, "LO", 1, 1, 2, None, 0)
    t_lo2 = ReferenceTask("lo2", 30, 10, 20, 20, "LO", 2, 1, 2, None, 0)
    taskset = ReferenceTaskset((t_hi, t_lo1, t_lo2), "0" * 64)
    result = build_saturated_protected_prefix(taskset, source_context_hash="0" * 64)
    assert result.cutoff_task_name == "hi"
    assert all(t in result.tail_task_names for t in ("lo1", "lo2"))


def test_case_5_protected_empty_tail_service_is_stutter():
    t_hi = ReferenceTask("hi", 15, 5, 2, 5, "HI", 0, 2, 5, None, 0)
    t_lo = ReferenceTask("lo1", 10, 10, 5, 5, "LO", 1, 5, 5, None, 0)
    taskset = ReferenceTaskset((t_hi, t_lo), "0" * 64)
    result = build_saturated_protected_prefix(taskset, source_context_hash="0" * 64)
    assert result.cutoff_task_name == "hi"
    assert "lo1" in result.tail_task_names


def test_case_6_completion_before_ddl_no_phantom_miss():
    t = ReferenceTask("t1", 10, 1, 5, 5, "HI", 0, 1, 2, None, 0)
    taskset = ReferenceTaskset((t,), "0" * 64)
    production = all_task_protected_prefix_rta(taskset, certificate_context_hash="0" * 64)
    assert "status" in production


def test_case_7_deadline_batch_cursor_skips_tail():
    full_entries = [
        {"job_key": ("lo_tail", 0), "task_name": "lo_tail", "release_time": 0, "actual_demand": 5, "hi_class": None},
        {"job_key": ("hi", 0), "task_name": "hi", "release_time": 0, "actual_demand": 3, "hi_class": "NORMAL"},
    ]
    prefix_entries = [
        {"job_key": ("hi", 0), "task_name": "hi", "release_time": 0, "actual_demand": 3, "hi_class": "NORMAL"},
    ]
    cursor, proof = construct_batch_cursor(
        full_entries, prefix_entries, frozenset({"hi"}), "DDLCursor",
    )
    assert cursor.tail_skip_count == 1
    assert proof.protected_entry_correspondence is True
    verification = verify_batch_cursor_proof(cursor, proof)
    assert verification["status"] == "PASS"


def test_case_8_arrival_batch_protected_independent_of_tail():
    full_entries = [
        {"job_key": ("hi1", 0), "task_name": "hi1", "release_time": 0, "actual_demand": 2, "hi_class": "NORMAL"},
        {"job_key": ("lo_tail", 0), "task_name": "lo_tail", "release_time": 0, "actual_demand": 10, "hi_class": None},
        {"job_key": ("hi2", 0), "task_name": "hi2", "release_time": 0, "actual_demand": 3, "hi_class": "ABNORMAL"},
        {"job_key": ("lo_tail2", 0), "task_name": "lo_tail2", "release_time": 0, "actual_demand": 8, "hi_class": None},
    ]
    prefix_entries = [
        {"job_key": ("hi1", 0), "task_name": "hi1", "release_time": 0, "actual_demand": 2, "hi_class": "NORMAL"},
        {"job_key": ("hi2", 0), "task_name": "hi2", "release_time": 0, "actual_demand": 3, "hi_class": "ABNORMAL"},
    ]
    cursor, proof = construct_batch_cursor(
        full_entries, prefix_entries, frozenset({"hi1", "hi2"}), "ARRCursor",
    )
    assert cursor.tail_skip_count == 2
    assert proof.protected_entry_correspondence is True
    assert cursor.k_full == cursor.full_batch_size
    assert cursor.k_prefix == cursor.prefix_batch_size
    assert cursor.measure == 0


def test_case_9_any_task_rta_fails_route_must_fail():
    taskset = ReferenceTaskset((
        ReferenceTask("hi", 10, 5, 10, 20, "HI", 0, 10, 20, None, 0),
    ), "0" * 64)
    production = all_task_protected_prefix_rta(taskset, certificate_context_hash="0" * 64)
    assert production["status"] == "FAIL"
    replay = replay_all_task_rta(
        taskset, production,
        expected_obligation_id="PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        expected_route_id="protected_prefix",
    )
    assert replay["status"] == "FAIL"


def test_case_10_theorem_backend_rejects_bare_true_lemma_without_receipt():
    from formal_toolchain.theory.backends.protected_prefix_simulation import (
        ProtectedPrefixSimulationBackend,
    )
    from formal_toolchain.reference.protected_priority_prefix.observable import observable_schema

    backend = ProtectedPrefixSimulationBackend()
    theorem = {
        "theorem_id": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
        "exact_statement": "test", "conclusion": "test", "source_reference": "test",
        "assurance_level": "MACHINE_CHECKED_PROJECT_LEMMA", "version": "1",
        "assumptions": [], "premise_obligation_ids": [],
    }
    stmt = {k: theorem[k] for k in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
    asmp = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}

    fake_proof = {
        "theorem_id": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
        "theorem_statement_hash": sha256_object(stmt),
        "theorem_assumption_hash": sha256_object(asmp),
        "quantification": "forall full execution exists one prefix execution forall natural-number closed boundaries",
        "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
        "required_lemmas": list(backend.REQUIRED_LEMMAS),
        "protected_observable_schema_hash": sha256_object(observable_schema()),
        "dependencies": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(fake_proof, f)
        path = f.name
    try:
        result = backend.verify(Path(path), theorem=theorem)
        assert result["status"] == "UNRESOLVED"
        assert "SIMULATION_DEPENDENCY_RECEIPTS_MISSING" in str(result.get("code", ""))
    finally:
        os.unlink(path)


def test_case_11_single_witness_compatibility_required():
    from formal_toolchain.theory.backends.protected_prefix_simulation import (
        ProtectedPrefixSimulationBackend,
    )
    from formal_toolchain.reference.protected_priority_prefix.observable import observable_schema

    backend = ProtectedPrefixSimulationBackend()
    theorem = {
        "theorem_id": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
        "exact_statement": "test", "conclusion": "test", "source_reference": "test",
        "assurance_level": "MACHINE_CHECKED_PROJECT_LEMMA", "version": "1",
        "assumptions": [], "premise_obligation_ids": [],
    }
    stmt = {k: theorem[k] for k in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
    asmp = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}

    fake_proof = {
        "theorem_id": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
        "theorem_statement_hash": sha256_object(stmt),
        "theorem_assumption_hash": sha256_object(asmp),
        "quantification": "forall full execution exists one prefix execution forall natural-number closed boundaries",
        "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
        "required_lemmas": list(backend.REQUIRED_LEMMAS),
        "protected_observable_schema_hash": sha256_object(observable_schema()),
        "dependencies": {dep: {"status": "PASS", "receipt_hash": "0" * 64} for dep in backend.REQUIRED_DEPENDENCIES},
        "single_witness_compatibility": {"status": "UNRESOLVED"},
        "full_taskset_fingerprint": "0" * 64,
        "prefix_taskset_fingerprint": "0" * 64,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(fake_proof, f)
        path = f.name
    try:
        result = backend.verify(Path(path), theorem=theorem)
        assert result["status"] == "UNRESOLVED"
        assert "COMPLETE_EXECUTION_WITNESS_UNVERIFIED" in str(result.get("code", ""))
    finally:
        os.unlink(path)


def test_case_12_transition_schema_cases_complete():
    assert canonical_case_ids() == (
        "REM_COMPLETION", "RECOVERY", "DEADLINE_OBSERVATION", "ARRIVAL_BATCH",
        "MODE_SWITCH", "RELEASE", "FINAL_DISPATCH", "SERVICE_UNIT", "TAIL_ONLY_SERVICE",
    )
    for case in CANONICAL_CASES:
        assert isinstance(case, PrimitiveTransitionSchema)
        assert len(case.protected_frame_fields) > 0


def test_model_conformance_derives_all_witness_fields():
    t_hi = ReferenceTask("hi", 20, 3, 5, 10, "HI", 0, 1, 2, None, 0)
    t_lo = ReferenceTask("lo", 30, 10, 15, 15, "LO", 1, 1, 2, None, 0)
    full = ReferenceTaskset((t_hi, t_lo), "0" * 64)
    result = build_saturated_protected_prefix(full, source_context_hash="0" * 64)
    runtime = build_runtime_schema_certificate()
    conformance = derive_prefix_model_conformance(
        full_taskset=full, prefix_taskset=result.prefix_taskset,
        construction=result, runtime_schema_certificate=runtime,
    )
    assert "witness" in conformance
    witness = conformance["witness"]
    assert isinstance(witness, dict)
    expected_fields = {f.name for f in PrefixModelConformanceWitness.__dataclass_fields__.values()}
    assert set(witness) == expected_fields
    assert witness["finite_nonempty_taskset"] is True
    assert witness["constrained_deadlines"] is True
    assert witness["strict_total_priority_order"] is True
    assert witness["all_hi_tasks_preserved"] is True


def test_registry_dag_includes_prefix_model_conformance():
    registry = resolve_registry("protected_prefix")
    ids = {str(e["id"]) for e in registry.entries}
    assert "PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE" in ids
    assert "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE" in ids
    entry = next(e for e in registry.entries if e["id"] == "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE")
    deps = entry.get("depends_on", [])
    assert "PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE" in deps
    assert "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC" in deps
    assert "THEORY_LIBRARY_VERSION" in deps
    assert "REFERENCE_MODEL_CONFORMANCE" not in deps


def test_theory_manifest_has_route_specific_structure():
    manifest_path = Path(__file__).parents[3] / "formal_toolchain" / "theory" / "theory_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "common_required_theorems" in manifest
    assert "route_required_theorems" in manifest
    assert manifest["schema_version"] == "theory_manifest_v4"
    pp_theorems = manifest["route_required_theorems"].get("protected_prefix", [])
    assert "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION" in pp_theorems
    assert "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION" in pp_theorems
    assert "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX" in pp_theorems
    sf_theorems = manifest["route_required_theorems"].get("strict_full", [])
    assert len(sf_theorems) == 0


def test_bad_prefix_backend_rejects_bare_true_fields():
    from formal_toolchain.theory.backends.protected_prefix_bad_prefix import (
        ProtectedPrefixBadPrefixBackend,
    )

    backend = ProtectedPrefixBadPrefixBackend()
    theorem = {
        "theorem_id": "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
        "exact_statement": "test", "conclusion": "test", "source_reference": "test",
        "assurance_level": "MACHINE_CHECKED_PROJECT_LEMMA", "version": "1",
        "assumptions": [], "premise_obligation_ids": [],
    }
    stmt = {k: theorem[k] for k in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
    asmp = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}

    fake_proof = {
        "theorem_id": "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
        "theorem_statement_hash": sha256_object(stmt),
        "theorem_assumption_hash": sha256_object(asmp),
        "global_mode_equality_required": False,
        "reflection_fields": {
            "same_job_key": True, "same_absolute_deadline": True,
            "same_actual_demand": True, "same_service_at_deadline": True,
            "same_miss_ledger_membership": True,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(fake_proof, f)
        path = f.name
    try:
        result = backend.verify(Path(path), theorem=theorem)
        assert result["status"] == "UNRESOLVED"
        assert "SIMULATION_RECEIPT_NOT_PASS" in str(result.get("code", ""))
    finally:
        os.unlink(path)


def test_safety_backend_rejects_missing_component_hashes():
    from formal_toolchain.theory.backends.protected_prefix_safety import (
        ProtectedPrefixSafetyBackend,
    )

    backend = ProtectedPrefixSafetyBackend()
    theorem = {
        "theorem_id": "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
        "exact_statement": "test", "conclusion": "test", "source_reference": "test",
        "assurance_level": "MACHINE_CHECKED_PROJECT_LEMMA", "version": "1",
        "assumptions": [], "premise_obligation_ids": [],
    }
    stmt = {k: theorem[k] for k in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
    asmp = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}

    fake_proof = {
        "theorem_id": "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
        "theorem_statement_hash": sha256_object(stmt),
        "theorem_assumption_hash": sha256_object(asmp),
        "conclusion": "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES",
        "contradiction_steps": [
            "full reference HI miss", "reflected prefix HI miss",
            "prefix all-task schedulability contradiction",
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(fake_proof, f)
        path = f.name
    try:
        result = backend.verify(Path(path), theorem=theorem)
        assert result["status"] == "UNRESOLVED"
        assert "COMPONENT_HASHES_MISSING" in str(result.get("code", ""))
    finally:
        os.unlink(path)
