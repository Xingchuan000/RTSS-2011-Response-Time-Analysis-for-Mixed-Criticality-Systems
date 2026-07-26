from copy import deepcopy

from formal_toolchain.binding.removal_binding import bind_removal_runtime
from formal_toolchain.conformance.micro_scenarios import (
    SCENARIO_ASSERTION_CONTRACTS,
    run_p0_micro_scenarios,
)
from formal_toolchain.conformance.required_obligations import check_hi_execution_contract
from formal_toolchain.semantics.frozen_runtime_contract import CONTRACT_VERSION


def _runtime(root):
    return {
        "micro_scenarios": run_p0_micro_scenarios(target_available=True),
        "removal_binding": bind_removal_runtime(root),
    }


def test_hi_execution_contract_passes_for_frozen_semantics(tmp_path):
    # The binder needs the repository root, not the pytest tmp directory.
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    runtime = _runtime(root)
    scenario = runtime["micro_scenarios"]["scenarios"]["hi_nontruncation"]
    assertions = scenario["assertions"]
    assert scenario["formal_semantics_contract"] == CONTRACT_VERSION
    assert all(assertions[key] is True for key in SCENARIO_ASSERTION_CONTRACTS["hi_nontruncation"])
    result = check_hi_execution_contract(runtime)
    assert result["status"] == "PASS"
    assert result["hi_not_dropped"] is True


def test_hi_execution_contract_rejects_missing_or_false_lifecycle_fact():
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    runtime = _runtime(root)
    for mutation in ("missing", "false"):
        changed = deepcopy(runtime)
        assertions = changed["micro_scenarios"]["scenarios"]["hi_nontruncation"]["assertions"]
        if mutation == "missing":
            assertions.pop("hi_not_dropped")
        else:
            assertions["hi_not_dropped"] = False
        result = check_hi_execution_contract(changed)
        assert result["status"] == "FAIL"
        assert "scenario_assertion:hi_not_dropped" in result["failed"]


def test_hi_execution_contract_rejects_mutable_runtime_dependency():
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    runtime = _runtime(root)
    runtime["micro_scenarios"]["scenarios"]["hi_nontruncation"]["mutable_runtime_dependency"] = "SHARED_RUNTIME"
    result = check_hi_execution_contract(runtime)
    assert result["status"] == "FAIL"
    assert "scenario_runtime_independent" in result["failed"]


def test_handler_decomposition_schema_is_single_sourced():
    from pathlib import Path
    from formal_toolchain.bridge.handler_decomposition import (
        HANDLER_DECOMPOSITION_SCHEMA_VERSION,
    )

    root = Path(__file__).resolve().parents[3]
    assert HANDLER_DECOMPOSITION_SCHEMA_VERSION == "handler_decomposition_v4_frozen_semantics"
    for relative in (
        "formal_toolchain/bridge/compile_bridge.py",
        "formal_toolchain/bridge/prefix_refinement.py",
        "formal_toolchain/verifier/bridge_replay.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "handler_decomposition_v3_math_fixed" not in source
        assert "HANDLER_DECOMPOSITION_SCHEMA_VERSION" in source


def test_closed_prefix_verifier_accepts_standard_metadata_kwargs():
    from formal_toolchain.verifier.bridge_proof_checker import _verify_cases

    result = _verify_cases(
        {},
        "CLOSED_PREFIX_REFINEMENT",
        "0" * 64,
        contexts={},
        predecessors={},
    )
    assert result["status"] == "UNRESOLVED"
    assert result["code"] == "BRIDGE_PROOF_OBJECT_ID_MISMATCH"


def test_closed_prefix_verifier_accepts_certificate_envelope_with_v2_witness(monkeypatch):
    """The outer certificate envelope is not the closed-prefix witness schema."""
    from formal_toolchain.bridge.prefix_refinement import (
        CLOSED_PREFIX_REFINEMENT_WITNESS_SCHEMA_VERSION,
    )
    from formal_toolchain.verifier import bridge_proof_checker as checker

    candidate = {
        "artifact_schema_version": "certificate_envelope_v2",
        "witness": {
            "schema_version": CLOSED_PREFIX_REFINEMENT_WITNESS_SCHEMA_VERSION,
            "parameterized_relation_schema_hash": checker.parameterized_state_relation_schema_hash(),
            "pointwise_closed_prefix_relation": True,
            "theorem_proof_receipt_hash": "0" * 64,
        },
    }
    # Stop immediately after the schema gates. A legacy-schema result here would
    # mean the verifier has again confused the envelope with the witness.
    result = checker._verify_universal_closed_prefix(candidate, "1" * 64)
    assert result.get("code") != "CLOSED_PREFIX_LEGACY_SCHEMA_REJECTED"
    assert result.get("code") != "CLOSED_PREFIX_UNKNOWN_SCHEMA"


def test_closed_prefix_verifier_rejects_only_legacy_witness_schema():
    from formal_toolchain.verifier import bridge_proof_checker as checker

    candidate = {
        "artifact_schema_version": "certificate_envelope_v2",
        "witness": {"schema_version": "closed_prefix_refinement_v1"},
    }
    result = checker._verify_universal_closed_prefix(candidate, "1" * 64)
    assert result == {
        "status": "FAIL",
        "route": "PROOF_BUNDLE_INVALID",
        "code": "CLOSED_PREFIX_LEGACY_SCHEMA_REJECTED",
    }


def test_all_task_rta_soundness_receipt_is_shared_by_fresh_and_route_checks():
    from formal_toolchain.reference.rta_production import all_task_protected_prefix_rta
    from formal_toolchain.reference.rta_replay import replay_all_task_rta
    from formal_toolchain.reference.rta_soundness import derive_all_task_rta_soundness
    from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset

    taskset = ReferenceTaskset((
        ReferenceTask("hi", 10, 10, 1, 2, "HI", 0, 1, 2, None, 0),
    ), "a" * 64)
    production = all_task_protected_prefix_rta(
        taskset, certificate_context_hash="b" * 64
    )
    replay = replay_all_task_rta(
        taskset,
        production,
        expected_obligation_id="PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        expected_route_id="protected_prefix",
    )
    derived = derive_all_task_rta_soundness(
        replay=replay,
        taskset=taskset,
        theorem_id="PREFIX_ALL_TASK_RTA_SOUNDNESS",
    )
    assert derived["status"] == "PASS"
    assert derived["soundness_receipt"]["status"] == "PASS"
    assert derived["soundness_receipt"]["all_task_name_set"] == ["hi"]
