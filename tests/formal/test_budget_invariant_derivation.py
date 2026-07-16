"""budget_invariant_derivation 的黄金、边界和 fail-closed 测试。"""

from __future__ import annotations

import pytest

from formal_toolchain.bridge.budget_invariant_derivation import (
    _certificate_passed,
    _require_pass,
    _task_criticality,
    derive_budget_invariant_evidence,
    verify_fh_artifact_linkage,
)
from formal_toolchain.core.hashing import sha256_object


# ---------------------------------------------------------------------------
# 工厂 helper
# ---------------------------------------------------------------------------


def _make_candidate(*, upper=None, lower=None, active_upper=None):
    upper = dict(upper or {"tau1": 5, "tau7": 10})
    lower = dict(lower or {"tau1": 0, "tau7": 3})
    active = dict(active_upper or {"tau1": 5, "tau7": 10})
    return {
        "status": "PASS",
        "schema_version": "candidate_envelope_v1",
        "method": "finite_domain_enumeration",
        "lower": lower,
        "upper": upper,
        "active_release_budget_upper": active,
        "domain_certificate_hash": "a" * 64,
        "witnesses": [],
    }


def _make_common(*, active_release_budget_immutable=True, controller_budget_write=False):
    return {
        "status": "PASS",
        "schema_version": "common_transition_preservation_v1",
        "active_release_budget_immutable": active_release_budget_immutable,
        "controller_budget_write": controller_budget_write,
        "invariant_checked": True,
    }


def _make_deployed():
    return {
        "status": "PASS",
        "schema_version": "deployed_policy_preservation_v1",
        "leaf_count": 2,
        "selected_action_ids": [0, 1],
        "witnesses": [{"action_id": 0, "checked": True}],
        "implicit_noop_checked": True,
    }


def _make_certified_envelope(candidate, certified_certificate, *, upper=None, lower=None, active_upper=None):
    preservation = dict(certified_certificate)
    return {
        "status": "PASS",
        "schema_version": "certified_envelope_v1",
        "candidate_envelope_hash": sha256_object(candidate),
        "context_hash": "b" * 64,
        "preservation_certificate_hash": sha256_object(preservation),
        "preservation_certificate": preservation,
        "lower": dict(lower or candidate["lower"]),
        "upper": dict(upper or candidate["upper"]),
        "active_release_budget_upper": dict(active_upper or candidate["active_release_budget_upper"]),
    }


def _make_certified_certificate(candidate, common, deployed):
    return {
        "artifact_schema_version": "synthetic_phase_fh_certificate_v1",
        "obligation_id": "CERTIFIED_ENVELOPE",
        "obligation_status": "PASS",
        "certificate_context_hash": "c" * 64,
        "direct_predecessor_hashes": {},
        "checker_id": "test",
        "checker_version": "1",
        "inputs": {},
        "witness": {
            "candidate_hash": sha256_object(candidate),
            "common_hash": sha256_object(common),
            "deployed_hash": sha256_object(deployed),
        },
        "evidence": [{"fresh_process": True, "status": "PASS"}],
        "failure": None,
    }


def _make_reference_taskset(*, tasks=None):
    return {
        "schema_version": "reference_taskset_v1",
        "tasks": tasks or [
            {"name": "tau1", "criticality": "LO", "priority_index": 0,
             "period": 10, "deadline": 10, "c_lo": 2, "c_hi": 2, "code_c_lo": 2, "code_c_hi": 2},
            {"name": "tau7", "criticality": "HI", "priority_index": 1,
             "period": 20, "deadline": 20, "c_lo": 3, "c_hi": 5, "code_c_lo": 3, "code_c_hi": 5},
        ],
        "priority_order": ["tau1", "tau7"],
        "source_context_hash": "d" * 64,
    }


# ---------------------------------------------------------------------------
# 正常推导
# ---------------------------------------------------------------------------


def test_budget_invariants_are_derived():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    evidence = derive_budget_invariant_evidence(
        reference_taskset=reference,
        candidate=candidate,
        common=common,
        deployed=deployed,
        certified_envelope=cert_env,
        certified_certificate=cert_cert,
    )

    assert set(evidence) == {
        "LO_BUDGET_UPPER_INVARIANT",
        "HI_BUDGET_LOWER_INVARIANT",
        "ACTIVE_RELEASE_BUDGET_INVARIANT",
    }
    assert all(item["status"] == "PASS" for item in evidence.values())
    assert all(
        item["schema_version"] == "budget_invariant_derivation_v1"
        for item in evidence.values()
    )


# ---------------------------------------------------------------------------
# HI 任务名不以 HI 结尾
# ---------------------------------------------------------------------------


def test_hi_task_not_named_with_hi_suffix():
    """tau7 的 criticality 是 HI 但名称不以 HI 结尾，criticality 必须从
    reference taskset 读取而非从名称推断。"""
    reference = _make_reference_taskset(tasks=[
        {"name": "tau7", "criticality": "HI", "priority_index": 0,
         "period": 10, "deadline": 10, "c_lo": 3, "c_hi": 5, "code_c_lo": 3, "code_c_hi": 5},
        {"name": "tau1", "criticality": "LO", "priority_index": 1,
         "period": 20, "deadline": 20, "c_lo": 2, "c_hi": 2, "code_c_lo": 2, "code_c_hi": 2},
    ])
    candidate = _make_candidate(upper={"tau7": 10, "tau1": 5}, lower={"tau7": 3, "tau1": 0},
                                active_upper={"tau7": 10, "tau1": 5})
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert,
                                         upper={"tau7": 10, "tau1": 5},
                                         lower={"tau7": 3, "tau1": 0},
                                         active_upper={"tau7": 10, "tau1": 5})

    evidence = derive_budget_invariant_evidence(
        reference_taskset=reference, candidate=candidate, common=common,
        deployed=deployed, certified_envelope=cert_env,
        certified_certificate=cert_cert,
    )

    hi_rows = evidence["HI_BUDGET_LOWER_INVARIANT"]["rows"]
    lo_rows = evidence["LO_BUDGET_UPPER_INVARIANT"]["rows"]

    assert any(row["task"] == "tau7" for row in hi_rows)
    assert not any(row["task"] == "tau7" for row in lo_rows)
    assert any(row["task"] == "tau1" for row in lo_rows)
    assert not any(row["task"] == "tau1" for row in hi_rows)


# ---------------------------------------------------------------------------
# 缺失 common preservation
# ---------------------------------------------------------------------------


def test_missing_common_preservation_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    common = _make_common()
    common["status"] = "UNRESOLVED"
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    with pytest.raises(ValueError, match="COMMON_TRANSITION_PRESERVATION"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


# ---------------------------------------------------------------------------
# provenance hash 不匹配
# ---------------------------------------------------------------------------


def test_candidate_hash_mismatch_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    common = _make_common()
    deployed = _make_deployed()

    tampered_candidate = dict(candidate)
    tampered_candidate["upper"] = dict(candidate["upper"])
    tampered_candidate["upper"]["tau1"] = 999  # 修改但不更新 cert

    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    with pytest.raises(ValueError, match="CANDIDATE_ENVELOPE_HASH_NOT_BOUND"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=tampered_candidate,
            common=common, deployed=deployed,
            certified_envelope=cert_env, certified_certificate=cert_cert,
        )


def test_common_hash_mismatch_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    common = _make_common()
    deployed = _make_deployed()

    tampered_common = dict(common)
    tampered_common["active_release_budget_immutable"] = False

    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    with pytest.raises(ValueError, match="COMMON_PRESERVATION_HASH_NOT_BOUND"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate,
            common=tampered_common, deployed=deployed,
            certified_envelope=cert_env, certified_certificate=cert_cert,
        )


def test_deployed_hash_mismatch_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    common = _make_common()
    deployed = _make_deployed()

    tampered_deployed = dict(deployed)
    tampered_deployed["leaf_count"] = 99

    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    with pytest.raises(ValueError, match="DEPLOYED_PRESERVATION_HASH_NOT_BOUND"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate,
            common=common, deployed=tampered_deployed,
            certified_envelope=cert_env, certified_certificate=cert_cert,
        )


# ---------------------------------------------------------------------------
# task domain 不一致
# ---------------------------------------------------------------------------


def test_task_domain_extra_task_fails():
    reference = _make_reference_taskset(tasks=[
        {"name": "tau1", "criticality": "LO", "priority_index": 0,
         "period": 10, "deadline": 10, "c_lo": 2, "c_hi": 2, "code_c_lo": 2, "code_c_hi": 2},
        {"name": "tau7", "criticality": "HI", "priority_index": 1,
         "period": 20, "deadline": 20, "c_lo": 3, "c_hi": 5, "code_c_lo": 3, "code_c_hi": 5},
        {"name": "tau_extra", "criticality": "LO", "priority_index": 2,
         "period": 30, "deadline": 30, "c_lo": 1, "c_hi": 1, "code_c_lo": 1, "code_c_hi": 1},
    ])
    candidate = _make_candidate()
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    with pytest.raises(ValueError, match="TASK_DOMAIN_MISMATCH"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


def test_task_domain_missing_task_fails():
    reference = _make_reference_taskset(tasks=[
        {"name": "tau1", "criticality": "LO", "priority_index": 0,
         "period": 10, "deadline": 10, "c_lo": 2, "c_hi": 2, "code_c_lo": 2, "code_c_hi": 2},
    ])
    candidate = _make_candidate()
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    with pytest.raises(ValueError, match="TASK_DOMAIN_MISMATCH"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


# ---------------------------------------------------------------------------
# active release immutability 未证明
# ---------------------------------------------------------------------------


def test_active_release_immutability_not_proved_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    common = _make_common(active_release_budget_immutable=False)
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    with pytest.raises(ValueError, match="ACTIVE_RELEASE_IMMUTABILITY_NOT_PROVED"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


def test_controller_future_budget_write_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    common = _make_common(controller_budget_write=True)
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    with pytest.raises(ValueError, match="CONTROLLER_FUTURE_BUDGET_ONLY_NOT_PROVED"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


# ---------------------------------------------------------------------------
# certified upper/lower mismatch
# ---------------------------------------------------------------------------


def test_certified_upper_mismatch_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate(upper={"tau1": 5, "tau7": 10})
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert,
                                         upper={"tau1": 5, "tau7": 999})

    with pytest.raises(ValueError, match="CERTIFIED_UPPER_MISMATCH"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


def test_certified_lower_mismatch_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate(lower={"tau1": 0, "tau7": 3})
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert,
                                         lower={"tau1": 0, "tau7": 999})

    with pytest.raises(ValueError, match="CERTIFIED_LOWER_MISMATCH"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


def test_certified_active_upper_mismatch_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate(active_upper={"tau1": 5, "tau7": 10})
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert,
                                         active_upper={"tau1": 5, "tau7": 999})

    with pytest.raises(ValueError, match="CERTIFIED_ACTIVE_UPPER_MISMATCH"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def test_candidate_not_pass_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    candidate["status"] = "FAIL"
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    with pytest.raises(ValueError, match="CANDIDATE_ENVELOPE_NOT_PASS"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


def test_reference_taskset_missing_tasks_field():
    with pytest.raises(ValueError, match="REFERENCE_TASKSET_TASKS_REQUIRED"):
        _task_criticality({"schema_version": "v1"})


def test_reference_taskset_empty_tasks():
    with pytest.raises(ValueError, match="REFERENCE_TASKSET_TASKS_REQUIRED"):
        _task_criticality({"tasks": []})


def test_duplicate_task_names_fails():
    with pytest.raises(ValueError, match="DUPLICATE_TASK_NAME"):
        _task_criticality({"tasks": [
            {"name": "tau1", "criticality": "LO"},
            {"name": "tau1", "criticality": "HI"},
        ]})


def test_invalid_criticality_fails():
    with pytest.raises(ValueError, match="TASK_CRITICALITY_INVALID"):
        _task_criticality({"tasks": [
            {"name": "tau1", "criticality": "INVALID"},
        ]})


def test_missing_criticality_field_fails():
    with pytest.raises(ValueError, match="REFERENCE_TASKSET_TASKS_REQUIRED"):
        _task_criticality({"tasks": [
            {"name": "tau1"},
        ]})


def test_missing_name_field_fails():
    with pytest.raises(ValueError, match="REFERENCE_TASKSET_TASKS_REQUIRED"):
        _task_criticality({"tasks": [
            {"criticality": "LO"},
        ]})


# ---------------------------------------------------------------------------
# _certificate_passed 状态字段兼容
# ---------------------------------------------------------------------------


def test_certificate_passed_handles_various_status_fields():
    assert _certificate_passed({
        "artifact_schema_version": "certificate_envelope_v1",
        "obligation_status": "PASS",
    })
    assert not _certificate_passed({
        "artifact_schema_version": "certificate_envelope_v1",
        "obligation_status": "FAIL",
    })
    assert _certificate_passed({
        "schema_version": "candidate_envelope_v1",
        "status": "PASS",
    })
    assert _certificate_passed({
        "z3_proof_result": "PASS",
    })
    assert not _certificate_passed({
        "z3_proof_result": "FAIL",
    })
    assert _certificate_passed({
        "artifact_schema_version": "synthetic_phase_f_v1",
        "obligation_status": "PASS",
    })
    assert _certificate_passed({
        "artifact_schema_version": "synthetic_phase_fh_certificate_v1",
        "obligation_status": "PASS",
    })


# ---------------------------------------------------------------------------
# _require_pass
# ---------------------------------------------------------------------------


def test_require_pass_accepts_pass_and_rejects_fail():
    _require_pass({"status": "PASS", "schema_version": "candidate_envelope_v1"},
                  name="CANDIDATE_ENVELOPE")
    _require_pass({"status": "PASS", "schema_version": "candidate_envelope_v1"},
                  name="CANDIDATE_ENVELOPE", schema_version="candidate_envelope_v1")
    with pytest.raises(ValueError, match="CANDIDATE_ENVELOPE_NOT_PASS"):
        _require_pass({"status": "FAIL", "schema_version": "candidate_envelope_v1"},
                      name="CANDIDATE_ENVELOPE")
    with pytest.raises(ValueError, match="CANDIDATE_ENVELOPE_SCHEMA_MISMATCH"):
        _require_pass({"status": "PASS", "schema_version": "wrong_v1"},
                      name="CANDIDATE_ENVELOPE", schema_version="candidate_envelope_v1")


# ---------------------------------------------------------------------------
# verify_fh_artifact_linkage
# ---------------------------------------------------------------------------


def test_fresh_process_missing_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_cert["evidence"] = [{"status": "PASS"}]  # 缺少 fresh_process
    cert_env = _make_certified_envelope(candidate, cert_cert)

    with pytest.raises(ValueError, match="FH_ARTIFACT_PROVENANCE_MISMATCH"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


def test_certified_envelope_preservation_hash_mismatch_fails():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)

    tampered_cert_cert = dict(cert_cert)
    cert_env = _make_certified_envelope(candidate, cert_cert)
    cert_env["preservation_certificate_hash"] = "b" + "a" * 63  # 与 preservation 不匹配

    with pytest.raises(ValueError, match="FH_ARTIFACT_PROVENANCE_MISMATCH"):
        derive_budget_invariant_evidence(
            reference_taskset=reference, candidate=candidate, common=common,
            deployed=deployed, certified_envelope=cert_env,
            certified_certificate=cert_cert,
        )


# ---------------------------------------------------------------------------
# rows content verification
# ---------------------------------------------------------------------------


def test_lo_budget_upper_rows_contain_required_fields():
    reference = _make_reference_taskset()
    candidate = _make_candidate(upper={"tau1": 5, "tau7": 10}, lower={"tau1": 0, "tau7": 3},
                                active_upper={"tau1": 5, "tau7": 10})
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    evidence = derive_budget_invariant_evidence(
        reference_taskset=reference, candidate=candidate, common=common,
        deployed=deployed, certified_envelope=cert_env, certified_certificate=cert_cert,
    )

    lo_rows = evidence["LO_BUDGET_UPPER_INVARIANT"]["rows"]
    assert len(lo_rows) == 1
    assert lo_rows[0]["task"] == "tau1"
    assert lo_rows[0]["criticality"] == "LO"
    assert lo_rows[0]["upper"] == 5
    assert lo_rows[0]["common_transition_preserved"] is True
    assert lo_rows[0]["deployed_policy_preserved"] is True


def test_hi_budget_lower_rows_contain_required_fields():
    reference = _make_reference_taskset()
    candidate = _make_candidate(upper={"tau1": 5, "tau7": 10}, lower={"tau1": 0, "tau7": 3},
                                active_upper={"tau1": 5, "tau7": 10})
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    evidence = derive_budget_invariant_evidence(
        reference_taskset=reference, candidate=candidate, common=common,
        deployed=deployed, certified_envelope=cert_env, certified_certificate=cert_cert,
    )

    hi_rows = evidence["HI_BUDGET_LOWER_INVARIANT"]["rows"]
    assert len(hi_rows) == 1
    assert hi_rows[0]["task"] == "tau7"
    assert hi_rows[0]["criticality"] == "HI"
    assert hi_rows[0]["lower"] == 3
    assert hi_rows[0]["common_transition_preserved"] is True
    assert hi_rows[0]["deployed_policy_preserved"] is True


def test_active_release_budget_invariant_rows_cover_all_tasks():
    reference = _make_reference_taskset()
    candidate = _make_candidate(upper={"tau1": 5, "tau7": 10}, lower={"tau1": 0, "tau7": 3},
                                active_upper={"tau1": 5, "tau7": 10})
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    evidence = derive_budget_invariant_evidence(
        reference_taskset=reference, candidate=candidate, common=common,
        deployed=deployed, certified_envelope=cert_env, certified_certificate=cert_cert,
    )

    ar_rows = evidence["ACTIVE_RELEASE_BUDGET_INVARIANT"]["rows"]
    assert len(ar_rows) == 2
    task_names = {row["task"] for row in ar_rows}
    assert task_names == {"tau1", "tau7"}
    for row in ar_rows:
        assert row["release_snapshot_immutable"] is True
        assert row["active_release_upper"] == row["future_budget_upper"]
        assert row["criticality"] in ("LO", "HI")


def test_source_hashes_are_present():
    reference = _make_reference_taskset()
    candidate = _make_candidate()
    common = _make_common()
    deployed = _make_deployed()
    cert_cert = _make_certified_certificate(candidate, common, deployed)
    cert_env = _make_certified_envelope(candidate, cert_cert)

    evidence = derive_budget_invariant_evidence(
        reference_taskset=reference, candidate=candidate, common=common,
        deployed=deployed, certified_envelope=cert_env, certified_certificate=cert_cert,
    )

    for key in ("LO_BUDGET_UPPER_INVARIANT", "HI_BUDGET_LOWER_INVARIANT",
                "ACTIVE_RELEASE_BUDGET_INVARIANT"):
        hashes = evidence[key]["source_hashes"]
        assert hashes["candidate"] == sha256_object(candidate)
        assert hashes["common"] == sha256_object(common)
        assert hashes["deployed"] == sha256_object(deployed)
