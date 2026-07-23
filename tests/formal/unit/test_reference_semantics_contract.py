import pytest

from formal_toolchain.reference.semantics_contract import (
    _effective_config_values,
    build_reference_semantics_contract_certificate,
    evaluate_reference_semantics_contract,
)
from formal_toolchain.core.hashing import sha256_object


def _mock_context_hash() -> str:
    return "c" * 64


def _mock_certificate(
    *,
    obligation_id: str = "EFFECTIVE_RUNTIME_CONFIG",
    obligation_status: str = "PASS",
    witness: dict | None = None,
) -> dict:
    if witness is None:
        witness = {}
    artifact = {
        "artifact_schema_version": "certificate_envelope_v2",
        "obligation_id": obligation_id,
        "obligation_status": obligation_status,
        "certificate_context_hash": _mock_context_hash(),
        "direct_predecessor_hashes": {},
        "checker_id": "test",
        "checker_version": "v1",
        "inputs": {},
        "witness": witness,
        "evidence": [],
        "failure": None,
    }
    artifact["artifact_hash"] = sha256_object({k: artifact[k] for k in (
        "artifact_schema_version", "obligation_id", "obligation_status",
        "certificate_context_hash", "direct_predecessor_hashes", "checker_id",
        "checker_version", "inputs", "witness", "evidence", "failure",
    )})
    return artifact


def _passing_cert(*args, **kwargs):
    return _mock_certificate(*args, **kwargs)


def test_reference_semantics_contract_passes_finite_taskset():
    result = evaluate_reference_semantics_contract({
        "fingerprint": "a" * 64,
        "tasks": [
            {"name": "hi", "criticality": "HI", "priority_index": 0, "c_lo": 2, "c_hi": 5},
            {"name": "lo", "criticality": "LO", "priority_index": 1, "c_lo": 3, "c_hi": 1},
        ],
    })
    assert result["status"] == "PASS", result


def test_reference_semantics_contract_checks_unique_switch():
    result = evaluate_reference_semantics_contract({
        "tasks": [{"name": "hi", "criticality": "HI", "priority_index": 0, "c_lo": 1, "c_hi": 2}],
    })
    assert result["checks"]["unique_switch_trigger"] is True


def _run_build_with_config(config_witness):
    ctx_hash = _mock_context_hash()
    taskset = {
        "fingerprint": "a" * 64,
        "tasks": [
            {"name": "hi", "criticality": "HI", "priority_index": 0, "c_lo": 2, "c_hi": 5},
            {"name": "lo", "criticality": "LO", "priority_index": 1, "c_lo": 3, "c_hi": 1},
        ],
    }
    config_cert = _mock_certificate(witness=config_witness)
    taskset_cert = _passing_cert(
        obligation_id="REFERENCE_TASKSET",
        witness={"reference_taskset": {"fingerprint": "a" * 64}},
    )
    priority_cert = _passing_cert(obligation_id="STRICT_PRIORITY_ORDER")
    return build_reference_semantics_contract_certificate(
        reference_taskset=taskset,
        reference_taskset_certificate=taskset_cert,
        effective_runtime_config_certificate=config_cert,
        strict_priority_certificate=priority_cert,
        contexts={"semantic_context": {"hash": ctx_hash},
                   "reference_context": {"hash": ctx_hash}},
        context_hash=ctx_hash,
    )


def test_config_mismatch_semantics_wrong():
    result = _run_build_with_config({
        "config": {
            "fields": {
                "semantics": {"value": "C_AMC"},
                "c_amc_sem_primary_on_switch_time": {"value": True},
                "drop_lo_jobs_on_hi_switch": {"value": False},
            }
        }
    })
    assert result["obligation_status"] == "FAIL"
    assert result["failure"]["code"] == "REFERENCE_EFFECTIVE_CONFIG_MISMATCH"


def test_config_mismatch_primary_on_switch_time():
    result = _run_build_with_config({
        "config": {
            "fields": {
                "semantics": {"value": "C_AMC_SEM"},
                "c_amc_sem_primary_on_switch_time": {"value": False},
                "drop_lo_jobs_on_hi_switch": {"value": False},
            }
        }
    })
    assert result["obligation_status"] == "FAIL"
    assert result["failure"]["code"] == "REFERENCE_EFFECTIVE_CONFIG_MISMATCH"


def test_config_mismatch_drop_lo_jobs():
    result = _run_build_with_config({
        "config": {
            "fields": {
                "semantics": {"value": "C_AMC_SEM"},
                "c_amc_sem_primary_on_switch_time": {"value": True},
                "drop_lo_jobs_on_hi_switch": {"value": True},
            }
        }
    })
    assert result["obligation_status"] == "FAIL"
    assert result["failure"]["code"] == "REFERENCE_EFFECTIVE_CONFIG_MISMATCH"
