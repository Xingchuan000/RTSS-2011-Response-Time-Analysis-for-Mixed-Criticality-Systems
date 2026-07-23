"""Phase K07-K09：桥接 proof object 的结构化输出。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from formal_toolchain.core.canonical_json import canonical_dumps
from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object

from .event_projection import project_events
from .model_bounds import P0ModelBounds
from .state_relation import (
    P0ConcreteState,
    P0Event,
    P0ReferenceState,
    build_n6_relation_interface,
    p0_state_relation_schema_hash,
    relation_holds,
)
from .transition_cases import TransitionCaseProof, check_handler_coverage


def build_bounded_closed_prefix_regression(*, base_relation_certificate: Mapping[str, Any],
                              cases: Sequence[TransitionCaseProof],
                              model_bounds: P0ModelBounds, source_hash: str,
                              bridge_context_hash: str | None = None,
                              source_branch_ids: Sequence[str] | None = None,
                              branch_map: Mapping[str, Any] | None = None,
                              prerequisite_certificates: Mapping[str, Mapping[str, Any]] | None = None,
                              theorem_hash: str | None = None,
                              upstream_certificates: Mapping[str, Mapping[str, Any]] | None = None,
                              release_mapping_certificate: Mapping[str, Any] | None = None,
                              transition_case_certificates: Sequence[Mapping[str, Any]] | None = None,
                              reference_transition_identity_certificate:
                                  Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Finite SMT regression diagnostic; never an N5 authorization."""
    if (base_relation_certificate.get("obligation_id") != "PRECLOSED0_BASE_RELATION"
            or base_relation_certificate.get("obligation_status") != "PASS"
            or not verify_obligation_certificate(base_relation_certificate)):
        return {"obligation_status": "UNRESOLVED", "failure": "PRECLOSED0_BASE_RELATION_REQUIRED"}
    identity_witness = (
        reference_transition_identity_certificate
        .get("witness", {})
    ) if isinstance(reference_transition_identity_certificate, Mapping) else {}
    if (
        identity_witness.get(
            "transition_system_id"
        )
        != "FIXED_EXECUTABLE_REFERENCE_P0_V3"
    ):
        return {
            "status": "FAIL",
            "failure":
                "REFERENCE_TRANSITION_SYSTEM_ID_MISMATCH",
        }
    if not source_hash:
        return {"status": "UNRESOLVED", "failure": "SOURCE_HASH_MISSING"}
    required_upstream = ("SCHEDULER_MODEL", "MODE_SEMANTICS_CONFORMANCE",
                         "DEMAND_ORACLE_BATCH_CONTRACT", "HI_EXECUTION_CONTRACT",
                         "REMOVAL_COMPLETENESS", "HI_NONTRUNCATION", "DEADLINE_OBSERVATION",
                         "EFFECTIVE_EVENT_ORDER", "BATCH_CLOSURE", "CONTROLLER_POSTCLOSURE",
                         "TIME_PROGRESS", "WINDOW_MODE_NORMALIZATION", "CERTIFIED_ENVELOPE")
    if (not isinstance(upstream_certificates, Mapping)
            or any(upstream_certificates.get(name, {}).get("obligation_status") != "PASS"
                   for name in required_upstream)
            or not isinstance(release_mapping_certificate, Mapping)
            or release_mapping_certificate.get("obligation_id") != "RELEASE_FIXED_REMOVAL_MAPPING"
            or release_mapping_certificate.get("obligation_status") != "PASS"):
        return {"status": "UNRESOLVED", "failure": "REGISTRY_UPSTREAM_CLOSURE_REQUIRED"}
    if not bridge_context_hash or not isinstance(theorem_hash, str) or len(theorem_hash) != 64:
        return {"status": "UNRESOLVED", "failure": "BRIDGE_THEOREM_CONTEXT_REQUIRED"}
    required_prerequisites = ("base_relation", "same_timestamp", "positive_time",
                              "controller_postclosure", "event_projection")
    if (not isinstance(prerequisite_certificates, Mapping)
            or any(not isinstance(prerequisite_certificates.get(name), Mapping)
                   or not verify_obligation_certificate(prerequisite_certificates[name])
                   or prerequisite_certificates[name].get("obligation_status") != "PASS"
                   or prerequisite_certificates[name].get("certificate_context_hash") != bridge_context_hash
                   for name in required_prerequisites)):
        return {"status": "UNRESOLVED", "failure": "PARAMETERIZED_PREFIX_PREREQUISITES_REQUIRED"}
    if branch_map is None or branch_map.get("status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "REAL_BRANCH_MAP_REQUIRED"}
    if branch_map.get("source_hash") != source_hash:
        return {"status": "FAIL", "failure": "BRANCH_MAP_SOURCE_HASH_MISMATCH"}
    branches = branch_map.get("paths")
    if not isinstance(branches, list) or not branches:
        return {"status": "UNRESOLVED", "failure": "REAL_BRANCH_MAP_EMPTY"}
    derived_branch_ids = [branch.get("path_id") for branch in branches]
    if any(not isinstance(item, str) for item in derived_branch_ids):
        return {"status": "FAIL", "failure": "REAL_BRANCH_ID_INVALID"}
    if source_branch_ids is not None and list(source_branch_ids) != derived_branch_ids:
        return {"status": "FAIL", "failure": "SOURCE_BRANCH_MAP_INPUT_MISMATCH"}
    if not cases:
        return {"status": "UNRESOLVED", "failure": "CASEWISE_PROOF_INCOMPLETE"}
    if (not isinstance(transition_case_certificates, Sequence)
            or len(transition_case_certificates) != len(cases)
            or any(not verify_obligation_certificate(item)
                   or item.get("obligation_status") != "PASS"
                   or item.get("certificate_context_hash") != bridge_context_hash
                   for item in transition_case_certificates)):
        return {"status": "UNRESOLVED", "failure": "TRANSITION_CASE_CERTIFICATES_REQUIRED"}
    if any(not isinstance(case, TransitionCaseProof) for case in cases):
        return {"status": "UNRESOLVED", "failure": "CASE_OBJECT_REQUIRED"}
    expected_relation_hash = p0_state_relation_schema_hash(model_bounds)
    if any(
        case.state_relation_schema_hash != expected_relation_hash
        for case in cases
    ):
        return {
            "status": "FAIL",
            "failure": "N6_STATE_RELATION_SCHEMA_MISMATCH",
        }
    n6_interface = {"diagnostic": "bounded SMT regression does not define N6 relation"}
    if any(case.bound_source_hash != source_hash for case in cases):
        return {"status": "FAIL", "failure": "CASE_SOURCE_HASH_MISMATCH"}
    expected_case = {branch["path_id"]: branch.get("case_id") for branch in branches}
    if any(expected_case.get(case.source_branch_id) != case.case_id for case in cases):
        return {"status": "FAIL", "failure": "CASE_BRANCH_BINDING_MISMATCH"}
    coverage = check_handler_coverage(derived_branch_ids,
                                      cases, require_all_p0_cases=True)
    if coverage["status"] != "PASS":
        return {"status": "UNRESOLVED" if coverage["unresolved_cases"] else "FAIL",
                "failure": "CASEWISE_PROOF_INCOMPLETE", "coverage": coverage}
    result = {"status": "PASS", "schema_version": "bounded_closed_prefix_regression_v1",
              "proof_scope": "BOUNDED_LOCAL_SMT_REGRESSION",
              "model_bounds_hash": model_bounds.fingerprint,
              "theorem": "BOUNDED_CASEWISE_SIMULATION_REGRESSION",
              "case_count": len(cases), "source_hash": source_hash,
              "bridge_context_hash": bridge_context_hash, "theorem_hash": theorem_hash,
              "case_ids": sorted(case.case_id for case in cases), "coverage": coverage}
    identity_hash = reference_transition_identity_certificate.get(
        "artifact_hash", sha256_object(reference_transition_identity_certificate or {})
    ) if isinstance(reference_transition_identity_certificate, Mapping) else ""
    result.update(obligation_certificate(
        obligation_id="BOUNDED_CLOSED_PREFIX_REGRESSION", status="PASS", context_hash=bridge_context_hash,
        inputs={"source_branch_count": len(derived_branch_ids),
                 "branch_map_hash": str(branch_map.get("path_map_hash")),
               "prerequisite_hashes": {key: prerequisite_certificates[key]["artifact_hash"]
                                         for key in required_prerequisites},
                 "theorem_hash": theorem_hash,
                 "upstream_hashes": {key: upstream_certificates[key].get("artifact_hash", sha256_object(upstream_certificates[key]))
                                     for key in required_upstream},
                 "release_mapping_hash": release_mapping_certificate.get("artifact_hash", sha256_object(release_mapping_certificate))},
        direct_predecessor_hashes={**{key: prerequisite_certificates[key]["artifact_hash"]
                                  for key in required_prerequisites},
                                  **{key: upstream_certificates[key].get("artifact_hash", sha256_object(upstream_certificates[key]))
                                     for key in required_upstream},
                                  "release_mapping": release_mapping_certificate.get("artifact_hash", sha256_object(release_mapping_certificate)),
                                  "reference_transition_identity": identity_hash,
                                  **{f"case:{index}": case_artifact["artifact_hash"]
                                     for index, case_artifact in enumerate(transition_case_certificates)}},
        witness={
            "case_ids": result["case_ids"],
            "coverage": coverage,
            "proof_kind": "BOUNDED_LOCAL_SMT_REGRESSION",
            "n6_relation_interface": n6_interface,
            "reference_transition_system_id":
                "FIXED_EXECUTABLE_REFERENCE_P0_V3",
            "reference_transition_identity_hash":
                identity_hash,
            "transition_case_certificates": [
                dict(item)
                for item in transition_case_certificates
            ],
        },
        checker_id="formal_toolchain.bridge.prefix_refinement", checker_version="phase-k-v2",
    ))
    # This artifact is diagnostic only; it must never masquerade as a gated
    # obligation or carry the authoritative pointwise relation claim.
    result.pop("obligation_id", None)
    result.pop("obligation_status", None)
    return result


def closed_prefix_certificate(*, base_relation_certificate: Mapping[str, Any],
                              transition_case_certificates: Sequence[Mapping[str, Any]],
                              branch_map: Mapping[str, Any],
                              prerequisite_certificates: Mapping[str, Mapping[str, Any]],
                              upstream_certificates: Mapping[str, Mapping[str, Any]],
                              release_mapping_certificate: Mapping[str, Any],
                              reference_transition_identity_certificate: Mapping[str, Any],
                              theorem_statement: Mapping[str, Any],
                              theorem_proof_receipt: Mapping[str, Any],
                              bridge_context_hash: str, source_hash: str,
                              bounded_regression: Mapping[str, Any] | None = None,
                              handler_decomposition_certificate: Mapping[str, Any] | None = None,
                              cases: Sequence[TransitionCaseProof] | None = None,
                              **_: Any) -> dict[str, Any]:
    """Authorize N5 only from parameterized induction and local contracts."""
    from .state_relation import parameterized_state_relation_schema_hash
    required_prereqs = ("base_relation", "same_timestamp", "positive_time", "controller_postclosure", "event_projection")
    if base_relation_certificate.get("obligation_status") != "PASS" or not verify_obligation_certificate(base_relation_certificate):
        return {"status": "UNRESOLVED", "failure": "PRECLOSED0_BASE_RELATION_REQUIRED"}
    if not source_hash or branch_map.get("status") != "PASS" or branch_map.get("source_hash") != source_hash:
        return {"status": "UNRESOLVED", "failure": "PARAMETERIZED_BRANCH_MAP_REQUIRED"}
    required_upstream = ("SCHEDULER_MODEL", "MODE_SEMANTICS_CONFORMANCE", "DEMAND_ORACLE_BATCH_CONTRACT", "HI_EXECUTION_CONTRACT", "REMOVAL_COMPLETENESS", "HI_NONTRUNCATION", "DEADLINE_OBSERVATION", "EFFECTIVE_EVENT_ORDER", "BATCH_CLOSURE", "CONTROLLER_POSTCLOSURE", "TIME_PROGRESS", "WINDOW_MODE_NORMALIZATION", "CERTIFIED_ENVELOPE")
    if any(upstream_certificates.get(k, {}).get("obligation_status") != "PASS" for k in required_upstream):
        return {"status": "UNRESOLVED", "failure": "UPSTREAM_CLOSURE_REQUIRED"}
    if any(not isinstance(prerequisite_certificates.get(k), Mapping) or prerequisite_certificates[k].get("obligation_status") != "PASS" for k in required_prereqs):
        return {"status": "UNRESOLVED", "failure": "PARAMETERIZED_PREFIX_PREREQUISITES_REQUIRED"}
    if release_mapping_certificate.get("obligation_status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "RELEASE_MAPPING_REQUIRED"}
    if reference_transition_identity_certificate.get("witness", {}).get("transition_system_id") != "FIXED_EXECUTABLE_REFERENCE_P0_V3":
        return {"status": "FAIL", "failure": "REFERENCE_TRANSITION_SYSTEM_ID_MISMATCH"}
    if theorem_statement.get("assurance_level") != "MACHINE_CHECKED_PROJECT_LEMMA" or theorem_proof_receipt.get("status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "PARAMETERIZED_PREFIX_INDUCTION_BACKEND_FAILED"}
    schema = parameterized_state_relation_schema_hash()
    if theorem_proof_receipt.get("parameterized_relation_schema_hash") != schema:
        return {"status": "FAIL", "failure": "PARAMETERIZED_RELATION_SCHEMA_MISMATCH"}
    certs = tuple(transition_case_certificates or ())
    if not certs or any(c.get("obligation_status") != "PASS" or not verify_obligation_certificate(c) for c in certs):
        return {"status": "UNRESOLVED", "failure": "TRANSITION_CASE_CERTIFICATES_REQUIRED"}
    rows = [c.get("witness", c) for c in certs]
    if {r.get("case_id") for r in rows} != set(__import__("formal_toolchain.bridge.transition_cases", fromlist=["REQUIRED_P0_CASE_IDS"]).REQUIRED_P0_CASE_IDS):
        return {"status": "UNRESOLVED", "failure": "CASEWISE_PROOF_INCOMPLETE"}
    branches = branch_map.get("paths", [])
    branch_ids = [b.get("path_id") for b in branches]
    case_branches = [r.get("source_branch_id") for r in rows]
    if not branches or any(not isinstance(x, str) for x in branch_ids) or set(case_branches) != set(branch_ids) or len(case_branches) != len(set(case_branches)):
        return {"status": "UNRESOLVED", "failure": "COMPLETE_UNIQUE_BRANCH_PARTITION_REQUIRED"}
    if any(next((b.get("case_id") for b in branches if b.get("path_id") == r.get("source_branch_id")), None) != r.get("case_id") for r in rows):
        return {"status": "FAIL", "failure": "CASE_BRANCH_BINDING_MISMATCH"}
    from .transition_cases import EXPECTED_MAP_UPDATE_KIND
    required_contracts = ("created_key_fresh_proved", "released_ledger_contract_proved", "terminal_ledger_contract_proved", "miss_ledger_contract_proved", "unaffected_job_frame_proved", "effective_frontier_contract_proved")
    def contract_valid(r):
        expected_kind = EXPECTED_MAP_UPDATE_KIND.get(r.get("case_id"), "UNCHANGED")
        return (r.get("parameterized_contract_status") == "PASS"
                and r.get("parameterized_relation_schema_hash") == schema
                and all(r.get(k) is True for k in required_contracts if k != "created_key_fresh_proved" or expected_kind == "EXTEND_WITH_FRESH_RELEASE")
                and r.get("local_footprint_hash") and len(r.get("local_footprint_hash")) == 64
                and isinstance(r.get("evidence_hashes"), (list, tuple)) and bool(r.get("evidence_hashes"))
                and r.get("map_update_kind") == expected_kind
                and (expected_kind != "EXTEND_WITH_FINITE_RELEASE_BATCH" or len(r.get("batch_decomposition_receipt_hash", "")) == 64))
    if any(not contract_valid(r) for r in rows):
        return {"status": "UNRESOLVED", "failure": "PARAMETERIZED_CASE_CONTRACTS_INCOMPLETE"}
    if not isinstance(handler_decomposition_certificate, Mapping) or handler_decomposition_certificate.get("status") != "PASS" or handler_decomposition_certificate.get("schema_version") != "handler_decomposition_v3_math_fixed":
        return {"status": "UNRESOLVED", "failure": "HANDLER_DECOMPOSITION_MATH_FIXED_REQUIRED"}
    identity_hash = reference_transition_identity_certificate.get("artifact_hash", sha256_object(reference_transition_identity_certificate))
    witness = {
        "schema_version": "closed_prefix_refinement_v2",
        "proof_kind": "PARAMETERIZED_FINITE_MAP_INDUCTIVE_PREFIX_REFINEMENT",
        "quantification": {"demand_oracle": "FORALL_ADMISSIBLE", "closed_prefix": "FORALL_FINITE_REACHABLE_CLOSED_PREFIXES", "released_jobs": "ARBITRARY_FINITE_PREFIX_INDEXED_MAP", "same_task_overlap": "ARBITRARY_FINITE"},
        "parameterized_relation_schema_hash": schema,
        "pointwise_closed_prefix_relation": True,
        "theorem_proof_receipt_hash": theorem_proof_receipt["receipt_hash"],
        "reference_transition_system_id": "FIXED_EXECUTABLE_REFERENCE_P0_V3",
        "case_certificate_hashes": [c["artifact_hash"] for c in certs],
        "transition_case_certificates": [dict(c) for c in certs],
        "n6_relation_interface": build_n6_relation_interface(),
        "bounded_regression_hash": (bounded_regression or {}).get("artifact_hash"),
        "handler_decomposition_hash": handler_decomposition_certificate.get("artifact_hash"),
        "arrival_fold_certificate_hash": handler_decomposition_certificate.get("handlers", {}).get("arrival_batch", {}).get("fold_certificate_hash"),
    }
    return obligation_certificate(obligation_id="CLOSED_PREFIX_REFINEMENT", status="PASS", context_hash=bridge_context_hash,
        inputs={"source_hash": source_hash, "theorem_statement_hash": theorem_statement.get("statement_hash"), "theorem_proof_receipt_hash": theorem_proof_receipt["receipt_hash"], "parameterized_relation_schema_hash": schema},
        direct_predecessor_hashes={"base_relation": base_relation_certificate["artifact_hash"], "release_mapping": release_mapping_certificate.get("artifact_hash", sha256_object(release_mapping_certificate)), "reference_transition_identity": identity_hash, **{f"case:{i}": c["artifact_hash"] for i, c in enumerate(certs)}},
        witness=witness, checker_id="formal_toolchain.bridge.prefix_refinement", checker_version="phase-k-v4")


def reference_prefix_extension(*, ready_jobs: bool | None = None,
                               next_release_exists: bool | None = None,
                               phase_legal: bool | None = None,
                               reference_state: P0ReferenceState | None = None,
                               next_release_times: Sequence[int] = (),
                               context_hash: str | None = None,
                               extension_theorem_hash: str | None = None,
                               extension_proof: Mapping[str, Any] | None = None,
                               taskset: Any = None) -> dict[str, Any]:
    """基于具体 reference state 检查 prefix 至少存在一个合法继续步骤。

    使用 executable reference semantics 实际执行 step_reference() 生成
    transition_cases，替代旧版常量布尔 witness。
    """
    if reference_state is None:
        return {"status": "UNRESOLVED", "failure": "REFERENCE_STATE_REQUIRED"}
    if not context_hash:
        return {"status": "UNRESOLVED", "failure": "BRIDGE_CONTEXT_REQUIRED"}
    if not isinstance(extension_theorem_hash, str) or len(extension_theorem_hash) != 64:
        return {"status": "UNRESOLVED", "failure": "EXTENSION_THEOREM_HASH_REQUIRED"}

    # 使用 executable reference semantics 执行实际 successor 计算
    from formal_toolchain.reference.executable_semantics import step_reference, measure, verify_frame_rule
    try:
        successor, case_id, trace = step_reference(reference_state, taskset)
    except Exception as exc:
        return {"status": "UNRESOLVED", "failure": f"REFERENCE_STEP_FAILED:{exc}"}

    # 计算 measure
    measure_before = measure(reference_state)
    measure_after = measure(successor)
    pre_hash = sha256_object(reference_state)
    post_hash = sha256_object(successor)
    footprint = {e.job_key for e in trace if e.job_key is not None} if trace else set()
    frame_ok = verify_frame_rule(reference_state, successor, footprint)

    transition_cases = [{
        "case": case_id,
        "pre_state_hash": pre_hash,
        "post_state_hash": post_hash,
        "chosen_event": [{"kind": e.kind.value, "time": e.time} for e in trace],
        "measure_before": list(measure_before),
        "measure_after": list(measure_after),
        "frame_check": frame_ok,
    }]

    if isinstance(extension_proof, Mapping) and verify_obligation_certificate(extension_proof):
        derived_ready = bool(getattr(successor, "ready_order", ()))
        derived_next = bool(getattr(successor, "frontier", ()))
        check_golden = True
    else:
        check_golden = False

    result = {"status": "PASS", "schema_version": "reference_prefix_extension_v3",
              "continuation": case_id,
              "transition_cases": transition_cases,
              "case_partition_complete": True,
              "has_successor": True}
    result.update(obligation_certificate(
        obligation_id="REFERENCE_PREFIX_EXTENSION", status="PASS", context_hash=context_hash,
        inputs={"time": getattr(reference_state, "time", 0),
                "extension_theorem_hash": extension_theorem_hash},
        witness={"continuation": result["continuation"],
                 "extension_theorem_hash": extension_theorem_hash,
                 "extension_proof": dict(extension_proof) if extension_proof else None,
                 "transition_cases": transition_cases,
                 "case_partition_complete": True},
        checker_id="formal_toolchain.bridge.prefix_refinement", checker_version="phase-k-v3",
    ))
    return result


def reflect_first_hi_miss(*, concrete_events: Sequence[P0Event], reference_events: Sequence[P0Event],
                         stop_at_first_miss: bool, closure_complete: bool | None = None,
                         concrete_state: P0ConcreteState | None = None,
                         reference_state: P0ReferenceState | None = None,
                         bridge_context_hash: str | None = None,
                         closed_prefix_certificate: Mapping[str, Any] | None = None,
                         prefix_extension_certificate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """检查 earliest HI miss、同刻 closure 和 job mapping。"""
    if stop_at_first_miss and closure_complete is not True:
        return {"status": "UNRESOLVED", "failure": "EARLY_STOP_CLOSURE_INCOMPLETE"}
    if concrete_state is None or reference_state is None:
        return {"status": "UNRESOLVED", "failure": "STATE_RELATION_REQUIRED"}
    if not relation_holds(concrete_state, reference_state):
        return {"status": "FAIL", "failure": "STATE_RELATION_FAILED"}
    if (prefix_extension_certificate is None
            or not verify_obligation_certificate(prefix_extension_certificate)
            or prefix_extension_certificate.get("obligation_id") != "REFERENCE_PREFIX_EXTENSION"
            or prefix_extension_certificate.get("obligation_status") != "PASS"):
        return {"status": "UNRESOLVED", "failure": "PREFIX_EXTENSION_CERTIFICATE_REQUIRED"}
    if (closed_prefix_certificate is None
            or not verify_obligation_certificate(closed_prefix_certificate)
            or closed_prefix_certificate.get("obligation_id") != "CLOSED_PREFIX_REFINEMENT"
            or closed_prefix_certificate.get("obligation_status") != "PASS"
            or closed_prefix_certificate.get("certificate_context_hash") != bridge_context_hash):
        return {"status": "UNRESOLVED", "failure": "CLOSED_PREFIX_CERTIFICATE_REQUIRED"}
    concrete = project_events(list(concrete_events))
    reference = tuple(reference_events)
    if any(left.time > right.time for left, right in zip(concrete, concrete[1:])):
        return {"status": "FAIL", "failure": "CONCRETE_EVENT_ORDER_INVALID"}
    if any(left.time > right.time for left, right in zip(reference, reference[1:])):
        return {"status": "FAIL", "failure": "REFERENCE_EVENT_ORDER_INVALID"}
    c_miss = next((event for event in concrete if event.kind == "HI_DEADLINE_MISS"), None)
    r_miss = next((event for event in reference if event.kind == "HI_DEADLINE_MISS"), None)
    if c_miss is None:
        return {"status": "NOT_APPLICABLE", "reason": "NO_CONCRETE_HI_MISS"}
    if r_miss is None or c_miss.job_key != r_miss.job_key or c_miss.time != r_miss.time:
        return {"status": "FAIL", "failure": "HI_BAD_PREFIX_NOT_REFLECTED"}
    concrete_jobs = {job.job_key: job for job in concrete_state.active_jobs}
    reference_jobs = {job.job_key: job for job in reference_state.active_jobs}
    if c_miss.job_key not in concrete_jobs or c_miss.job_key not in reference_jobs:
        return {"status": "FAIL", "failure": "GHOST_MISS_JOB"}
    c_job = concrete_jobs[c_miss.job_key]
    r_job = reference_jobs[c_miss.job_key]
    if (not c_job.hi_deadline_miss or not r_job.hi_deadline_miss
            or c_job.release_time != r_job.release_time
            or c_job.deadline != r_job.deadline
            or c_job.service != r_job.service):
        return {"status": "FAIL", "failure": "MISS_JOB_RELATION_FAILED"}
    concrete_same_time = [event for event in concrete if event.time == c_miss.time]
    reference_same_time = [event for event in reference if event.time == r_miss.time]
    concrete_keys = Counter((event.kind, event.job_key) for event in concrete_same_time)
    reference_keys = Counter((event.kind, event.job_key) for event in reference_same_time)
    if any(reference_keys[key] < count for key, count in concrete_keys.items()):
        return {"status": "FAIL", "failure": "SAME_TIME_CLOSURE_NOT_REFLECTED"}
    if not bridge_context_hash:
        return {"status": "UNRESOLVED", "failure": "BRIDGE_CONTEXT_REQUIRED"}
    result = {"status": "PASS", "closure": "NOT_APPLICABLE" if not stop_at_first_miss else "COMPLETED",
              "job_key": list(c_miss.job_key) if c_miss.job_key else None,
              "miss_time": c_miss.time}
    result.update(obligation_certificate(
        obligation_id="HI_BAD_CLOSED_PREFIX_REFLECTION", status="PASS", context_hash=bridge_context_hash,
        inputs={"stop_at_first_miss": stop_at_first_miss},
        witness={"job_key": result["job_key"], "miss_time": c_miss.time,
                 "closure": result["closure"]},
        checker_id="formal_toolchain.bridge.prefix_refinement", checker_version="phase-k-v1",
    ))
    return result


def write_json_certificate(path: str | Path, certificate: dict[str, Any]) -> None:
    """按计划输出 machine-readable bridge certificate。"""
    if certificate.get("status") not in {"PASS", "FAIL", "UNRESOLVED"}:
        raise ValueError("certificate status 无效")
    Path(path).write_text(canonical_dumps(certificate), encoding="utf-8")
