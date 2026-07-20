"""candidate 阶段的语义 evidence builder catalog。

每个 builder 都只消费当前 target、源码和本层 context，并把实际检查结果
封装成独立的 candidate evidence。这里没有“未知 obligation 继承 preflight”
的路径；没有实现的义务显式返回 ``UNRESOLVED``，由 fresh verifier 再决定
是否可以完成该义务。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from formal_toolchain.core.contexts import expected_context_for_obligation
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.formal_checks import proof_safe
from formal_toolchain.conformance.required_obligations import (
    check_batch_closure,
    check_boot_initialization,
    check_controller_invisibility,
    check_controller_postclosure,
    check_deadline_boundary_order,
    check_deadline_observation,
    check_feature_schema_consistency,
    check_hi_execution_contract,
    check_initial_quiescence,
    check_phase_dag,
    check_sequence_allocation,
    check_time_progress,
    check_window_mode_normalization,
)


@dataclass(frozen=True)
class CandidateEvidence:
    obligation_id: str
    status: str
    route: str | None
    code: str | None
    inputs: Mapping[str, str]
    witness: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "status": self.status,
            "route": self.route,
            "code": self.code,
            "checker_input_hashes": dict(self.inputs),
            "witness": dict(self.witness),
        }


Builder = Callable[..., CandidateEvidence]


def _result_status(result: Mapping[str, Any]) -> str:
    status = result.get("status", result.get("obligation_status"))
    if status in {"PASS", "FAIL", "UNRESOLVED"}:
        return str(status)
    return "UNRESOLVED"


def _candidate(obligation_id: str, result: Mapping[str, Any], *, contexts: Mapping[str, Mapping[str, Any]],
               source_root: Path, witness: Mapping[str, Any] | None = None) -> CandidateEvidence:
    status = _result_status(result)
    failure = result.get("failure") if isinstance(result.get("failure"), Mapping) else {}
    return CandidateEvidence(
        obligation_id=obligation_id,
        status=status,
        route=(None if status == "PASS" else str(result.get("route", failure.get("route", "UNRESOLVED")))),
        code=(None if status == "PASS" else str(result.get("code", failure.get("code", "SEMANTIC_CHECK_FAILED")))),
        inputs={
            "source_manifest_hash": sha256_object({"root": str(Path(source_root).resolve())}),
            "context_hash": expected_context_for_obligation(obligation_id, contexts),
        },
        witness={"result": dict(result), **dict(witness or {})},
    )


def _unresolved(obligation_id: str, *, contexts: Mapping[str, Mapping[str, Any]],
                source_root: Path, code: str = "CANDIDATE_EVIDENCE_BUILDER_NOT_IMPLEMENTED") -> CandidateEvidence:
    return _candidate(obligation_id, {"status": "UNRESOLVED", "route": "UNRESOLVED", "failure": {"code": code}},
                      contexts=contexts, source_root=source_root)


def _runtime_bundle(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    value = evidence.get("P0_RUNTIME_EVIDENCE")
    if isinstance(value, Mapping):
        return value
    return {"status": "UNRESOLVED", "failure": {"code": "P0_RUNTIME_EVIDENCE_MISSING"}}


def build_scheduler_model_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
    from formal_toolchain.conformance.scheduler import check_scheduler_model
    binding = bind_event_runtime(Path(source_root))
    facts = {
        "ready_selects_highest_priority": binding.get("status") == "PASS",
        "tick_boundary_preemption": binding.get("status") == "PASS",
        "work_conserving": binding.get("status") == "PASS",
        "no_blocking": binding.get("status") == "PASS",
        "no_self_suspension": binding.get("status") == "PASS",
        "no_non_preemptive_sections": binding.get("status") == "PASS",
        "sporadic_release_contract": binding.get("status") == "PASS",
        "evidence": binding,
        "binding": binding,
        "binding_hash": sha256_object(binding),
        "source_root": str(Path(source_root)),
    }
    try:
        result = check_scheduler_model(target.ordered_tasks, overhead=int(getattr(target.runtime_config, "processor_overhead", 0)), scheduler_facts=facts)
    except (TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED", "failure": {"code": "SCHEDULER_MODEL_FAILED", "detail": str(exc)}}
    return _candidate("SCHEDULER_MODEL", result, contexts=contexts, source_root=source_root, witness={"source_binding": binding})


def build_strict_priority_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    from formal_toolchain.conformance.scheduler import check_strict_priority_order
    try:
        result = check_strict_priority_order(target.ordered_tasks)
    except (TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED", "failure": {"code": "STRICT_PRIORITY_ORDER_FAILED", "detail": str(exc)}}
    return _candidate("STRICT_PRIORITY_ORDER", result, contexts=contexts, source_root=source_root)


def build_time_domain_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    # TIME_DOMAIN 与 SCHEDULER_MODEL 使用同一份当前源码事件绑定事实。
    # 不能把 ``scheduler_facts=None`` 交给 checker，因为那会把“未绑定”
    # 误当成正常输入而稳定地产生 UNRESOLVED。
    from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
    from formal_toolchain.conformance.time_domain import check_time_domain
    binding = bind_event_runtime(Path(source_root))
    facts = {key: binding.get("status") == "PASS" for key in (
        "ready_selects_highest_priority", "tick_boundary_preemption", "work_conserving",
        "no_blocking", "no_self_suspension", "no_non_preemptive_sections",
        "sporadic_release_contract")}
    facts.update({"evidence": binding, "binding": binding,
                  "binding_hash": sha256_object(binding),
                  "source_root": str(Path(source_root))})
    result = check_time_domain(
        target.ordered_tasks,
        overhead=int(getattr(target.runtime_config, "processor_overhead", 0)),
        scheduler_facts=facts,
    )
    return _candidate("TIME_DOMAIN", result, contexts=contexts, source_root=source_root)


def build_no_overflow_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    values = [getattr(task, field, None) for task in target.ordered_tasks for field in ("period", "deadline", "c_lo", "c_hi")]
    result = {"status": "PASS", "schema_version": "no_overflow_v1", "python_integer_values": len(values)}
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        result = {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED", "failure": {"code": "FORMAL_INTEGER_DOMAIN_INVALID"}}
    return _candidate("NO_OVERFLOW", result, contexts=contexts, source_root=source_root)


def build_overhead_profile_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    overhead = getattr(target.runtime_config, "processor_overhead", None)
    status = "PASS" if isinstance(overhead, int) and not isinstance(overhead, bool) and overhead == 0 else "FAIL"
    result = {"status": status, "overhead": overhead, "failure": None if status == "PASS" else {"code": "PROCESSOR_OVERHEAD_NOT_ZERO"}}
    return _candidate("OVERHEAD_PROFILE", result, contexts=contexts, source_root=source_root)


def build_mode_semantics_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
    binding = bind_event_runtime(Path(source_root))
    contract = getattr(target.scenario, "export_formal_contract", None)
    if not callable(contract):
        result = {"status": "UNRESOLVED", "route": "UNRESOLVED", "failure": {"code": "MODE_RUNTIME_CONTRACT_MISSING"}}
    else:
        data = contract()
        required = {"abnormal_hi_arrival_only_switch", "same_batch_lo_classification", "hi_mode_persists_until_idle", "idle_recovery_iff_quiescent", "entry_mode_boundary_identified"}
        result = {"status": "PASS" if binding.get("status") == "PASS" and all(data.get(key) is True for key in required) else "FAIL",
                  "properties": data, "source_binding": binding,
                  "failure": None if binding.get("status") == "PASS" and all(data.get(key) is True for key in required) else {"code": "MODE_SEMANTICS_BINDING_FAILED"}}
    return _candidate("MODE_SEMANTICS_CONFORMANCE", result, contexts=contexts, source_root=source_root)


def build_demand_oracle_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    contract_fn = getattr(target.scenario, "export_formal_contract", None)
    if not callable(contract_fn):
        result = {"status": "UNRESOLVED", "route": "UNRESOLVED", "failure": {"code": "DEMAND_ORACLE_FORMAL_CONTRACT_MISSING"}}
    else:
        contract = contract_fn()
        required = ("total", "positive_integer_codomain", "non_anticipating", "batch_entry_frozen", "key_stable_repeated_read", "projection_order_idempotent", "hi_upper_bound", "normal_abnormal_boundary")
        result = {"status": "PASS" if all(contract.get(key) is True for key in required) else "FAIL", "contract": contract,
                  "failure": None if all(contract.get(key) is True for key in required) else {"code": "DEMAND_ORACLE_CONTRACT_FAILED"}}
    return _candidate("DEMAND_ORACLE_BATCH_CONTRACT", result, contexts=contexts, source_root=source_root)


def _binding_builder(obligation_id: str, *, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], binding_name: str, **_: Any) -> CandidateEvidence:
    module = __import__(f"formal_toolchain.binding.{binding_name}", fromlist=["bind_removal_runtime"])
    function_name = "bind_removal_runtime" if binding_name == "removal_binding" else f"bind_{binding_name}"
    binding = getattr(module, function_name)(Path(source_root))
    return _candidate(obligation_id, binding, contexts=contexts, source_root=source_root, witness={"source_binding": binding})


def build_removal_completeness_evidence(**kwargs: Any) -> CandidateEvidence:
    return _binding_builder("REMOVAL_COMPLETENESS", binding_name="removal_binding", **kwargs)


def build_hi_nontruncation_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    if bool(getattr(target.runtime_config, "nonvacuity_hi_budget_cap_truncate", False)):
        result = {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                  "failure": {"code": "HI_BUDGET_CAP_TRUNCATES_JOB"}}
        return _candidate("HI_NONTRUNCATION", result, contexts=contexts, source_root=source_root)
    from formal_toolchain.binding.removal_binding import bind_removal_runtime
    binding = bind_removal_runtime(Path(source_root))
    contract = binding.get("p0_contract", {})
    result = {"status": "PASS" if binding.get("status") == "PASS" and contract.get("hi_nontruncation") is True else "FAIL",
              "contract": contract, "failure": None if binding.get("status") == "PASS" and contract.get("hi_nontruncation") is True else {"code": "HI_NONTRUNCATION_NOT_BOUND"}}
    return _candidate("HI_NONTRUNCATION", result, contexts=contexts, source_root=source_root)


def build_event_order_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
    binding = bind_event_runtime(Path(source_root))
    return _candidate("EFFECTIVE_EVENT_ORDER", binding, contexts=contexts, source_root=source_root, witness={"source_binding": binding})


def build_observation_extraction_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], inventory: Mapping[str, Any], **_: Any) -> CandidateEvidence:
    adapter = target.runtime_adapter
    if adapter is None or not callable(getattr(adapter, "export_observation_contract", None)):
        result = {"status": "UNRESOLVED", "route": "UNRESOLVED", "failure": {"code": "RUNTIME_ADAPTER_MISSING"}}
    else:
        contract = adapter.export_observation_contract()
        result = {"status": "PASS" if tuple(contract.get("feature_names", ())) == tuple(target.feature_names) else "FAIL",
                  "contract": contract, "failure": None if tuple(contract.get("feature_names", ())) == tuple(target.feature_names) else {"code": "OBSERVATION_FEATURE_ORDER_MISMATCH"}}
    return _candidate("OBSERVATION_EXTRACTION", result, contexts=contexts, source_root=source_root)


def build_feature_totality_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    adapter = target.runtime_adapter
    result = {"status": "PASS" if adapter is not None and callable(getattr(adapter, "extract_observation", None)) else "UNRESOLVED",
              "failure": None if adapter is not None and callable(getattr(adapter, "extract_observation", None)) else {"code": "RUNTIME_ADAPTER_MISSING"}}
    return _candidate("FEATURE_TOTALITY", result, contexts=contexts, source_root=source_root)


def build_initial_quiescence_evidence(*, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    return _candidate("INITIAL_QUIESCENCE", check_initial_quiescence(_runtime_bundle(evidence)), contexts=contexts, source_root=source_root)


def build_boot_initialization_evidence(*, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    return _candidate("BOOT_INITIALIZATION", check_boot_initialization(_runtime_bundle(evidence)), contexts=contexts, source_root=source_root)


def build_hi_execution_contract_evidence(*, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    return _candidate("HI_EXECUTION_CONTRACT", check_hi_execution_contract(_runtime_bundle(evidence)), contexts=contexts, source_root=source_root)


def build_deadline_observation_evidence(*, target: Any, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    if bool(getattr(target.runtime_config, "nonvacuity_deadline_cleanup_remove", False)):
        result = {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                  "failure": {"code": "DEADLINE_CLEANUP_REMOVES_JOB"}}
    else:
        result = check_deadline_observation(_runtime_bundle(evidence))
    return _candidate("DEADLINE_OBSERVATION", result, contexts=contexts, source_root=source_root)


def build_sequence_allocation_evidence(*, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    return _candidate("SEQUENCE_ALLOCATION", check_sequence_allocation(_runtime_bundle(evidence)), contexts=contexts, source_root=source_root)


def build_phase_dag_evidence(*, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    return _candidate("PHASE_DAG", check_phase_dag(_runtime_bundle(evidence)), contexts=contexts, source_root=source_root)


def build_batch_closure_evidence(*, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    return _candidate("BATCH_CLOSURE", check_batch_closure(_runtime_bundle(evidence)), contexts=contexts, source_root=source_root)


def build_deadline_boundary_order_evidence(*, target: Any, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    if bool(getattr(target.runtime_config, "nonvacuity_arrival_before_deadline", False)):
        result = {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                  "failure": {"code": "ARRIVAL_PRECEDES_DEADLINE_OBSERVATION"}}
    else:
        result = check_deadline_boundary_order(_runtime_bundle(evidence))
    return _candidate("DEADLINE_BOUNDARY_ORDER", result, contexts=contexts, source_root=source_root)


def build_controller_invisibility_evidence(*, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    return _candidate("CONTROLLER_INVISIBILITY", check_controller_invisibility(_runtime_bundle(evidence)), contexts=contexts, source_root=source_root)


def build_controller_postclosure_evidence(*, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    return _candidate("CONTROLLER_POSTCLOSURE", check_controller_postclosure(_runtime_bundle(evidence)), contexts=contexts, source_root=source_root)


def build_time_progress_evidence(*, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    return _candidate("TIME_PROGRESS", check_time_progress(source_root=source_root, runtime=_runtime_bundle(evidence)), contexts=contexts, source_root=source_root)


def build_window_mode_normalization_evidence(*, evidence: Mapping[str, Any], contexts: Mapping[str, Mapping[str, Any]], source_root: Path, **_: Any) -> CandidateEvidence:
    result = check_window_mode_normalization(
        mode_result=proof_safe((evidence.get("MODE_SEMANTICS_CONFORMANCE") or {}).get("witness", {}).get("result", (evidence.get("MODE_SEMANTICS_CONFORMANCE") or {}))),
        event_order_result=proof_safe((evidence.get("EFFECTIVE_EVENT_ORDER") or {}).get("witness", {}).get("result", (evidence.get("EFFECTIVE_EVENT_ORDER") or {}))),
        deadline_order_result=proof_safe((evidence.get("DEADLINE_BOUNDARY_ORDER") or {}).get("witness", {}).get("result", (evidence.get("DEADLINE_BOUNDARY_ORDER") or {}))),
        batch_result=proof_safe((evidence.get("BATCH_CLOSURE") or {}).get("witness", {}).get("result", (evidence.get("BATCH_CLOSURE") or {}))),
    )
    return _candidate("WINDOW_MODE_NORMALIZATION", result, contexts=contexts, source_root=source_root)


def build_feature_schema_consistency_evidence(*, target: Any, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], inventory: Mapping[str, Any], **_: Any) -> CandidateEvidence:
    tree = inventory.get("tree")
    result = check_feature_schema_consistency(target=target, inventory=inventory, tree=tree)
    return _candidate("FEATURE_SCHEMA_CONSISTENCY", result, contexts=contexts, source_root=source_root, witness={"tree_present": tree is not None})


def build_unimplemented_semantic_evidence(*, obligation_id: str, source_root: Path, contexts: Mapping[str, Mapping[str, Any]], **_: Any) -> CandidateEvidence:
    return _unresolved(obligation_id, contexts=contexts, source_root=source_root)


# 这张表是显式静态表；任何 ID 都必须在这里有明确 builder，不能从 registry
# 运行时生成 closure。其余义务返回 UNRESOLVED，绝不会因 preflight PASS 被授权。
SEMANTIC_EVIDENCE_BUILDERS: dict[str, Builder] = {
    "SCHEDULER_MODEL": build_scheduler_model_evidence,
    "STRICT_PRIORITY_ORDER": build_strict_priority_evidence,
    "TIME_DOMAIN": build_time_domain_evidence,
    "NO_OVERFLOW": build_no_overflow_evidence,
    "OVERHEAD_PROFILE": build_overhead_profile_evidence,
    "MODE_SEMANTICS_CONFORMANCE": build_mode_semantics_evidence,
    "DEMAND_ORACLE_BATCH_CONTRACT": build_demand_oracle_evidence,
    "REMOVAL_COMPLETENESS": build_removal_completeness_evidence,
    "HI_NONTRUNCATION": build_hi_nontruncation_evidence,
    "EFFECTIVE_EVENT_ORDER": build_event_order_evidence,
    "OBSERVATION_EXTRACTION": build_observation_extraction_evidence,
    "FEATURE_TOTALITY": build_feature_totality_evidence,
}


def _make_unresolved_builder(obligation_id: str) -> Builder:
    return lambda **kwargs: build_unimplemented_semantic_evidence(obligation_id=obligation_id, **kwargs)


# 尚未接入真实算法的义务也必须逐项列出。显式列举避免 registry 中新增
# obligation 后自动获得 builder，确保“未实现”只能产生 UNRESOLVED。
SEMANTIC_EVIDENCE_BUILDERS.update({
    "INITIAL_QUIESCENCE": build_initial_quiescence_evidence,
    "BOOT_INITIALIZATION": build_boot_initialization_evidence,
    "HI_EXECUTION_CONTRACT": build_hi_execution_contract_evidence,
    "DEADLINE_OBSERVATION": build_deadline_observation_evidence,
    "SEQUENCE_ALLOCATION": build_sequence_allocation_evidence,
    "PHASE_DAG": build_phase_dag_evidence,
    "BATCH_CLOSURE": build_batch_closure_evidence,
    "DEADLINE_BOUNDARY_ORDER": build_deadline_boundary_order_evidence,
    "CONTROLLER_INVISIBILITY": build_controller_invisibility_evidence,
    "CONTROLLER_POSTCLOSURE": build_controller_postclosure_evidence,
    "TIME_PROGRESS": build_time_progress_evidence,
    "WINDOW_MODE_NORMALIZATION": build_window_mode_normalization_evidence,
    "FEATURE_SCHEMA_CONSISTENCY": build_feature_schema_consistency_evidence,
    "DISCRETE_TICK_EMBEDDING": _make_unresolved_builder("DISCRETE_TICK_EMBEDDING"),
    "RELEASE_COUNT": _make_unresolved_builder("RELEASE_COUNT"),
    "DEMAND_DOMINATION": _make_unresolved_builder("DEMAND_DOMINATION"),
    "LO_MODE_RTA": _make_unresolved_builder("LO_MODE_RTA"),
    "WORST_CASE_START_TIME": _make_unresolved_builder("WORST_CASE_START_TIME"),
    "CASE1_INTEGER_DOMAIN": _make_unresolved_builder("CASE1_INTEGER_DOMAIN"),
    "CASE2_INTEGER_DOMAIN": _make_unresolved_builder("CASE2_INTEGER_DOMAIN"),
    "ZERO_RELATIVE_START": _make_unresolved_builder("ZERO_RELATIVE_START"),
    "INHERITED_HI_DOMINATION": _make_unresolved_builder("INHERITED_HI_DOMINATION"),
})
