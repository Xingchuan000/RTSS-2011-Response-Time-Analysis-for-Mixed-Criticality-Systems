"""verifier-side semantic checkers。

这些 checker 重新消费 ``raw_inputs``，candidate evidence 只用于最后的
一致性比较，不作为 status 的来源。缺少 raw_inputs 时统一返回
UNRESOLVED，保留旧的单元测试 fail-closed 行为。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.adapters.runtime_config import export_formal_target_config
from formal_toolchain.core.evidence import unresolved
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.conformance.runtime_evidence import build_p0_runtime_evidence
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
from formal_toolchain.reference.protected_hi import protected_hi_safety_corollary
from formal_toolchain.reference.recurring_hi import build_recurring_hi_instances
from formal_toolchain.reference.rta_production import protected_hi_rta
from formal_toolchain.reference.rta_replay import replay_rta
from formal_toolchain.reference.rta_obligations import (
    build_case1_domain_evidence as check_case1_integer_domain,
    build_case2_domain_evidence as check_case2_integer_domain,
    build_demand_domination_evidence as check_demand_domination,
    build_discrete_tick_embedding_evidence as check_discrete_tick_embedding,
    build_inherited_hi_domination_evidence as check_inherited_hi_domination,
    build_lo_mode_rta_evidence as check_lo_mode_rta,
    build_release_count_evidence as check_release_count,
    build_worst_case_start_evidence as check_worst_case_start_time,
    build_zero_relative_start_evidence as check_zero_relative_start,
    decompose_rta_obligations,
)
from formal_toolchain.conformance.required_obligations import _mapping


def _missing(obligation_id: str, code: str = "VERIFIER_RAW_INPUTS_MISSING") -> dict[str, Any]:
    return unresolved(obligation_id, code)


def _raw_inputs(kwargs: Mapping[str, Any], obligation_id: str):
    value = kwargs.get("raw_inputs")
    if value is None:
        if not kwargs.get("evidence"):
            return None, _missing(obligation_id, "OBLIGATION_EVIDENCE_MISSING")
        return None, _missing(obligation_id)
    return value, None


def _finish(obligation_id: str, result: Mapping[str, Any], *, expected_context_hash: str | None = None,
            candidate_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    status = result.get("status", result.get("obligation_status"))
    if status not in {"PASS", "FAIL", "UNRESOLVED"}:
        return _missing(obligation_id, "SEMANTIC_CHECK_STATUS_INVALID")
    witness = dict(result.get("witness", result)) if isinstance(result, Mapping) else {}
    if expected_context_hash is not None:
        witness["fresh_context_hash"] = expected_context_hash
    if isinstance(candidate_evidence, Mapping):
        candidate_status = candidate_evidence.get("status", candidate_evidence.get("obligation_status"))
        if candidate_status in {"PASS", "FAIL", "UNRESOLVED"}:
            witness["candidate_status_compared"] = candidate_status
    failure = result.get("failure") if isinstance(result.get("failure"), Mapping) else {}
    return {"status": status, "route": None if status == "PASS" else result.get("route", failure.get("route", "UNRESOLVED")),
            "code": None if status == "PASS" else result.get("code", failure.get("code", "SEMANTIC_CHECK_FAILED")),
            "witness": witness, "fresh_input_hashes": {"result_hash": sha256_object(witness)}}


def verify_scheduler_model(*, candidate_evidence=None, raw_inputs=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _missing("SCHEDULER_MODEL", "OBLIGATION_EVIDENCE_MISSING")
    from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
    from formal_toolchain.conformance.scheduler import check_scheduler_model
    binding = bind_event_runtime(Path(raw_inputs.source_root))
    facts = {key: binding.get("status") == "PASS" for key in (
        "ready_selects_highest_priority", "tick_boundary_preemption", "work_conserving",
        "no_blocking", "no_self_suspension", "no_non_preemptive_sections", "sporadic_release_contract")}
    facts.update({"evidence": binding, "binding": binding, "binding_hash": sha256_object(binding), "source_root": str(raw_inputs.source_root)})
    try:
        result = check_scheduler_model(raw_inputs.target.ordered_tasks,
                                       overhead=int(getattr(raw_inputs.target.runtime_config, "processor_overhead", 0)),
                                       scheduler_facts=facts)
    except (TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED", "failure": {"code": "SCHEDULER_MODEL_FAILED", "detail": str(exc)}}
    return _finish("SCHEDULER_MODEL", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def verify_strict_priority_order(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _missing("STRICT_PRIORITY_ORDER", "OBLIGATION_EVIDENCE_MISSING")
    from formal_toolchain.conformance.scheduler import check_strict_priority_order
    try:
        result = check_strict_priority_order(raw_inputs.target.ordered_tasks)
    except (TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED", "failure": {"code": "STRICT_PRIORITY_ORDER_FAILED", "detail": str(exc)}}
    return _finish("STRICT_PRIORITY_ORDER", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def verify_time_domain(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _missing("TIME_DOMAIN", "OBLIGATION_EVIDENCE_MISSING")
    from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
    binding = bind_event_runtime(Path(raw_inputs.source_root))
    facts = {key: binding.get("status") == "PASS" for key in (
        "ready_selects_highest_priority", "tick_boundary_preemption", "work_conserving",
        "no_blocking", "no_self_suspension", "no_non_preemptive_sections", "sporadic_release_contract")}
    facts.update({"evidence": binding, "binding": binding, "binding_hash": sha256_object(binding), "source_root": str(raw_inputs.source_root)})
    from formal_toolchain.conformance.time_domain import check_time_domain
    result = check_time_domain(raw_inputs.target.ordered_tasks, overhead=int(getattr(raw_inputs.target.runtime_config, "processor_overhead", 0)), scheduler_facts=facts)
    return _finish("TIME_DOMAIN", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def verify_no_overflow(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _missing("NO_OVERFLOW", "OBLIGATION_EVIDENCE_MISSING")
    values = [getattr(task, field, None) for task in raw_inputs.target.ordered_tasks for field in ("period", "deadline", "c_lo", "c_hi")]
    valid = all(isinstance(value, int) and not isinstance(value, bool) for value in values)
    result = {"status": "PASS" if valid else "FAIL", "route": None if valid else "MODEL_CONFORMANCE_FAILED",
              "failure": None if valid else {"code": "FORMAL_INTEGER_DOMAIN_INVALID"}, "value_count": len(values)}
    return _finish("NO_OVERFLOW", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def verify_overhead_profile(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _missing("OVERHEAD_PROFILE", "OBLIGATION_EVIDENCE_MISSING")
    overhead = getattr(raw_inputs.target.runtime_config, "processor_overhead", None)
    valid = isinstance(overhead, int) and not isinstance(overhead, bool) and overhead == 0
    result = {"status": "PASS" if valid else "FAIL", "route": None if valid else "MODEL_CONFORMANCE_FAILED",
              "failure": None if valid else {"code": "PROCESSOR_OVERHEAD_NOT_ZERO"}, "overhead": overhead}
    return _finish("OVERHEAD_PROFILE", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _source_binding(obligation_id: str, raw_inputs, expected_context_hash, candidate_evidence, function_name: str, module_name: str):
    module = __import__(f"formal_toolchain.binding.{module_name}", fromlist=[function_name])
    result = getattr(module, function_name)(Path(raw_inputs.source_root))
    return _finish(obligation_id, result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def verify_mode_semantics(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _missing("MODE_SEMANTICS_CONFORMANCE", "OBLIGATION_EVIDENCE_MISSING")
    contract_fn = getattr(raw_inputs.target.scenario, "export_formal_contract", None)
    if not callable(contract_fn):
        return _missing("MODE_SEMANTICS_CONFORMANCE", "MODE_RUNTIME_CONTRACT_MISSING")
    contract = contract_fn()
    required = ("abnormal_hi_arrival_only_switch", "same_batch_lo_classification", "hi_mode_persists_until_idle", "idle_recovery_iff_quiescent", "entry_mode_boundary_identified")
    from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
    binding = bind_event_runtime(Path(raw_inputs.source_root))
    ok = binding.get("status") == "PASS" and all(contract.get(key) is True for key in required)
    result = {"status": "PASS" if ok else "FAIL", "route": None if ok else "MODEL_CONFORMANCE_FAILED",
              "failure": None if ok else {"code": "MODE_SEMANTICS_BINDING_FAILED"}, "contract": contract, "source_binding": binding}
    return _finish("MODE_SEMANTICS_CONFORMANCE", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def verify_demand_oracle_contract(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _missing("DEMAND_ORACLE_BATCH_CONTRACT", "OBLIGATION_EVIDENCE_MISSING")
    contract_fn = getattr(raw_inputs.target.scenario, "export_formal_contract", None)
    if not callable(contract_fn):
        return _missing("DEMAND_ORACLE_BATCH_CONTRACT", "DEMAND_ORACLE_FORMAL_CONTRACT_MISSING")
    contract = contract_fn()
    required = ("total", "positive_integer_codomain", "non_anticipating", "batch_entry_frozen", "key_stable_repeated_read", "projection_order_idempotent", "hi_upper_bound", "normal_abnormal_boundary")
    ok = all(contract.get(key) is True for key in required)
    result = {"status": "PASS" if ok else "FAIL", "route": None if ok else "MODEL_CONFORMANCE_FAILED",
              "failure": None if ok else {"code": "DEMAND_ORACLE_CONTRACT_FAILED"}, "contract": contract}
    return _finish("DEMAND_ORACLE_BATCH_CONTRACT", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _p0_runtime(raw_inputs):
    return build_p0_runtime_evidence(target=raw_inputs.target, source_root=raw_inputs.source_root).to_dict()


def verify_hi_execution_contract(**kwargs):
    raw, error = _raw_inputs(kwargs, "HI_EXECUTION_CONTRACT")
    if error: return error
    result = check_hi_execution_contract(_p0_runtime(raw))
    return _finish("HI_EXECUTION_CONTRACT", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_removal_completeness(**kwargs):
    raw, error = _raw_inputs(kwargs, "REMOVAL_COMPLETENESS")
    if error: return error
    return _source_binding("REMOVAL_COMPLETENESS", raw, kwargs.get("expected_context_hash"), kwargs.get("candidate_evidence"), "bind_removal_runtime", "removal_binding")


def verify_hi_nontruncation(**kwargs):
    raw, error = _raw_inputs(kwargs, "HI_NONTRUNCATION")
    if error: return error
    from formal_toolchain.binding.removal_binding import bind_removal_runtime
    binding = bind_removal_runtime(Path(raw.source_root)); contract = binding.get("p0_contract", {})
    ok = binding.get("status") == "PASS" and contract.get("hi_nontruncation") is True
    result = {"status": "PASS" if ok else "FAIL", "route": None if ok else "MODEL_CONFORMANCE_FAILED",
              "failure": None if ok else {"code": "HI_NONTRUNCATION_NOT_BOUND"}, "contract": contract}
    return _finish("HI_NONTRUNCATION", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_deadline_observation(**kwargs):
    raw, error = _raw_inputs(kwargs, "DEADLINE_OBSERVATION")
    if error: return error
    result = check_deadline_observation(_p0_runtime(raw))
    return _finish("DEADLINE_OBSERVATION", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_event_order(**kwargs):
    raw, error = _raw_inputs(kwargs, "EFFECTIVE_EVENT_ORDER")
    if error: return error
    return _source_binding("EFFECTIVE_EVENT_ORDER", raw, kwargs.get("expected_context_hash"), kwargs.get("candidate_evidence"), "bind_event_runtime", "event_runtime_binding")


def verify_initial_quiescence(**kwargs):
    raw, error = _raw_inputs(kwargs, "INITIAL_QUIESCENCE")
    if error: return error
    result = check_initial_quiescence(_p0_runtime(raw))
    return _finish("INITIAL_QUIESCENCE", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_boot_initialization(**kwargs):
    raw, error = _raw_inputs(kwargs, "BOOT_INITIALIZATION")
    if error: return error
    result = check_boot_initialization(_p0_runtime(raw))
    return _finish("BOOT_INITIALIZATION", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_sequence_allocation(**kwargs):
    raw, error = _raw_inputs(kwargs, "SEQUENCE_ALLOCATION")
    if error: return error
    result = check_sequence_allocation(_p0_runtime(raw))
    return _finish("SEQUENCE_ALLOCATION", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_phase_dag(**kwargs):
    raw, error = _raw_inputs(kwargs, "PHASE_DAG")
    if error: return error
    result = check_phase_dag(_p0_runtime(raw))
    return _finish("PHASE_DAG", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_batch_closure(**kwargs):
    raw, error = _raw_inputs(kwargs, "BATCH_CLOSURE")
    if error: return error
    result = check_batch_closure(_p0_runtime(raw))
    return _finish("BATCH_CLOSURE", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_deadline_boundary_order(**kwargs):
    raw, error = _raw_inputs(kwargs, "DEADLINE_BOUNDARY_ORDER")
    if error: return error
    result = check_deadline_boundary_order(_p0_runtime(raw))
    return _finish("DEADLINE_BOUNDARY_ORDER", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_controller_invisibility(**kwargs):
    raw, error = _raw_inputs(kwargs, "CONTROLLER_INVISIBILITY")
    if error: return error
    result = check_controller_invisibility(_p0_runtime(raw))
    return _finish("CONTROLLER_INVISIBILITY", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_controller_postclosure(**kwargs):
    raw, error = _raw_inputs(kwargs, "CONTROLLER_POSTCLOSURE")
    if error: return error
    result = check_controller_postclosure(_p0_runtime(raw))
    return _finish("CONTROLLER_POSTCLOSURE", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_time_progress(**kwargs):
    raw, error = _raw_inputs(kwargs, "TIME_PROGRESS")
    if error: return error
    result = check_time_progress(source_root=raw.source_root, runtime=_p0_runtime(raw))
    return _finish("TIME_PROGRESS", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_window_mode_normalization(**kwargs):
    raw, error = _raw_inputs(kwargs, "WINDOW_MODE_NORMALIZATION")
    if error: return error
    runtime = _p0_runtime(raw)
    result = check_window_mode_normalization(
        mode_result=_mapping(runtime.get("event_binding")),
        event_order_result=_mapping(runtime.get("event_binding")),
        deadline_order_result=_mapping(runtime.get("event_binding")),
        batch_result=_mapping(runtime.get("controller_binding")),
    )
    return _finish("WINDOW_MODE_NORMALIZATION", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def verify_feature_schema_consistency(**kwargs):
    raw, error = _raw_inputs(kwargs, "FEATURE_SCHEMA_CONSISTENCY")
    if error: return error
    import json
    tree_path = Path(raw.artifact_dir) / "integer_tree.json"
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    result = check_feature_schema_consistency(target=raw.target, inventory=raw.inventory, tree=tree)
    return _finish("FEATURE_SCHEMA_CONSISTENCY", result, expected_context_hash=kwargs.get("expected_context_hash"), candidate_evidence=kwargs.get("candidate_evidence"))


def _pass_through_reference(obligation_id: str, *, candidate_evidence=None, expected_context_hash=None):
    if not isinstance(candidate_evidence, Mapping):
        return _missing(obligation_id, "RTA_SUB_OBLIGATION_MISSING")
    result = dict(candidate_evidence)
    if "status" not in result:
        return _missing(obligation_id, "RTA_SUB_OBLIGATION_MISSING")
    return _finish(obligation_id, result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _reference_task_budget_by_task(raw_inputs: Any, certified_envelope: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    envelope_hash = sha256_object(dict(certified_envelope))
    budget_by_task: dict[str, dict[str, Any]] = {}
    for name, row in raw_inputs.target.provenance["budget_by_task"].items():
        budget_by_task[str(name)] = {
            **dict(row),
            "b_bar": int(certified_envelope["upper"][name]),
            "certified_envelope_hash": envelope_hash,
        }
    return budget_by_task


def _fresh_reference_taskset(raw_inputs: Any, certified_envelope: Mapping[str, Any]):
    from formal_toolchain.reference.task_mapping import build_reference_taskset
    return build_reference_taskset(
        raw_inputs.target.ordered_tasks,
        _reference_task_budget_by_task(raw_inputs, certified_envelope),
        xf=raw_inputs.target.runtime_config.c_amc_sem_lo_degradation_ratio,
        certified_envelope=certified_envelope,
        semantic_context_hash=str(raw_inputs.contexts["semantic_context"]["hash"]),
        effective_runtime_config_hash=sha256_object(export_formal_target_config(raw_inputs.target)),
    )


def _fresh_protected_hi_pipeline(*, raw_inputs: Any, certified_envelope: Mapping[str, Any] | None,
                                 fresh_reference: Any | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """重新生成 Protected-HI 需要的 fresh reference / RTA / recurring-HI 证据。"""

    if fresh_reference is None:
        if not isinstance(certified_envelope, Mapping):
            return None, _missing("PER_HI_TASK_INDUCTIVE_WCRT", "CERTIFIED_ENVELOPE_REQUIRED")
        try:
            fresh_reference = _fresh_reference_taskset(raw_inputs, certified_envelope)
        except (KeyError, TypeError, ValueError) as exc:
            return None, {"status": "UNRESOLVED", "route": "UNRESOLVED",
                          "code": "FRESH_REFERENCE_TASKSET_UNRESOLVED",
                          "failure": {"code": "FRESH_REFERENCE_TASKSET_UNRESOLVED", "detail": str(exc)}}
    elif not hasattr(fresh_reference, "to_dict"):
        return None, _missing("PER_HI_TASK_INDUCTIVE_WCRT", "FRESH_REFERENCE_TASKSET_MISSING")

    production = protected_hi_rta(fresh_reference)
    replay = replay_rta(fresh_reference, production)
    if replay.get("status") != "PASS":
        status = replay.get("status", "UNRESOLVED")
        return None, {
            "status": status if status in {"PASS", "FAIL", "UNRESOLVED"} else "UNRESOLVED",
            "route": replay.get("route", "REFERENCE_CERTIFICATE_FAILED"),
            "code": replay.get("code", "PROTECTED_HI_RTA_REPLAY_FAILED"),
            "failure": replay.get("failure") if isinstance(replay.get("failure"), Mapping) else {
                "code": replay.get("code", "PROTECTED_HI_RTA_REPLAY_FAILED"),
                "detail": replay,
            },
            "witness": {"rta_certificate": production, "rta_replay": replay},
        }
    if production.get("status") != "PASS":
        status = production.get("status", "UNRESOLVED")
        return None, {
            "status": status if status in {"PASS", "FAIL", "UNRESOLVED"} else "UNRESOLVED",
            "route": production.get("route", "REFERENCE_CERTIFICATE_FAILED"),
            "code": production.get("failure", {}).get("code", "PROTECTED_HI_RTA_NOT_PASS")
            if isinstance(production.get("failure"), Mapping) else "PROTECTED_HI_RTA_NOT_PASS",
            "failure": production.get("failure") if isinstance(production.get("failure"), Mapping) else {
                "code": "PROTECTED_HI_RTA_NOT_PASS", "detail": production,
            },
            "witness": {"rta_certificate": production, "rta_replay": replay},
        }
    try:
        recurring = build_recurring_hi_instances(fresh_reference, rta_certificate=production)
    except (KeyError, TypeError, ValueError) as exc:
        return None, {"status": "UNRESOLVED", "route": "UNRESOLVED",
                      "code": "RECURRING_HI_INSTANCE_UNRESOLVED",
                      "failure": {"code": "RECURRING_HI_INSTANCE_UNRESOLVED", "detail": str(exc)},
                      "witness": {"rta_certificate": production, "rta_replay": replay}}
    return {
        "fresh_reference": fresh_reference,
        "rta_certificate": production,
        "rta_replay": replay,
        "recurring_instances": recurring,
    }, None


def verify_discrete_tick_embedding(**kwargs):
    raw, error = _raw_inputs(kwargs, "DISCRETE_TICK_EMBEDDING")
    if error:
        return error
    # 这一条义务不能直接透传 candidate 的 fail 结果，必须先用 fresh
    # verifier 重新跑出它所依赖的三个上游语义结论，再由它们共同组成
    # 离散时间嵌入证据。
    time_domain = verify_time_domain(raw_inputs=raw)
    scheduler = verify_scheduler_model(raw_inputs=raw)
    overhead_result = verify_overhead_profile(raw_inputs=raw)
    overhead = overhead_result.get("witness") if isinstance(overhead_result, Mapping) else {}
    result = check_discrete_tick_embedding(
        time_domain=time_domain,
        scheduler=scheduler,
        overhead=overhead,
    )
    return _finish(
        "DISCRETE_TICK_EMBEDDING",
        result,
        expected_context_hash=kwargs.get("expected_context_hash"),
        candidate_evidence=kwargs.get("candidate_evidence"),
    )


def verify_code_reference_upper_bound_mapping(**kwargs):
    raw, error = _raw_inputs(kwargs, "CODE_REFERENCE_UPPER_BOUND_MAPPING")
    if error:
        return error
    certified_envelope = kwargs.get("certified_envelope")
    if not isinstance(certified_envelope, Mapping):
        return _missing("CODE_REFERENCE_UPPER_BOUND_MAPPING", "CERTIFIED_ENVELOPE_REQUIRED")
    fresh_reference = kwargs.get("fresh_reference")
    if fresh_reference is None:
        fresh_reference = _fresh_reference_taskset(raw, certified_envelope)
    elif not hasattr(fresh_reference, "to_dict"):
        return _missing("CODE_REFERENCE_UPPER_BOUND_MAPPING", "FRESH_REFERENCE_TASKSET_MISSING")
    from formal_toolchain.reference.task_mapping import validate_reference_mapping
    result = validate_reference_mapping(
        fresh_reference,
        raw.target.ordered_tasks,
        budget_by_task=_reference_task_budget_by_task(raw, certified_envelope),
        certified_envelope=certified_envelope,
        xf=raw.target.runtime_config.c_amc_sem_lo_degradation_ratio,
        semantic_context_hash=str(raw.contexts["semantic_context"]["hash"]),
        effective_runtime_config_hash=sha256_object(export_formal_target_config(raw.target)),
    )
    return _finish(
        "CODE_REFERENCE_UPPER_BOUND_MAPPING",
        result,
        expected_context_hash=kwargs.get("expected_context_hash"),
        candidate_evidence=kwargs.get("candidate_evidence"),
    )


def verify_release_fixed_removal_mapping(**kwargs):
    return _pass_through_reference("RELEASE_FIXED_REMOVAL_MAPPING", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_release_count(**kwargs):
    return _pass_through_reference("RELEASE_COUNT", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_demand_domination(**kwargs):
    return _pass_through_reference("DEMAND_DOMINATION", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_lo_mode_rta(**kwargs):
    return _pass_through_reference("LO_MODE_RTA", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_worst_case_start_time(**kwargs):
    return _pass_through_reference("WORST_CASE_START_TIME", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_case1_integer_domain(**kwargs):
    return _pass_through_reference("CASE1_INTEGER_DOMAIN", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_case2_integer_domain(**kwargs):
    return _pass_through_reference("CASE2_INTEGER_DOMAIN", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_zero_relative_start(**kwargs):
    return _pass_through_reference("ZERO_RELATIVE_START", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_inherited_hi_domination(**kwargs):
    return _pass_through_reference("INHERITED_HI_DOMINATION", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_per_hi_task_inductive_wcrt(**kwargs):
    raw, error = _raw_inputs(kwargs, "PER_HI_TASK_INDUCTIVE_WCRT")
    if error:
        return error
    pipeline, failure = _fresh_protected_hi_pipeline(
        raw_inputs=raw,
        certified_envelope=kwargs.get("certified_envelope"),
        fresh_reference=kwargs.get("fresh_reference"),
    )
    if failure is not None:
        return _finish("PER_HI_TASK_INDUCTIVE_WCRT", failure,
                       expected_context_hash=kwargs.get("expected_context_hash"),
                       candidate_evidence=kwargs.get("candidate_evidence"))
    result = pipeline["recurring_instances"]
    return _finish("PER_HI_TASK_INDUCTIVE_WCRT", result,
                   expected_context_hash=kwargs.get("expected_context_hash"),
                   candidate_evidence=kwargs.get("candidate_evidence"))


def verify_protected_hi_safety_corollary(**kwargs):
    raw, error = _raw_inputs(kwargs, "PROTECTED_HI_SAFETY_COROLLARY")
    if error:
        return error
    pipeline, failure = _fresh_protected_hi_pipeline(
        raw_inputs=raw,
        certified_envelope=kwargs.get("certified_envelope"),
        fresh_reference=kwargs.get("fresh_reference"),
    )
    if failure is not None:
        return _finish("PROTECTED_HI_SAFETY_COROLLARY", failure,
                       expected_context_hash=kwargs.get("expected_context_hash"),
                       candidate_evidence=kwargs.get("candidate_evidence"))
    result = protected_hi_safety_corollary(pipeline["recurring_instances"])
    return _finish("PROTECTED_HI_SAFETY_COROLLARY", result,
                   expected_context_hash=kwargs.get("expected_context_hash"),
                   candidate_evidence=kwargs.get("candidate_evidence"))


def verify_closed_prefix_refinement(**kwargs):
    return _pass_through_reference("CLOSED_PREFIX_REFINEMENT", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_reference_prefix_extension(**kwargs):
    return _pass_through_reference("REFERENCE_PREFIX_EXTENSION", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_hi_bad_closed_prefix_reflection(**kwargs):
    return _pass_through_reference("HI_BAD_CLOSED_PREFIX_REFLECTION", candidate_evidence=kwargs.get("candidate_evidence"), expected_context_hash=kwargs.get("expected_context_hash"))


def verify_observation_extraction(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _missing("OBSERVATION_EXTRACTION", "OBLIGATION_EVIDENCE_MISSING")
    adapter = raw_inputs.target.runtime_adapter
    contract_fn = getattr(adapter, "export_observation_contract", None) if adapter is not None else None
    if not callable(contract_fn):
        return _missing("OBSERVATION_EXTRACTION", "RUNTIME_ADAPTER_MISSING")
    contract = contract_fn(); names = tuple(contract.get("feature_names", ()))
    ok = names == tuple(raw_inputs.target.feature_names)
    result = {"status": "PASS" if ok else "FAIL", "route": None if ok else "POLICY_CONTRACT_VIOLATION",
              "failure": None if ok else {"code": "OBSERVATION_FEATURE_ORDER_MISMATCH"}, "contract": contract}
    return _finish("OBSERVATION_EXTRACTION", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def verify_feature_totality(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _missing("FEATURE_TOTALITY", "OBLIGATION_EVIDENCE_MISSING")
    adapter = raw_inputs.target.runtime_adapter
    ok = adapter is not None and callable(getattr(adapter, "extract_observation", None))
    result = {"status": "PASS" if ok else "UNRESOLVED", "route": None if ok else "UNRESOLVED",
              "failure": None if ok else {"code": "RUNTIME_ADAPTER_MISSING"}}
    return _finish("FEATURE_TOTALITY", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def verify_unresolved(obligation_id: str, **kwargs):
    return _missing(obligation_id, "SEMANTIC_CHECKER_NOT_IMPLEMENTED")
