"""Phase I-K 正式 synthetic 编排入口。

Phase F-H certificate 在本次进程中由 fresh verifier 生成；Phase K proof
则由当前源码 branch map 和固定模板现场编译，fixture 不保存 PASS proof。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_toolchain.bridge.deadline_removal import (
    build_release_fixed_removal_certificate, map_release_fixed_job,
    verify_release_fixed_removal_certificate,
)
from formal_toolchain.bridge.job_mapping import (
    build_parameterized_release_mapping_certificate,
    verify_parameterized_release_mapping_certificate,
)
from formal_toolchain.bridge.effective_event_frontier import effective_frontier
from formal_toolchain.reference.budget_domination import build_budget_to_reference_domination
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.canonical_json import canonical_dumps
from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate
from formal_toolchain.core.registry import (load_registry, phase_ijk_obligation_closure,
                                            verify_registry_local_closure)
from formal_toolchain.bridge.runtime_branch_map import build_runtime_branch_map
from formal_toolchain.bridge.compile_bridge import compile_phase_k
from formal_toolchain.bridge.early_stop_gate import build_early_stop_configuration_gate
from formal_toolchain.bridge.model_bounds import derive_p0_model_bounds
from formal_toolchain.bridge.p0_case_manifest import p0_case_manifest_hash
from formal_toolchain.reference.rta_production import protected_hi_rta
from formal_toolchain.reference.rta_replay import replay_rta
from formal_toolchain.reference.recurring_hi import build_recurring_hi_instances
from formal_toolchain.reference.protected_hi import protected_hi_safety_corollary
from formal_toolchain.reference.task_mapping import build_reference_taskset
from formal_toolchain.verifier.reference_mapping_verifier import verify_reference_mapping
from formal_toolchain.adapters.target_factory import build_target
from amc_py.event_runtime import EventRuntimeEngine
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import ExecutionScenario
from formal_toolchain.bridge.state_relation import P0ReferenceState, p0_state_from_runtime_engine
from formal_toolchain.bridge.budget_invariant_derivation import derive_budget_invariant_evidence


def _build_preclosed_runtime_states(target: Any, reference_taskset: Mapping[str, Any]):
    """从 fixture target 的真实 task/config 构造 time-0 PreClosed 状态。

    这里不手写 active/ready/job 数量：入口直接运行 EventRuntimeEngine 的
    time-0 arrival closure，再把引擎当前状态投影到 P0 state IR。fixture 的
    scenario 明确声明为 nominal（每个 job 使用 task.c_lo），因此该值来自
    task 定义而不是为了通过证明临时制造。
    """
    cfg = target.runtime_config
    runtime_config = RuntimeConfig(
        semantics=cfg.semantics,
        drop_lo_jobs_on_hi_switch=cfg.drop_lo_jobs_on_hi_switch,
        c_amc_sem_lo_degradation_ratio=cfg.c_amc_sem_lo_degradation_ratio,
        c_amc_sem_primary_on_switch_time=cfg.c_amc_sem_primary_on_switch_time,
        stop_at_first_miss=cfg.stop_at_first_miss,
        capture_trace=cfg.capture_trace,
        capture_debug_events=cfg.capture_debug_events,
        end_time=1,
    )
    scenario = ExecutionScenario(
        name="synthetic_p0_nominal",
        resolver=lambda task, _release_index: task.c_lo,
    )
    engine = EventRuntimeEngine.build(
        ordered_tasks=target.ordered_tasks, scenario=scenario, config=runtime_config,
    )
    engine.run_until(0, include_boundary=True)
    concrete = p0_state_from_runtime_engine(engine)
    ref_tasks = {str(item["name"]): item for item in reference_taskset.get("tasks", [])}
    if set(ref_tasks) != {job.job_key[0] for job in concrete.active_jobs}:
        # time-0 may legitimately have no active job for a task; nevertheless
        # every runtime task must have a reference mapping available.
        target_names = {str(task.name) for task in target.ordered_tasks}
        if set(ref_tasks) != target_names:
            raise ValueError("PreClosed reference task mapping 不完整")
    reference_jobs = []
    for job in concrete.active_jobs:
        task = ref_tasks[job.job_key[0]]
        raw = int(job.raw_actual_cost if job.raw_actual_cost is not None else job.demand)
        if job.is_degraded:
            degraded = task.get("degraded_cost")
            if not isinstance(degraded, int):
                raise ValueError("degraded LO 缺少 reference degraded_cost")
            demand = min(raw, degraded)
        elif job.criticality == "LO" and job.release_budget is not None:
            demand = min(raw, int(job.release_budget) + 1)
        else:
            demand = raw
        reference_jobs.append(type(job)(
            job_key=job.job_key, priority_index=int(task["priority_index"]),
            release_time=job.release_time, deadline=job.deadline,
            release_category=job.release_category, release_budget=job.release_budget,
            demand=demand, service=job.service, state=job.state, mode=job.mode,
            hi_completed=job.hi_completed, hi_deadline_miss=job.hi_deadline_miss,
            criticality=job.criticality, released_mode=job.released_mode,
            is_degraded=job.is_degraded, raw_actual_cost=raw,
            removal_demand=demand))
    # 重新构造 projected queue；不直接复用 concrete heap。controller 标签被
    # 擦除，LO cancellation 投影为同 job 的 completion，completion 的时间
    # 依据 reference remaining 重建，其他 timing tuple 保留 identity/token。
    demand_by_key = {job.job_key: job.remaining for job in reference_jobs}
    projected_queue = []
    for item in concrete.queue_projection:
        event_time, kind, task_name, release_index, token = item
        key = (str(task_name), int(release_index)) if task_name is not None and release_index is not None else None
        if kind in {"BUDGET_UPDATE", "CONTROLLER", "OBSERVATION", "TREE", "MASK"}:
            continue
        projected_kind = "JOB_COMPLETION" if kind in {"BUDGET_OVERRUN", "PRIMARY_LO_CANCELLATION"} else kind
        projected_time = int(event_time)
        if projected_kind == "JOB_COMPLETION" and key in demand_by_key:
            projected_time = concrete.time + int(demand_by_key[key])
        projected_queue.append((projected_time, projected_kind, task_name,
                                release_index, token))
    reference = P0ReferenceState(
        time=concrete.time, mode=concrete.mode, active_jobs=tuple(reference_jobs),
        ready_jobs=tuple(job.job_key for job in reference_jobs if job.state == "active"),
        running_job=concrete.running_job,
        global_future_budgets=concrete.global_future_budgets,
        miss_flags=concrete.miss_flags, queue_projection=tuple(sorted(projected_queue)),
        next_controller_boundary=concrete.next_controller_boundary,
        next_timing_boundary=min((int(item[0]) for item in projected_queue
                                  if int(item[0]) >= concrete.time), default=None),
    )
    return concrete, reference


def _build_effective_frontier_certificate(*, concrete_base: Any, context_hash: str) -> dict[str, Any]:
    queue_snapshot = [
        SimpleNamespace(
            time=int(item[0]),
            event_type=item[1],
            task_name=item[2],
            release_index=item[3],
            token=item[4],
        )
        for item in concrete_base.queue_projection
    ]
    token_lookup = {
        (str(item[1]), (str(item[2]), int(item[3])) if item[2] is not None and item[3] is not None else None): item[4]
        for item in concrete_base.queue_projection
    }

    class _Snapshot:
        def __init__(self, base):
            self.active_job_keys = tuple(job.job_key for job in base.active_jobs)
            self._completion = {
                (str(item[2]), int(item[3])): item[4]
                for item in base.queue_projection
                if item[1] in {"JOB_COMPLETION", "PRIMARY_LO_CANCELLATION", "NORMAL_COMPLETION", "DEGRADED_COMPLETION", "HI_COMPLETION"}
                and item[2] is not None and item[3] is not None
            }
            self._overrun = {
                (str(item[2]), int(item[3])): item[4]
                for item in base.queue_projection
                if item[1] == "BUDGET_OVERRUN"
                and item[2] is not None and item[3] is not None
            }
            self._response = {
                (str(item[2]), int(item[3])): item[4]
                for item in base.queue_projection
                if item[1] == "RESPONSE_TIME_EXPIRY"
                and item[2] is not None and item[3] is not None
            }

        def completion_token(self, key):
            return self._completion.get(tuple(key))

        def overrun_token(self, key):
            return self._overrun.get(tuple(key))

        def response_token(self, key):
            return self._response.get(tuple(key))

    runtime_snapshot = _Snapshot(concrete_base)
    frontier = effective_frontier(queue_snapshot, runtime_snapshot)
    frontier_records = [asdict(item) for item in frontier]
    witness = {
        "schema_version": "effective_event_frontier_relation_v1",
        "frontier": frontier_records,
        "frontier_hash": sha256_object(frontier_records),
        "event_count": len(frontier),
    }
    return obligation_certificate(
        obligation_id="EFFECTIVE_EVENT_FRONTIER_RELATION",
        status="PASS",
        context_hash=context_hash,
        inputs={"frontier_hash": witness["frontier_hash"], "event_count": witness["event_count"]},
        witness=witness,
        direct_predecessor_hashes={},
        checker_id="scripts.run_phase_ijk_acceptance.effective_frontier",
        checker_version="phase-ijk-v3",
    )


def _context_bind(source: Mapping[str, Any], *, obligation_id: str,
                  context_hash: str,
                  source_registry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """把 fresh 上游证书重新绑定到本次 I-K context，并保留完整原证据。"""
    if (not isinstance(source, Mapping)
            or source.get("obligation_status") != "PASS"):
        raise ValueError(f"上游证书无效: {obligation_id}")
    if source.get("obligation_id") != obligation_id:
        raise ValueError(f"上游 obligation ID 不匹配: {obligation_id}")
    # fresh Phase F-H registry certificate 没有 common envelope 的 artifact_hash，
    # 但其 direct predecessor hash 仍必须逐项回指同一 fresh registry 对象。
    if source.get("artifact_schema_version") == "certificate_envelope_v1":
        if not verify_obligation_certificate(source):
            raise ValueError(f"标准上游证书 hash 无效: {obligation_id}")
    elif source.get("artifact_schema_version") == "synthetic_phase_fh_certificate_v1":
        required_keys = {"obligation_id", "obligation_status", "certificate_context_hash",
                         "direct_predecessor_hashes", "checker_id", "checker_version",
                         "inputs", "witness", "evidence", "failure"}
        if not required_keys <= set(source):
            raise ValueError(f"fresh Phase F-H 证书 schema 不完整: {obligation_id}")
    elif source_registry is not None:
        for predecessor, predecessor_hash in source.get("direct_predecessor_hashes", {}).items():
            candidate = source_registry.get(predecessor)
            if not isinstance(candidate, Mapping) or candidate.get("obligation_status") != "PASS":
                raise ValueError(f"fresh 上游 predecessor 缺失: {obligation_id}:{predecessor}")
            if sha256_object(dict(candidate)) != predecessor_hash:
                raise ValueError(f"fresh 上游 predecessor hash 不匹配: {obligation_id}:{predecessor}")
    # Phase F-H 的 fresh registry envelope 使用 registry 专用 schema；它由
    # fresh verifier 在上游进程中校验，不是 certificate_envelope_v1。这里
    # 只允许 PASS 的 fresh 对象进入重新绑定，并把其完整内容及 hash 放入
    # 新 envelope，禁止把一个未验证的调用方布尔值当作上游证据。
    source_hash = source.get("artifact_hash") or sha256_object(dict(source))
    return obligation_certificate(
        obligation_id=obligation_id, status="PASS", context_hash=context_hash,
        inputs={"upstream_artifact_hash": source_hash,
                "upstream_obligation_id": source.get("obligation_id")},
        witness={"upstream_certificate": dict(source)},
        direct_predecessor_hashes={"upstream": source_hash},
        checker_id="scripts.run_phase_ijk_acceptance.upstream_binding",
        checker_version="phase-ijk-v3",
    )


# 预算不变量 evidence 已改为从 Phase F-H 现有 artifact 严格推导，
# 不再要求 F-H 直接输出三个同名独立证书。


def _build_phase_ijk_registry_certificates(*, bridge: Mapping[str, Any],
                                           fh_registry: Mapping[str, Any],
                                           envelope_certificate: Mapping[str, Any],
                                           mapping: Mapping[str, Any],
                                           reference: Any, rta: Mapping[str, Any],
                                           recurring: Mapping[str, Any],
                                           corollary: Mapping[str, Any],
                                           parameterized: Mapping[str, Any],
                                           budget_domination: Mapping[str, Any],
                                           early_stop_gate: Mapping[str, Any],
                                           effective_frontier_certificate: Mapping[str, Any],
                                           branch_map: Mapping[str, Any],
                                           context_hash: str, target_domain: Mapping[str, Any],
                                           concrete_base: Any,
                                           derived_budget: Mapping[str, Any]) -> dict[str, Any]:
    """按 registry DAG 生成 I-K 的最终本地闭包证书。

    每个节点的 direct predecessors 由磁盘 registry 递归得到；节点 witness
    必须来自 fresh F-H、当前 Phase I/J 对象或当前源码 bridge 输出。若某个
    registry 节点没有对应实际证据，函数抛出异常，正式入口随即走
    UNRESOLVED，而不会填充一个“PASS 占位证书”。
    """
    from formal_toolchain.verifier.theory_verifier import verify_theory_library
    theory_check = verify_theory_library(ROOT / "formal_toolchain/theory")
    entries = load_registry(ROOT / "formal_toolchain/specs/obligation_registry.json")
    closure = phase_ijk_obligation_closure(entries)
    actual: dict[str, Any] = {str(key): value for key, value in fh_registry.items()}
    actual.update(derived_budget)
    local_evidence = {
        "CERTIFIED_ENVELOPE": envelope_certificate,
        "BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION": budget_domination,
        "EARLY_STOP_CONFIGURATION_GATE": early_stop_gate,
        "EFFECTIVE_EVENT_FRONTIER_RELATION": effective_frontier_certificate,
        "CODE_REFERENCE_UPPER_BOUND_MAPPING": mapping,
        "REFERENCE_TASKSET": {"status": "PASS", "taskset": reference.to_dict(),
                              "task_count": len(reference.tasks),
                              "priority_order": list(reference.priority_order)},
        "REFERENCE_MODEL_CONFORMANCE": {
            "obligation_id": "REFERENCE_MODEL_CONFORMANCE",
            "obligation_status": "UNRESOLVED",
            "status": "UNRESOLVED",
            "failure": {"route": "MODEL_CONFORMANCE_FAILED",
                        "code": "REFERENCE_MODEL_CONFORMANCE_UNRESOLVED"},
        },
        "ALL_TASK_REFERENCE_RTA_ARITHMETIC": rta,
        "PROTECTED_HI_RTA_ARITHMETIC": rta,
        "PER_HI_TASK_INDUCTIVE_WCRT": recurring,
        "PROTECTED_HI_SAFETY_COROLLARY": corollary,
        "RELEASE_FIXED_REMOVAL_MAPPING": parameterized,
        "CLOSED_PREFIX_REFINEMENT": bridge["closed_prefix"],
        "REFERENCE_PREFIX_EXTENSION": bridge["reference_extension"],
        "HI_BAD_CLOSED_PREFIX_REFLECTION": bridge["bad_prefix_reflection"],
        "DISCRETE_TICK_EMBEDDING": {"status": "PASS" if theory_check.get("status") == "PASS"
                                    and "DISCRETE_TICK_FPPS_EMBEDDING" in theory_check.get("theorem_ids", [])
                                    else "UNRESOLVED",
                                    "theorem_id": "DISCRETE_TICK_FPPS_EMBEDDING"},
        "ZERO_RELATIVE_START": recurring,
        "INHERITED_HI_DOMINATION": recurring,
        "LO_MODE_RTA": rta,
        "WORST_CASE_START_TIME": rta,
        "CASE1_INTEGER_DOMAIN": rta,
        "CASE2_INTEGER_DOMAIN": rta,
        "RELEASE_COUNT": rta,
        "DEMAND_DOMINATION": mapping,
        "THEORY_LIBRARY_VERSION": {"status": theory_check.get("status"), "manifest": json.loads((ROOT / "formal_toolchain/theory/theory_manifest.json").read_text(encoding="utf-8")), "verification": theory_check},
        "TIME_PROGRESS": bridge["prerequisites"]["positive_time"],
        "CONTROLLER_POSTCLOSURE": bridge["prerequisites"]["controller_postclosure"],
        "BATCH_CLOSURE": bridge["prerequisites"]["same_timestamp"],
        "EFFECTIVE_EVENT_ORDER": bridge["prerequisites"]["event_projection"],
        "REFERENCE_TASKSET_SCHEDULABLE": {
            "obligation_id": "REFERENCE_TASKSET_SCHEDULABLE",
            "obligation_status": "UNRESOLVED",
            "status": "UNRESOLVED",
            "failure": {"route": "REFERENCE_CERTIFICATE_FAILED",
                        "code": "REFERENCE_TASKSET_SCHEDULABILITY_UNRESOLVED"},
        },
        "REFERENCE_HI_SUBSET_SAFETY": {
            "obligation_id": "REFERENCE_HI_SUBSET_SAFETY",
            "obligation_status": "UNRESOLVED",
            "status": "UNRESOLVED",
            "failure": {"route": "REFERENCE_CERTIFICATE_FAILED",
                        "code": "REFERENCE_HI_SUBSET_SAFETY_UNRESOLVED"},
        },
        "FINITE_BAD_PREFIX_CONTRADICTION": {
            "obligation_id": "FINITE_BAD_PREFIX_CONTRADICTION",
            "obligation_status": "UNRESOLVED",
            "status": "UNRESOLVED",
            "failure": {"route": "REFERENCE_CERTIFICATE_FAILED",
                        "code": "FINITE_BAD_PREFIX_CONTRADICTION_UNRESOLVED"},
        },
        "FINAL_CLAIM_COMPOSITION": {
            "obligation_id": "FINAL_CLAIM_COMPOSITION",
            "obligation_status": "UNRESOLVED",
            "status": "UNRESOLVED",
            "failure": {"route": "REFERENCE_CERTIFICATE_FAILED",
                        "code": "FINAL_CLAIM_COMPOSITION_UNRESOLVED"},
        },
    }
    # fresh Phase F-H registry evidence is authoritative. I-K may add missing
    # Phase-I/J/bridge nodes, but不得用较弱的本地摘要覆盖同名 fresh cert。
    for key, value in local_evidence.items():
        actual.setdefault(key, value)
    certificates: dict[str, Any] = {}
    by_id = {str(entry["id"]): entry for entry in entries}
    def build(obligation_id: str) -> dict[str, Any]:
        if obligation_id in certificates:
            return certificates[obligation_id]
        entry = by_id[obligation_id]
        source = actual.get(obligation_id)
        if not isinstance(source, Mapping):
            raise ValueError(f"registry 节点缺少真实证据: {obligation_id}")
        source_status = source.get("obligation_status", source.get("status"))
        if source_status not in {"PASS", "FAIL", "UNRESOLVED"}:
            raise ValueError(f"registry 节点状态无效: {obligation_id}")
        predecessors = {str(dep): build(str(dep)) for dep in entry.get("depends_on", [])
                        if str(dep) in closure}
        failure = None if source_status == "PASS" else {
            "route": str(source.get("route", "UNRESOLVED")),
            "code": str((source.get("failure") or {}).get("code", source.get("code", "REGISTRY_CERTIFICATE_FAILED"))),
            "detail": source.get("failure", source.get("message")),
        }
        cert = obligation_certificate(
            obligation_id=obligation_id, status=str(source_status), context_hash=context_hash,
            inputs={"registry_entry": entry, "source_artifact_hash": source.get("artifact_hash", sha256_object(dict(source)))},
            witness={"source_evidence": dict(source)},
            direct_predecessor_hashes={dep: sha256_object(value) for dep, value in predecessors.items()},
            checker_id="scripts.run_phase_ijk_acceptance.registry_closure",
            checker_version="phase-ijk-v3",
            failure=failure,
        )
        certificates[obligation_id] = cert
        return cert
    for obligation_id in closure:
        build(obligation_id)
    return certificates


def _unresolved(code: str, **details: Any) -> tuple[int, dict[str, Any]]:
    return 1, {"workflow_status": "UNRESOLVED", "phase": "I-K",
               "schema_version": "phase_ijk_result_v3_breaking",
               "migration_id": "phase-ijk-seventh-round-semantic-binding-v1",
               "phase_result": "PHASE_IJK_UNRESOLVED",
               "final_safety_claim": "NOT_EVALUATED",
               "unimplemented_later_phases": ["L", "M"],
               "failure": {"code": code, "route": "UNRESOLVED", **details}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="synthetic_p0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = ROOT
    fixture = root / "tests/formal/fixtures" / args.fixture
    required = ("target_recipe.json", "phase_ijk_inputs.json", "phase_k_case_map.json")
    if not fixture.is_dir():
        code, result = _unresolved("FIXTURE_NOT_FOUND")
    elif any(not (fixture / name).is_file() for name in required):
        code, result = _unresolved("AUTHORITATIVE_PHASE_IK_INPUT_MISSING",
                                   required=list(required))
    else:
        try:
            recipe = json.loads((fixture / "target_recipe.json").read_text(encoding="utf-8"))
            inputs = json.loads((fixture / "phase_ijk_inputs.json").read_text(encoding="utf-8"))
            # Phase F-H certificate 必须在本次正式流程中由 fresh verifier 生成；
            # 不把预先制作的 envelope 当作 Phase K 正向输入。
            with tempfile.TemporaryDirectory(prefix="phase_fh_") as fresh_dir:
                fresh = subprocess.run([sys.executable, str(ROOT / "scripts/run_phase_fh_acceptance.py"),
                                        "--fixture", args.fixture, "--out", fresh_dir],
                                       cwd=ROOT, capture_output=True, text=True, check=False)
                if fresh.returncode != 0:
                    raise ValueError("fresh Phase F-H verifier 未通过: " + fresh.stdout[-1000:])
                envelope = json.loads((Path(fresh_dir) / "verified/certified_envelope.json").read_text(encoding="utf-8"))
                envelope_certificate = json.loads((Path(fresh_dir) / "verified/certified_envelope_certificate.json").read_text(encoding="utf-8"))
                candidate_envelope = json.loads((Path(fresh_dir) / "candidate/candidate_envelope.json").read_text(encoding="utf-8"))
                common_preservation = json.loads((Path(fresh_dir) / "candidate/common_preservation.json").read_text(encoding="utf-8"))
                deployed_preservation = json.loads((Path(fresh_dir) / "candidate/deployed_preservation.json").read_text(encoding="utf-8"))
                fh_result = json.loads((Path(fresh_dir) / "proof_result.json").read_text(encoding="utf-8"))
            preservation = envelope.get("preservation_certificate")
            if (envelope.get("preservation_certificate_hash") != sha256_object(envelope_certificate)
                    or preservation != envelope_certificate
                    or envelope_certificate.get("obligation_status") != "PASS"
                    or envelope_certificate.get("evidence", [{}])[0].get("fresh_process") is not True):
                raise ValueError("certified envelope 未绑定 fresh Phase F-H verifier certificate")
            case_map = json.loads((fixture / "phase_k_case_map.json").read_text(encoding="utf-8"))
            if case_map.get("source_hash") != inputs.get("source_hash"):
                raise ValueError("Phase K case map source hash mismatch")
            if case_map.get("schema_version") != "phase_k_transition_path_map_v2_cfg_ir" or not isinstance(case_map.get("paths"), dict):
                raise ValueError("Phase K case map schema invalid")
            expected_theorems = {
                "casewise_simulation": "CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",
                "prefix_extension": "REFERENCE_PREFIX_EXTENSION",
                "bad_prefix_reflection": "FINITE_HI_BAD_PREFIX_REFLECTION",
            }
            if inputs.get("theorem_ids") != expected_theorems:
                raise ValueError("Phase K theorem_ids 必须精确绑定 theory manifest")
            target = build_target(recipe["factory"], recipe.get("kwargs", {}))
            reference = build_reference_taskset(
                target.ordered_tasks, inputs["budget_by_task"], xf=inputs["xf"],
                certified_envelope=envelope,
                semantic_context_hash=inputs["semantic_context_hash"],
                effective_runtime_config_hash=inputs["effective_runtime_config_hash"],
            )
            derived_budget = derive_budget_invariant_evidence(
                reference_taskset=reference.to_dict(),
                candidate=candidate_envelope,
                common=common_preservation,
                deployed=deployed_preservation,
                certified_envelope=envelope,
                certified_certificate=envelope_certificate,
            )
            mapping = verify_reference_mapping(
                reference=reference, ordered_tasks=target.ordered_tasks,
                budget_by_task=inputs["budget_by_task"], certified_envelope=envelope,
                xf=inputs["xf"], semantic_context_hash=inputs["semantic_context_hash"],
                effective_runtime_config_hash=inputs["effective_runtime_config_hash"],
            )
            if mapping.get("obligation_status") != "PASS":
                code, result = _unresolved("REFERENCE_MAPPING_FAILED", mapping=mapping)
            else:
                rta = protected_hi_rta(reference)
                replay = replay_rta(reference, rta)
                if not verify_obligation_certificate(rta) or replay.get("status") != "PASS":
                    code, result = _unresolved("RTA_OR_REPLAY_FAILED", rta=rta, replay=replay)
                else:
                    recurring_error = None
                    try:
                        recurring = build_recurring_hi_instances(reference, rta_certificate=rta)
                    except (KeyError, TypeError, ValueError) as exc:
                        recurring_error = str(exc)
                        recurring = {
                            "status": "UNRESOLVED",
                            "obligation_status": "UNRESOLVED",
                            "route": "REFERENCE_CERTIFICATE_FAILED",
                            "failure": {"code": "PER_HI_TASK_INDUCTIVE_WCRT_UNRESOLVED",
                                        "detail": recurring_error},
                        }
                    if recurring.get("status") == "PASS":
                        corollary = protected_hi_safety_corollary(recurring)
                    else:
                        corollary = {
                            "status": "UNRESOLVED",
                            "obligation_status": "UNRESOLVED",
                            "route": "REFERENCE_CERTIFICATE_FAILED",
                            "failure": {"code": "PROTECTED_HI_SAFETY_COROLLARY_UNRESOLVED",
                                        "detail": recurring.get("failure", recurring_error)},
                        }
                    branch_map = build_runtime_branch_map(
                        root, source_hash=inputs["source_hash"],
                        path_map=case_map,
                    )
                    if branch_map.get("status") != "PASS":
                        code, result = _unresolved("THEOREM_OR_BRANCH_BINDING_UNRESOLVED",
                                                   corollary=corollary, branch_map=branch_map)
                    else:
                        parameterized = build_parameterized_release_mapping_certificate(
                            source_context_hash=reference.source_context_hash)
                        if not verify_parameterized_release_mapping_certificate(parameterized):
                            code, result = _unresolved("PARAMETERIZED_RELEASE_MAPPING_INVALID")
                        else:
                            finite = None
                            if "release_mappings" in inputs:
                                mappings = [map_release_fixed_job(**row) for row in inputs["release_mappings"]]
                                finite_certificate = build_release_fixed_removal_certificate(
                                    mappings, source_context_hash=reference.source_context_hash)
                                finite = {"certificate": finite_certificate,
                                          "verified": verify_release_fixed_removal_certificate(finite_certificate)}
                            budget_domination = build_budget_to_reference_domination(
                                reference_taskset=reference.to_dict(), certified_envelope=envelope,
                                reference_context_hash=reference.source_context_hash,
                                direct_predecessor_hashes={},
                            )
                            model_bounds = derive_p0_model_bounds(reference.to_dict())
                            bridge_context_hash = sha256_object({
                                "schema_version": "bridge_context_v2",
                                "reference_context_hash": reference.source_context_hash,
                                "source_hash": branch_map["source_hash"],
                                "branch_map_hash": branch_map["path_map_hash"],
                                "p0_case_manifest_hash": p0_case_manifest_hash(),
                                "model_bounds_hash": model_bounds.fingerprint,
                            })
                            concrete_base, reference_base = _build_preclosed_runtime_states(
                                target, reference.to_dict())
                            upstream_source = dict(fh_result.get("registry_certificates", {}))
                            upstream_source["CERTIFIED_ENVELOPE"] = envelope_certificate
                            required_upstream = (
                                "SCHEDULER_MODEL", "MODE_SEMANTICS_CONFORMANCE",
                                "DEMAND_ORACLE_BATCH_CONTRACT", "HI_EXECUTION_CONTRACT",
                                "REMOVAL_COMPLETENESS", "HI_NONTRUNCATION", "DEADLINE_OBSERVATION",
                                "EFFECTIVE_EVENT_ORDER", "BATCH_CLOSURE", "CONTROLLER_POSTCLOSURE",
                                "TIME_PROGRESS", "WINDOW_MODE_NORMALIZATION", "CERTIFIED_ENVELOPE")
                            upstream = {name: _context_bind(upstream_source[name],
                                                              obligation_id=name,
                                                              context_hash=bridge_context_hash,
                                                              source_registry=upstream_source)
                                        for name in required_upstream}
                            protected_bound = None
                            if corollary.get("status") == "PASS":
                                protected_bound = _context_bind(
                                    corollary,
                                    obligation_id="PROTECTED_HI_SAFETY_COROLLARY",
                                    context_hash=bridge_context_hash,
                                )
                            early_stop_gate = build_early_stop_configuration_gate(
                                runtime_config=target.runtime_config,
                                context_hash=bridge_context_hash,
                                closure_completion_certificate=protected_bound,
                            )
                            effective_frontier_certificate = _build_effective_frontier_certificate(
                                concrete_base=concrete_base, context_hash=bridge_context_hash,
                            )
                            release_mapping_bound = _context_bind(
                                parameterized,
                                obligation_id="RELEASE_FIXED_REMOVAL_MAPPING",
                                context_hash=bridge_context_hash,
                            )
                            bridge = compile_phase_k(source_root=root, branch_map=branch_map,
                                                     reference_taskset=reference.to_dict(),
                                                     bridge_context_hash=bridge_context_hash,
                                                     model_bounds=model_bounds,
                                                     concrete_base=concrete_base,
                                                     reference_base=reference_base,
                                                     upstream_certificates=upstream,
                                                     release_mapping_certificate=release_mapping_bound,
                                                     closure_completion_certificate=protected_bound)
                            if bridge.get("status") == "PASS":
                                registry_certificates = _build_phase_ijk_registry_certificates(
                                    bridge=bridge,
                                    fh_registry=fh_result.get("registry_certificates", {}),
                                    envelope_certificate=envelope_certificate,
                                    mapping=mapping, reference=reference, rta=rta,
                                    recurring=recurring, corollary=corollary,
                                    parameterized=release_mapping_bound, branch_map=branch_map,
                                    budget_domination=budget_domination,
                                    early_stop_gate=early_stop_gate,
                                    effective_frontier_certificate=effective_frontier_certificate,
                                    context_hash=bridge_context_hash,
                                    target_domain={"status": "PASS", "budget_by_task": target.provenance.get("budget_by_task"),
                                                   "runtime_semantics": target.runtime_config.semantics.value},
                                    concrete_base=concrete_base,
                                    derived_budget=derived_budget)
                                registry_closure = verify_registry_local_closure(
                                    load_registry(ROOT / "formal_toolchain/specs/obligation_registry.json"),
                                    registry_certificates, context_hash=bridge_context_hash)
                                if registry_closure.get("status") != "PASS":
                                    raise ValueError("Phase I-K registry local closure failed: " + str(registry_closure))
                                code, result = 0, {"workflow_status": "VERIFIED", "phase": "I-K",
                                    "schema_version": "phase_ijk_result_v3_breaking",
                                    "migration_id": "phase-ijk-seventh-round-semantic-binding-v1",
                                    "phase_result": "PHASE_IJK_ACCEPTED", "final_safety_claim": "NOT_EVALUATED",
                                    "mapping": mapping, "rta": rta, "replay": replay,
                                    "recurring": recurring, "corollary": corollary,
                                    "branch_map": branch_map, "parameterized_release_mapping": parameterized,
                                    "finite_boundary_evidence": finite, "bridge": bridge,
                                    "registry_certificates": registry_certificates,
                                    "registry_closure": registry_closure}
                            else:
                                code, result = _unresolved("PARAMETERIZED_BRIDGE_COMPILATION_FAILED",
                                    parameterized_release_mapping=parameterized,
                                    finite_boundary_evidence=finite, bridge=bridge,
                                    required=["source_code", "phase_k_case_map", "z3"])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            code, result = _unresolved("PHASE_IK_INPUT_INVALID", message=str(exc))
    if args.out is not None:
        output_file = args.out if args.out.suffix == ".json" else args.out / "phase_ijk_result.json"
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file.write_text(canonical_dumps(result), encoding="utf-8")
        bridge = result.get("bridge")
        if isinstance(bridge, dict) and bridge.get("status") == "PASS":
            generated = {
                "branch_map.json": result["branch_map"],
                "transition_case_proofs.json": bridge["transition_cases"],
                "base_relation_certificate.json": bridge["prerequisites"]["base_relation"],
                "same_timestamp_closure_certificate.json": bridge["prerequisites"]["same_timestamp"],
                "positive_time_service_certificate.json": bridge["prerequisites"]["positive_time"],
                "controller_postclosure_certificate.json": bridge["prerequisites"]["controller_postclosure"],
                "event_projection_certificate.json": bridge["prerequisites"]["event_projection"],
                "closed_prefix_refinement_certificate.json": bridge["closed_prefix"],
                "reference_prefix_extension_certificate.json": bridge["reference_extension"],
                "hi_bad_prefix_reflection_certificate.json": bridge["bad_prefix_reflection"],
            }
            for name, artifact in generated.items():
                (output_dir / name).write_text(canonical_dumps(artifact), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
