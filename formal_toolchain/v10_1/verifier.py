"""End-to-end V10.1 BASE -> PCSSC verifier.

No Event Graph formula is allocated on the terminal PASS path.  Retired
terminal machinery is not imported or called by this verifier.
"""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any

from .kernel.environment_encoder import declare_environment
from .kernel.formula_solver import FormulaReceipt, solve_formula
from .kernel.universal_conformance import prove_universal_conformance
from .base_refinement import check_dynamic_to_base_refinement, run_original_c_amc_sem_schedulability_test
from .bindings import build_bindings, load_request
from .constants import (
    FRAMEWORK_REVISION, PRIMARY_CLAIM, PROOF_ROUTE, RESULT_INVALID, RESULT_PROVED, RESULT_UNRESOLVED,
    SCOPE, TARGET_PROVED_BASE, TARGET_PROVED_PCSSC, TARGET_PROVED_PCSSC_CASE_CONSISTENT,
    TARGET_PROVED_PCSSC_CASE_CONDITIONED_CARRY,
    TARGET_PROVED_PCSSC_MIXED_PHASE_TERMINALS_V10_17,
)
from .completion_certificates import (
    CompletionCertificateError,
    build_base_completion_certificates,
    completion_prefix_for_target,
    export_pcssc_completion_certificate,
    merge_certified_completion,
)
from .controller_macro import (
    ControllerMacroUnresolved, build_controller_macro_path, max_controller_activations,
)
from .pcssc import prove_target_pcssc
from .safe_prefix import (
    SchedulerSafePrefixInvariant, build_p5_scheduler_summary_soundness_obligations,
    certify_p7_exact_periodic_eta,
)
from .kernel.symbolic_state import BoundModel


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_progress(out: Path, stage: str, **details: Any) -> None:
    _write(out / "verifier_progress.json", {
        "schema_version": "v10_1_verifier_progress_v1",
        "stage": stage,
        **details,
    })



def _receipt_status(receipt: FormulaReceipt) -> str:
    return "PASS" if receipt.result == "UNSAT" else (
        "FAIL" if receipt.result == "SAT" else "UNRESOLVED"
    )


def _fail_summary(
    request: dict[str, Any],
    statuses: dict[str, str],
    *,
    code: str,
    message: str | None = None,
    binding_root_hash: str | None = None,
) -> dict[str, Any]:
    row = {
        "schema_version": "v10_1_verified_summary_v1",
        "workflow_status": "FAILED",
        "result_status": RESULT_UNRESOLVED if code != "C_AMC_SEM_SCOPE_BINDING_ERROR" else RESULT_INVALID,
        "failure_route": RESULT_UNRESOLVED,
        "failure_code": code,
        "failure_message": message,
        "proof_route": PROOF_ROUTE,
        "scope": SCOPE,
        "primary_claim": request.get("primary_claim", PRIMARY_CLAIM),
        "target_id": request.get("target_id"),
        "target_kind": request.get("target_kind"),
        "taskset_seed": request.get("taskset_seed"),
        "tree_variant": request.get("tree_variant"),
        "obligation_statuses": statuses,
        "event_graph_in_pass_dependency": False,
        "framework_revision": FRAMEWORK_REVISION,
    }
    if binding_root_hash is not None:
        row["binding_root_hash"] = binding_root_hash
    return row


def verify_bundle_v10_1(
    request_path: Path,
    out: Path,
    *,
    source_root: Path,
    timeout_ms: int = 0,
) -> dict[str, Any]:
    request_path = Path(request_path).resolve()
    out = Path(out).resolve()
    source_root = Path(source_root).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    statuses: dict[str, str] = {}
    receipts: dict[str, Any] = {
        "schema_version": "v10_1_proof_receipts_v1",
        "proof_route": PROOF_ROUTE,
        "framework_revision": FRAMEWORK_REVISION,
        "timeout_ms": int(timeout_ms),
        "event_graph_in_pass_dependency": False,
    }

    try:
        request = load_request(request_path)
        bindings = build_bindings(request_path, source_root=source_root)
        model = BoundModel.from_bindings(bindings, max_jobs_per_task=2)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        # If request loading itself failed, recover only enough identity for a
        # clear fail-closed summary.
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except Exception:
            request = {}
        summary = _fail_summary(
            request, statuses, code="C_AMC_SEM_SCOPE_BINDING_ERROR", message=str(exc)
        )
        _write(out / "proof_summary.json", summary)
        return summary

    binding_hash = str(bindings["binding_root_hash"])
    statuses.update({
        "ADMISSIBLE_ENVIRONMENT_BINDING": "PASS",
        "P0_EVENT_ORDER_BINDING": "PASS",
        "NUMERIC_OBSERVATION_BINDING": "PASS",
        "EXPLICIT_NOOP_AND_FIRST_VALID_BINDING": "PASS",
        "FINITE_SAME_TIMESTAMP_CLOSURE": "PASS",
        "CONSTRAINED_DEADLINE_D_LE_T": "PASS" if all(t.deadline <= t.period for t in model.tasks) else "FAIL",
    })
    if statuses["CONSTRAINED_DEADLINE_D_LE_T"] != "PASS":
        summary = _fail_summary(request, statuses, code="C_AMC_SEM_SCOPE_BINDING_ERROR",
                                message="V10.1 requires D<=T for every bound task",
                                binding_root_hash=binding_hash)
        _write(out / "proof_summary.json", summary)
        return summary

    _write_progress(out, "FULL_KERNEL_CONFORMANCE", timeout_ms=int(timeout_ms))
    conformance = prove_universal_conformance(model, source_root=source_root, timeout_ms=timeout_ms)
    receipts["full_kernel_conformance"] = conformance.as_dict()
    if conformance.status != "PASS":
        value = "FAIL" if conformance.status == "FAIL" else "UNRESOLVED"
        statuses.update({
            "POLICY_TIMING_KERNEL_STEP_CONFORMANCE": value,
            "TIMING_PROJECTION_PREFIX_REFINEMENT": value,
            "FIRST_HI_BAD_PREFIX_REFLECTION": value,
        })
        summary = _fail_summary(
            request, statuses,
            code=conformance.failure_code or "CERTIFICATE_SOUNDNESS_OBLIGATION_MISSING",
            binding_root_hash=binding_hash,
        )
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary
    statuses.update({
        "POLICY_TIMING_KERNEL_STEP_CONFORMANCE": "PASS",
        "TIMING_PROJECTION_PREFIX_REFINEMENT": "PASS",
        "FIRST_HI_BAD_PREFIX_REFLECTION": "PASS",
    })

    # V10.1 still needs the inherited SafePrefix bridge from the concrete P0
    # kernel to arbitrary first-bad prefixes.  This is not an Event-Graph
    # terminal: it is the common inductive domain used by first-bad reflection
    # and by the full-legal controller feature envelope.
    _write_progress(out, "P5_CONTROLLER_SUMMARY_SOUNDNESS", timeout_ms=int(timeout_ms))
    p5_rows: list[dict[str, Any]] = []
    for obligation in build_p5_scheduler_summary_soundness_obligations(
        model, prefix="v10.verify.p5.summary"
    ):
        receipt = solve_formula(
            obligation.obligation_id, obligation.counterexample, timeout_ms=timeout_ms
        )
        row = receipt.as_dict()
        row["explanation"] = obligation.explanation
        p5_rows.append(row)
        if receipt.result != "UNSAT":
            statuses["P5_CONTROLLER_SUMMARY_SOUNDNESS"] = _receipt_status(receipt)
            receipts["p5_controller_summary_soundness"] = p5_rows
            summary = _fail_summary(
                request, statuses,
                code=f"P5_CONTROLLER_SUMMARY_{receipt.result}:{obligation.obligation_id}",
                binding_root_hash=binding_hash,
            )
            _write(out / "proof_receipts.json", receipts)
            _write(out / "proof_summary.json", summary)
            return summary
    receipts["p5_controller_summary_soundness"] = p5_rows
    statuses["P5_CONTROLLER_SUMMARY_SOUNDNESS"] = "PASS"

    invariant = SchedulerSafePrefixInvariant(model)
    _write_progress(out, "SAFE_PREFIX_INITIAL", timeout_ms=int(timeout_ms))
    initial = solve_formula(
        "SAFE_PREFIX_INVARIANT_INITIAL_COUNTEREXAMPLE",
        invariant.initial_counterexample(prefix="v10.verify.initial"),
        timeout_ms=timeout_ms,
    )
    receipts["safe_prefix_initial"] = initial.as_dict()
    statuses["SAFE_PREFIX_INVARIANT_INITIAL"] = _receipt_status(initial)
    if initial.result != "UNSAT":
        summary = _fail_summary(
            request, statuses, code=f"SAFE_PREFIX_INITIAL_{initial.result}",
            binding_root_hash=binding_hash,
        )
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary

    _write_progress(out, "SAFE_PREFIX_CONDITIONAL_INDUCTIVENESS", timeout_ms=int(timeout_ms))
    phase_rows: list[dict[str, Any]] = []
    for phase in range(8):
        ind_env = declare_environment(
            f"v10.verify.ind.p{phase}.env", model, release_count=1
        )
        if phase == 7:
            child_rows: list[dict[str, Any]] = []
            failed_child: FormulaReceipt | None = None
            failed_clause: str | None = None

            eta_certificate = certify_p7_exact_periodic_eta(model)
            child_rows.append(eta_certificate)
            if eta_certificate["result"] != "UNSAT":
                statuses["SAFE_PREFIX_INDUCTIVE_P7"] = "FAIL"
                statuses["SAFE_PREFIX_INVARIANT_CONDITIONAL_INDUCTIVENESS"] = "FAIL"
                phase_rows.append({
                    "obligation_id": "SAFE_PREFIX_INDUCTIVE_P7",
                    "result": str(eta_certificate["result"]),
                    "decomposition": (
                        "FINITE_PERIODIC_ETA_CERTIFICATE_PLUS_NAMED_POST_INVARIANT_"
                        "CONJUNCTS_WITH_SPARSE_P7_SSA"
                    ),
                    "children": child_rows,
                })
                receipts["safe_prefix_conditional_inductiveness_by_phase"] = phase_rows
                summary = _fail_summary(
                    request, statuses,
                    code="SAFE_PREFIX_INDUCTIVE_P7_SAT:exact_periodic_eta",
                    binding_root_hash=binding_hash,
                )
                _write(out / "proof_receipts.json", receipts)
                _write(out / "proof_summary.json", summary)
                return summary

            for obligation in invariant.p7_clause_inductiveness_obligations(
                ind_env, prefix="v10.verify.ind"
            ):
                child = solve_formula(
                    obligation.obligation_id, obligation.counterexample,
                    timeout_ms=timeout_ms, capture_model=True,
                )
                child_row = child.as_dict(include_model=child.result == "SAT")
                child_row["clause_name"] = obligation.clause_name
                child_row["explanation"] = obligation.explanation
                child_rows.append(child_row)
                if child.result != "UNSAT":
                    failed_child = child
                    failed_clause = obligation.clause_name
                    break
            p7_status = "PASS" if failed_child is None else _receipt_status(failed_child)
            statuses["SAFE_PREFIX_INDUCTIVE_P7"] = p7_status
            phase_rows.append({
                "obligation_id": "SAFE_PREFIX_INDUCTIVE_P7",
                "result": "UNSAT" if failed_child is None else failed_child.result,
                "decomposition": (
                    "FINITE_PERIODIC_ETA_CERTIFICATE_PLUS_NAMED_POST_INVARIANT_"
                    "CONJUNCTS_WITH_SPARSE_P7_SSA"
                ),
                "children": child_rows,
            })
            if failed_child is not None:
                statuses["SAFE_PREFIX_INVARIANT_CONDITIONAL_INDUCTIVENESS"] = p7_status
                receipts["safe_prefix_conditional_inductiveness_by_phase"] = phase_rows
                summary = _fail_summary(
                    request, statuses,
                    code=f"SAFE_PREFIX_INDUCTIVE_P7_{failed_child.result}:{failed_clause}",
                    binding_root_hash=binding_hash,
                )
                _write(out / "proof_receipts.json", receipts)
                _write(out / "proof_summary.json", summary)
                return summary
            continue

        inductive = solve_formula(
            f"SAFE_PREFIX_INDUCTIVE_P{phase}",
            invariant.phase_inductiveness_counterexample(
                ind_env, phase, prefix="v10.verify.ind", use_p5_summary=(phase == 5)
            ),
            timeout_ms=timeout_ms,
            capture_model=True,
        )
        phase_rows.append(inductive.as_dict(include_model=inductive.result == "SAT"))
        statuses[f"SAFE_PREFIX_INDUCTIVE_P{phase}"] = _receipt_status(inductive)
        if inductive.result != "UNSAT":
            statuses["SAFE_PREFIX_INVARIANT_CONDITIONAL_INDUCTIVENESS"] = _receipt_status(inductive)
            receipts["safe_prefix_conditional_inductiveness_by_phase"] = phase_rows
            summary = _fail_summary(
                request, statuses, code=f"SAFE_PREFIX_INDUCTIVE_P{phase}_{inductive.result}",
                binding_root_hash=binding_hash,
            )
            _write(out / "proof_receipts.json", receipts)
            _write(out / "proof_summary.json", summary)
            return summary
    receipts["safe_prefix_conditional_inductiveness_by_phase"] = phase_rows
    statuses["SAFE_PREFIX_INVARIANT_CONDITIONAL_INDUCTIVENESS"] = "PASS"
    statuses["SAFE_PREFIX_FIRST_BAD_REACHABILITY_BRIDGE"] = "PASS"
    _write(out / "proof_receipts.partial.json", receipts)

    # BASE route: first establish that every deployed service trace is inside
    # the original C-AMC-sem paper scope.  A failure here is a scope error, not
    # a reason to let PCSSC silently prove a different system.
    _write_progress(out, "BASE_C_AMC_SEM_REFINEMENT")
    base_refinement = check_dynamic_to_base_refinement(model, bindings)
    receipts["base_c_amc_sem"] = {"refinement": base_refinement.as_dict()}
    statuses["DYNAMIC_TO_BASE_C_AMC_SEM_TRACE_REFINEMENT"] = base_refinement.status
    if base_refinement.status != "PASS":
        summary = _fail_summary(
            request, statuses, code="C_AMC_SEM_SCOPE_BINDING_ERROR",
            message=base_refinement.failure_code or "dynamic service exceeds paper C-AMC-sem scope",
            binding_root_hash=binding_hash,
        )
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary

    base_refinement_hash = sha256(
        json.dumps(
            base_refinement.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    base_sched = run_original_c_amc_sem_schedulability_test(model)
    receipts["base_c_amc_sem"]["section4_1"] = base_sched
    statuses["BASE_C_AMC_SEM_SECTION4_1_CERTIFICATE"] = str(base_sched["status"])
    if base_sched["status"] not in {"PASS", "UNRESOLVED", "FAIL"}:
        summary = _fail_summary(
            request, statuses, code="BASE_C_AMC_SEM_CERTIFICATE_STATUS_INVALID",
            message=str(base_sched["status"]), binding_root_hash=binding_hash,
        )
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary

    # V10.17 keeps one strict-priority completion map.  BASE entries are
    # unconditional; later PCSSC entries carry guarded safe-prefix semantics and
    # are appended only after a fully closed target certificate exists.
    try:
        certified_completion_by_task = build_base_completion_certificates(
            model,
            {
                str(name): int(bound)
                for name, bound in base_sched.get("completion_bound_by_task", {}).items()
            },
        )
    except CompletionCertificateError as exc:
        statuses["CERTIFIED_COMPLETION_PREFIX_SOUND"] = "UNRESOLVED"
        summary = _fail_summary(
            request, statuses, code="CERTIFICATE_SOUNDNESS_OBLIGATION_MISSING",
            message=str(exc), binding_root_hash=binding_hash,
        )
        receipts["certified_completion_propagation_failure"] = str(exc)
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary
    receipts["certified_completion_propagation"] = {
        "obligation_id": "CERTIFIED_HP_COMPLETION_PROPAGATION",
        "status": "PASS",
        "framework_revision": FRAMEWORK_REVISION,
        "priority_order": [task.name for task in model.tasks],
        "initial_base_certificates": {
            name: certificate.as_dict()
            for name, certificate in certified_completion_by_task.items()
        },
        "base_unconditional_exports": [
            {
                "obligation_id": (
                    f"BASE_UNCONDITIONAL_COMPLETION_EXPORT::{name},"
                    f"Rbase={certificate.response_bound},{base_refinement_hash}"
                ),
                "status": "PASS",
                "task": name,
                "response_bound": int(certificate.response_bound),
                "theorem_basis": certificate.theorem_basis,
                "base_refine_hash": base_refinement_hash,
                "task_specific_wcrt_bound": True,
                "completion_correspondence_bound": True,
            }
            for name, certificate in certified_completion_by_task.items()
        ],
        "pcssc_exports": [],
        "target_prefix_uses": [],
        "controller_macro_rebuilds_due_to_propagation": 0,
    }
    statuses["CERTIFIED_COMPLETION_PREFIX_SOUND"] = "PASS"
    statuses["PRIORITY_ORDERED_CERTIFICATE_DAG_ACYCLIC"] = "PASS"

    # V10.1 proves HI safety, not all-task deadline schedulability.  The Section
    # 4.1 analyzer runs in fixed-priority order and stops at the first failed
    # task, hence every successful task before that point is a certified FP
    # prefix.  A lower-priority LO failure cannot invalidate an already-proved
    # higher-priority HI target.
    base_safe_names = set(str(name) for name in base_sched.get("hi_safe_targets", ()))
    pending_hi_tasks = tuple(task for task in model.hi_tasks if task.name not in base_safe_names)
    statuses["BASE_C_AMC_SEM_HI_PREFIX_CERTIFICATE"] = (
        "PASS" if not pending_hi_tasks else "UNRESOLVED"
    )

    target_rows_by_name: dict[str, dict[str, Any]] = {}
    for task in model.hi_tasks:
        if task.name not in base_safe_names:
            continue
        statuses[f"HI_TARGET_SAFE::{task.name}"] = "PASS"
        statuses[f"TERMINAL_ROUTE::{task.name}"] = TARGET_PROVED_BASE
        base_completion_certificate = certified_completion_by_task.get(task.name)
        target_rows_by_name[task.name] = {
            "target": task.name,
            "status": "PASS",
            "terminal_route": TARGET_PROVED_BASE,
            "certificate": "ZHANG_ZHENG_GU_2024_SECTION_4_1_HI_PREFIX",
            "response_bound": (
                None if base_completion_certificate is None
                else int(base_completion_certificate.response_bound)
            ),
            "completion_certificate": (
                None if base_completion_certificate is None
                else base_completion_certificate.as_dict()
            ),
            "prefix_rule": base_sched.get("hi_prefix_rule"),
        }

    if not pending_hi_tasks:
        statuses["ALL_HI_TARGETS_SAFE"] = "PASS"
        receipts["certified_completion_propagation"]["final_certificates"] = {
            name: certificate.as_dict()
            for name, certificate in certified_completion_by_task.items()
        }
        base_targets = [target_rows_by_name[task.name] for task in model.hi_tasks]
        summary = {
            "schema_version": "v10_1_verified_summary_v1",
            "workflow_status": "PASS",
            "result_status": RESULT_PROVED,
            "proof_route": PROOF_ROUTE,
            "framework_revision": FRAMEWORK_REVISION,
            "scope": SCOPE,
            "primary_claim": request["primary_claim"],
            "target_id": request["target_id"],
            "target_kind": request["target_kind"],
            "taskset_seed": request["taskset_seed"],
            "tree_variant": request["tree_variant"],
            "binding_root_hash": binding_hash,
            "obligation_statuses": statuses,
            "target_certificates": base_targets,
            "base_route_status": base_sched["status"],
            "base_hi_route_status": "PASS",
            "event_graph_in_pass_dependency": False,
            "terminal_semantics": "BASE_C_AMC_SEM_SECTION4_1_HI_PREFIX",
        }
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        _write_progress(out, "COMPLETE", result_status=RESULT_PROVED)
        return summary

    # A full-task BASE failure is only a sufficient-test failure.  Already
    # BASE-proved HI targets are retained; PCSSC is constructed only for HI
    # targets that were not covered by the successful priority prefix.

    max_deadline = max(int(task.deadline) for task in pending_hi_tasks)
    max_depth = max_controller_activations(max_deadline, model.agent_period)
    _write_progress(out, "BUILD_CONTROLLER_MACRO", max_controller_activations=max_depth)
    try:
        controller_path = build_controller_macro_path(
            model, max_activations=max_depth, timeout_ms=timeout_ms
        )
    except (ControllerMacroUnresolved, ValueError, KeyError) as exc:
        message = str(exc)
        if message.startswith("FEATURE_"):
            statuses["FEATURE_TRANSFER_COVERAGE"] = "UNRESOLVED"
            failure_code = "FEATURE_TRANSFER_UNRESOLVED"
        else:
            statuses["CONTROLLER_PREFIX_COVERAGE"] = "UNRESOLVED"
            failure_code = "CONTROLLER_PREFIX_COVERAGE_UNRESOLVED"
        summary = _fail_summary(
            request, statuses, code=failure_code,
            message=message, binding_root_hash=binding_hash,
        )
        receipts["controller_macro_failure"] = str(exc)
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary
    receipts["controller_macro"] = controller_path.as_dict()
    statuses["BOOT_REACHABLE_BUDGET_INVARIANT"] = "PASS"
    statuses["FLOW_START_SOUND"] = "PASS"
    statuses["INTER_EPOCH_FLOW_SOUND"] = "PASS"
    statuses["FEATURE_TRANSFER_COVERAGE"] = "PASS"
    statuses["FULL_LEGAL_FEATURE_DOMAIN_REACHABILITY_BRIDGE"] = "PASS"
    statuses["GUARDED_FIRSTVALID_CONTROLLER_IMAGE_SOUND"] = "PASS"

    unresolved: list[str] = []
    for index, task in enumerate(pending_hi_tasks):
        _write_progress(
            out, "PCSSC_TARGET", task=task.name, task_index=index,
            hi_task_count=len(pending_hi_tasks), deadline=int(task.deadline),
            max_controller_activations=max_controller_activations(task.deadline, model.agent_period),
        )
        try:
            completion_prefix = completion_prefix_for_target(
                model, task.name, certified_completion_by_task
            )
        except CompletionCertificateError as exc:
            statuses["PRIORITY_ORDERED_CERTIFICATE_DAG_ACYCLIC"] = "UNRESOLVED"
            statuses[f"HI_TARGET_SAFE::{task.name}"] = "UNRESOLVED"
            unresolved.append(f"{task.name}:{exc}")
            receipts["certified_completion_propagation"]["status"] = "UNRESOLVED"
            receipts["certified_completion_propagation"]["target_prefix_uses"].append({
                "target": task.name, "status": "UNRESOLVED", "failure": str(exc),
            })
            target_rows_by_name[task.name] = {
                "target": task.name,
                "status": "UNRESOLVED",
                "response_bound": None,
                "failure_code": str(exc),
                "receipts": [],
                "conservatism_ledger": [],
                "tested_horizons": [],
            }
            _write(out / "proof_receipts.partial.json", receipts)
            continue
        receipts["certified_completion_propagation"]["target_prefix_uses"].append({
            "target": task.name,
            "status": "PASS",
            "certificates": {
                name: certificate.as_dict()
                for name, certificate in completion_prefix.items()
            },
        })
        cert = prove_target_pcssc(
            model, task.name, controller_path,
            priority_assignment_hash=str(bindings["seed_task_binding"]["priority_assignment_hash"]),
            tie_break_hash=str(bindings["seed_task_binding"]["tie_break_hash"]),
            release_model=str(bindings["environment_binding"]["domain"]["release_model"]),
            release_model_hash=str(bindings["environment_binding"]["release_model_hash"]),
            release_domain_hash=str(bindings["environment_binding"]["release_domain_hash"]),
            source_manifest_semantic_hash=str(bindings["source_manifest_semantic_hash"]),
            release_generator_source_hash=str(
                bindings["p0_event_order_binding"]["source_hashes"]["event_runtime"]
            ),
            certified_completion_by_task=completion_prefix,
        )
        row = cert.as_dict()
        target_rows_by_name[task.name] = row
        receipts.setdefault("pcssc_targets", []).append(row)
        statuses[f"HI_TARGET_SAFE::{task.name}"] = "PASS" if cert.status == "PASS" else "UNRESOLVED"
        if cert.status == "PASS":
            if cert.terminal_route not in {
                TARGET_PROVED_PCSSC,
                TARGET_PROVED_PCSSC_CASE_CONSISTENT,
                TARGET_PROVED_PCSSC_CASE_CONDITIONED_CARRY,
                TARGET_PROVED_PCSSC_MIXED_PHASE_TERMINALS_V10_17,
            }:
                statuses[f"HI_TARGET_SAFE::{task.name}"] = "UNRESOLVED"
                unresolved.append(f"{task.name}:PCSSC_PASS_MISSING_VALID_TERMINAL_ROUTE")
                row["status"] = "UNRESOLVED"
                row["completion_export_failure"] = "PCSSC_PASS_MISSING_VALID_TERMINAL_ROUTE"
                _write(out / "proof_receipts.partial.json", receipts)
                continue
            if cert.completion_theorem_basis is None:
                statuses[f"HI_TARGET_SAFE::{task.name}"] = "UNRESOLVED"
                unresolved.append(f"{task.name}:PCSSC_PASS_MISSING_COMPLETION_THEOREM_BASIS")
                row["status"] = "UNRESOLVED"
                row["completion_export_failure"] = "PCSSC_PASS_MISSING_COMPLETION_THEOREM_BASIS"
                _write(out / "proof_receipts.partial.json", receipts)
                continue
            statuses[f"TERMINAL_ROUTE::{task.name}"] = cert.terminal_route
            try:
                exported = export_pcssc_completion_certificate(
                    model, task.name, status=cert.status, response_bound=cert.response_bound,
                    theorem_basis=cert.completion_theorem_basis,
                )
                certified_completion_by_task[task.name] = merge_certified_completion(
                    certified_completion_by_task.get(task.name), exported
                )
            except CompletionCertificateError as exc:
                statuses[f"CERTIFIED_COMPLETION_SOURCE::{task.name}"] = "UNRESOLVED"
                statuses[f"HI_TARGET_SAFE::{task.name}"] = "UNRESOLVED"
                statuses.pop(f"TERMINAL_ROUTE::{task.name}", None)
                unresolved.append(f"{task.name}:{exc}")
                receipts["certified_completion_propagation"]["status"] = "UNRESOLVED"
                row["status"] = "UNRESOLVED"
                row["completion_export_failure"] = str(exc)
            else:
                statuses[f"CERTIFIED_COMPLETION_SOURCE::{task.name}"] = "PASS"
                row["completion_certificate"] = exported.as_dict()
                receipts["certified_completion_propagation"]["pcssc_exports"].append({
                    "obligation_id": f"CERTIFIED_COMPLETION_SOURCE::{task.name}",
                    "status": "PASS",
                    **exported.as_dict(),
                })
        else:
            unresolved.append(f"{task.name}:{cert.failure_code}")
        _write(out / "proof_receipts.partial.json", receipts)

    target_rows = [target_rows_by_name[task.name] for task in model.hi_tasks]
    receipts["certified_completion_propagation"]["final_certificates"] = {
        name: certificate.as_dict()
        for name, certificate in certified_completion_by_task.items()
    }

    if unresolved:
        summary = _fail_summary(
            request, statuses, code="POLICY_SINGLE_SWITCH_CERTIFICATE_UNRESOLVED",
            message=",".join(unresolved), binding_root_hash=binding_hash,
        )
        summary["target_certificates"] = target_rows
        summary["base_route_status"] = base_sched["status"]
        summary["base_hi_route_status"] = str(base_sched.get("hi_safety_status", "UNRESOLVED"))
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        _write_progress(out, "COMPLETE", result_status=RESULT_UNRESOLVED,
                        unresolved_targets=unresolved)
        return summary

    statuses["ALL_HI_TARGETS_SAFE"] = "PASS"
    summary = {
        "schema_version": "v10_1_verified_summary_v1",
        "workflow_status": "PASS",
        "result_status": RESULT_PROVED,
        "proof_route": PROOF_ROUTE,
        "scope": SCOPE,
        "primary_claim": request["primary_claim"],
        "target_id": request["target_id"],
        "target_kind": request["target_kind"],
        "taskset_seed": request["taskset_seed"],
        "tree_variant": request["tree_variant"],
        "binding_root_hash": binding_hash,
        "obligation_statuses": statuses,
        "target_certificates": target_rows,
        "base_route_status": base_sched["status"],
        "base_hi_route_status": str(base_sched.get("hi_safety_status", "UNRESOLVED")),
        "event_graph_in_pass_dependency": False,
        "terminal_semantics": "BASE_OR_PCSSC_CASE_CONSISTENT_WITH_V10_13_LO_ENTRY_AND_V10_17_MIXED_PRE_HI_PHASE_TERMINALS",
    }
    _write(out / "proof_receipts.json", receipts)
    _write(out / "proof_summary.json", summary)
    _write_progress(out, "COMPLETE", result_status=RESULT_PROVED)
    return summary


__all__ = ["verify_bundle_v10_1"]
