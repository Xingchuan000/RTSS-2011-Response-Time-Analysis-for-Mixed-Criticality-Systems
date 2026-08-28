"""Trusted, verifier-regenerated V9.2 end-to-end proof checker.

The candidate bundle is intentionally untrusted.  It transports frozen binding
identity only.  Every mathematical formula used for the final claim is rebuilt
from request + current source in this process and solved in a fresh Z3 solver.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.v9_2.bindings import build_bindings, load_request
from formal_toolchain.v9_2.carry_in import build_two_slot_carry_in_obligations
from formal_toolchain.v9_2.counterexample_replay import (
    DeployedRuntimeCounterexampleReplayer, classify_sat_event_window,
)
from formal_toolchain.v9_2.constants import (
    PROOF_ROUTE, RESULT_CONCRETE_COUNTEREXAMPLE, RESULT_INVALID, RESULT_PROVED,
    RESULT_UNRESOLVED, SCOPE,
)
from formal_toolchain.v9_2.environment_encoder import declare_environment
from formal_toolchain.v9_2.formula_solver import FormulaReceipt, solve_formula
from formal_toolchain.v9_2.p5_summary import build_p5_summary_soundness_obligations
from formal_toolchain.v9_2.readiness import blocker_rows, proof_pipeline_ready
from formal_toolchain.v9_2.safe_prefix_invariant import SafePrefixInvariant
from formal_toolchain.v9_2.symbolic_state import BoundModel
from formal_toolchain.v9_2.universal_conformance import prove_universal_conformance
from formal_toolchain.v9_2.event_refinement import prove_event_refinement
from formal_toolchain.v9_2.event_window_encoder import (
    ENCODER_VERSION, build_incremental_event_first_bad_window, derive_finite_event_bound,
)
from formal_toolchain.v9_2.incremental_event_bmc import (
    SOLVER_STRATEGY, solve_incremental_event_window,
)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")




def _write_progress(out: Path, stage: str, **details: Any) -> None:
    _write(out / "verifier_progress.json", {
        "schema_version": "v9_2_verifier_progress_v1",
        "stage": stage,
        **details,
    })


def _estimated_symbols_per_state(model: BoundModel) -> int:
    # t,p,mode + budgets/eta + 13 symbolic fields per job + frontier/ledger
    # + four per-task history signals + five event-count windows.
    task_count = len(model.tasks)
    return (6 + 6 * task_count + 13 * task_count * model.max_jobs_per_task
            + 5 * model.event_window)

def _fail_summary(request: dict[str, Any], statuses: dict[str, str], *, code: str,
                  message: str | None = None, result: str = RESULT_UNRESOLVED) -> dict[str, Any]:
    return {
        "schema_version": "v9_2_verified_summary_v2",
        "workflow_status": "FAILED",
        "result_status": result,
        "failure_route": result,
        "failure_code": code,
        "failure_message": message,
        "proof_route": PROOF_ROUTE,
        "scope": SCOPE,
        "primary_claim": request["primary_claim"],
        "target_id": request["target_id"],
        "target_kind": request["target_kind"],
        "taskset_seed": request["taskset_seed"],
        "tree_variant": request["tree_variant"],
        "obligation_statuses": statuses,
    }


def _candidate_integrity(candidate: dict[str, Any]) -> bool:
    declared = candidate.get("candidate_root_hash")
    if not isinstance(declared, str):
        return False
    payload = dict(candidate)
    payload.pop("candidate_root_hash", None)
    return sha256_object(payload) == declared


def _receipt_status(receipt: FormulaReceipt) -> str:
    return "PASS" if receipt.result == "UNSAT" else (
        "FAIL" if receipt.result == "SAT" else "UNRESOLVED"
    )


def _proof_summary(
    request: dict[str, Any],
    statuses: dict[str, str],
    *,
    binding_root_hash: str,
    receipts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "v9_2_verified_summary_v2",
        "workflow_status": "PASS",
        "result_status": RESULT_PROVED,
        "proof_route": PROOF_ROUTE,
        "scope": SCOPE,
        "primary_claim": request["primary_claim"],
        "target_id": request["target_id"],
        "target_kind": request["target_kind"],
        "taskset_seed": request["taskset_seed"],
        "tree_variant": request["tree_variant"],
        "binding_root_hash": binding_root_hash,
        "event_window_encoder_version": ENCODER_VERSION,
        "obligation_statuses": statuses,
        "proof_receipt_hash": sha256_object(receipts),
        "event_layer_added_abstractions": [],
        "exact_event_macro_semantics": True,
        "event_to_full_realizability_verified": True,
        "small_horizon_differential_consistency_verified": True,
        "exact_p5_in_event_window": True,
        "microstep_terminal_fallback_used": False,
    }


def verify_bundle_v9_2(
    request_path: Path,
    bundle: Path,
    out: Path,
    *,
    source_root: Path,
    timeout_ms: int = 120_000,
    concrete_replayer: Any = None,
    max_boot_replay_ticks: int = 2_000,
) -> dict[str, Any]:
    request_path = Path(request_path).resolve()
    bundle = Path(bundle).resolve()
    out = Path(out).resolve()
    source_root = Path(source_root).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    request = load_request(request_path)
    statuses: dict[str, str] = {}
    receipts: dict[str, Any] = {
        "schema_version": "v9_2_event_fresh_proof_receipts_v1",
        "proof_route": PROOF_ROUTE,
        "timeout_ms": int(timeout_ms),
        "candidate_assertions_trusted": False,
    }

    candidate_path = bundle / "candidate_manifest.json"
    bindings_path = bundle / "bindings.json"
    if not candidate_path.is_file() or not bindings_path.is_file():
        summary = _fail_summary(
            request, statuses, code="V9_2_CANDIDATE_BUNDLE_INCOMPLETE", result=RESULT_INVALID
        )
        _write(out / "proof_summary.json", summary)
        return summary
    candidate = _read_json(candidate_path)
    candidate_bindings = _read_json(bindings_path)
    if candidate.get("proof_route") != PROOF_ROUTE or not _candidate_integrity(candidate):
        summary = _fail_summary(
            request, statuses, code="V9_2_CANDIDATE_MANIFEST_INTEGRITY_FAILED", result=RESULT_INVALID
        )
        _write(out / "proof_summary.json", summary)
        return summary

    try:
        recomputed = build_bindings(request_path, source_root=source_root)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        summary = _fail_summary(
            request, statuses, code="V9_2_BINDING_REGENERATION_FAILED",
            message=str(exc), result=RESULT_INVALID,
        )
        _write(out / "proof_summary.json", summary)
        return summary
    if candidate_bindings.get("binding_root_hash") != recomputed["binding_root_hash"]:
        summary = _fail_summary(request, statuses, code="BINDING_RECOMPUTE_MISMATCH", result=RESULT_INVALID)
        _write(out / "proof_summary.json", summary)
        return summary
    if candidate.get("binding_root_hash") != recomputed["binding_root_hash"]:
        summary = _fail_summary(request, statuses, code="CANDIDATE_BINDING_ROOT_MISMATCH", result=RESULT_INVALID)
        _write(out / "proof_summary.json", summary)
        return summary

    statuses.update({
        "ADMISSIBLE_ENVIRONMENT_BINDING": "PASS",
        "P0_EVENT_ORDER_BINDING": "PASS",
        "NUMERIC_OBSERVATION_BINDING": "PASS",
        "EXPLICIT_NOOP_AND_FIRST_VALID_BINDING": "PASS",
        "FINITE_SAME_TIMESTAMP_CLOSURE": "PASS",
    })

    if not proof_pipeline_ready():
        summary = _fail_summary(
            request,
            statuses,
            code="EVENT_WINDOW_ENCODING_UNRESOLVED",
            message="V9.2 end-to-end proof implementation still has explicit readiness blockers.",
        )
        summary["binding_root_hash"] = recomputed["binding_root_hash"]
        summary["event_window_encoder_version"] = ENCODER_VERSION
        summary["implementation_gaps"] = blocker_rows()
        _write(out / "proof_summary.json", summary)
        return summary

    try:
        model = BoundModel.from_bindings(recomputed, max_jobs_per_task=2)
    except (ValueError, TypeError, KeyError) as exc:
        summary = _fail_summary(
            request, statuses, code="V9_2_BOUND_MODEL_REGENERATION_FAILED", message=str(exc)
        )
        _write(out / "proof_summary.json", summary)
        return summary
    if any(task.deadline > task.period for task in model.tasks):
        summary = _fail_summary(
            request, statuses, code="V9_2_TWO_SLOT_CARRY_IN_REQUIRES_D_LE_T"
        )
        _write(out / "proof_summary.json", summary)
        return summary

    # 1) Source-bound universal frozen-runtime correspondence.
    _write_progress(out, "UNIVERSAL_CONFORMANCE", timeout_ms=int(timeout_ms))
    conformance = prove_universal_conformance(
        model, source_root=source_root, timeout_ms=timeout_ms
    )
    receipts["universal_conformance"] = conformance.as_dict()
    _write(out / "proof_receipts.partial.json", receipts)
    if conformance.status != "PASS":
        for name in (
            "POLICY_TIMING_KERNEL_STEP_CONFORMANCE",
            "TIMING_PROJECTION_PREFIX_REFINEMENT",
            "FIRST_HI_BAD_PREFIX_REFLECTION",
        ):
            statuses[name] = "FAIL" if conformance.status == "FAIL" else "UNRESOLVED"
        summary = _fail_summary(
            request, statuses,
            code=conformance.failure_code or "V9_2_UNIVERSAL_CONFORMANCE_UNRESOLVED",
        )
        summary["binding_root_hash"] = recomputed["binding_root_hash"]
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary
    statuses.update({
        "POLICY_TIMING_KERNEL_STEP_CONFORMANCE": "PASS",
        "TIMING_PROJECTION_PREFIX_REFINEMENT": "PASS",
        "FIRST_HI_BAD_PREFIX_REFLECTION": "PASS",
    })

    # 2) P5 controller-summary soundness.  Safe-prefix induction does not need
    # observation/tree arithmetic once these compositional obligations prove that
    # every exact FirstValid update is contained in the bounded-budget/history
    # summary relation.  FirstBadWindow still uses the exact deployed P5 encoder.
    _write_progress(out, "P5_CONTROLLER_SUMMARY_SOUNDNESS", timeout_ms=int(timeout_ms))
    p5_summary_receipts: list[dict[str, Any]] = []
    for obligation in build_p5_summary_soundness_obligations(model, prefix="verify.p5.summary"):
        receipt = solve_formula(
            obligation.obligation_id, obligation.counterexample, timeout_ms=timeout_ms
        )
        row = receipt.as_dict()
        row["explanation"] = obligation.explanation
        p5_summary_receipts.append(row)
        if receipt.result != "UNSAT":
            statuses["P5_CONTROLLER_SUMMARY_SOUNDNESS"] = _receipt_status(receipt)
            receipts["p5_controller_summary_soundness"] = p5_summary_receipts
            summary = _fail_summary(
                request, statuses,
                code=f"P5_CONTROLLER_SUMMARY_{receipt.result}:{obligation.obligation_id}",
            )
            summary["binding_root_hash"] = recomputed["binding_root_hash"]
            summary["event_window_encoder_version"] = ENCODER_VERSION
            _write(out / "proof_receipts.json", receipts)
            _write(out / "proof_summary.json", summary)
            return summary
    receipts["p5_controller_summary_soundness"] = p5_summary_receipts
    statuses["P5_CONTROLLER_SUMMARY_SOUNDNESS"] = "PASS"
    _write(out / "proof_receipts.partial.json", receipts)

    # 3) Safe-prefix invariant: initial and conditional inductiveness.
    invariant = SafePrefixInvariant(model)
    _write_progress(out, "SAFE_PREFIX_INITIAL", timeout_ms=int(timeout_ms))
    initial = solve_formula(
        "SAFE_PREFIX_INVARIANT_INITIAL_COUNTEREXAMPLE",
        invariant.initial_counterexample(prefix="verify.initial"),
        timeout_ms=timeout_ms,
    )
    receipts["safe_prefix_initial"] = initial.as_dict()
    _write(out / "proof_receipts.partial.json", receipts)
    statuses["SAFE_PREFIX_INVARIANT_INITIAL"] = _receipt_status(initial)
    if initial.result != "UNSAT":
        diagnostics: list[dict[str, Any]] = []
        if initial.result == "SAT":
            for clause_name, formula in invariant.initial_clause_counterexamples(
                prefix="verify.initial.diag"
            ).items():
                clause_receipt = solve_formula(
                    f"SAFE_PREFIX_INITIAL_CLAUSE_{clause_name}",
                    formula, timeout_ms=timeout_ms, capture_model=True,
                )
                if clause_receipt.result == "SAT":
                    diagnostics.append({
                        "clause": clause_name,
                        **clause_receipt.as_dict(include_model=True),
                    })
        receipts["safe_prefix_initial_diagnostics"] = diagnostics
        summary = _fail_summary(
            request, statuses,
            code=f"SAFE_PREFIX_INITIAL_{initial.result}",
        )
        if diagnostics:
            summary["violated_invariant_clauses"] = [row["clause"] for row in diagnostics]
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary

    _write_progress(out, "SAFE_PREFIX_CONDITIONAL_INDUCTIVENESS", timeout_ms=int(timeout_ms))
    phase_receipts: list[dict[str, Any]] = []
    failed_phase: tuple[int, FormulaReceipt] | None = None
    for phase in range(8):
        ind_env = declare_environment(f"verify.ind.p{phase}.env", model, release_count=1)
        inductive = solve_formula(
            f"SAFE_PREFIX_INDUCTIVE_P{phase}",
            invariant.phase_inductiveness_counterexample(
                ind_env, phase, prefix="verify.ind",
                use_p5_summary=(phase == 5),
            ),
            timeout_ms=timeout_ms,
            capture_model=True,
        )
        phase_receipts.append(inductive.as_dict(include_model=inductive.result == "SAT"))
        statuses[f"SAFE_PREFIX_INDUCTIVE_P{phase}"] = _receipt_status(inductive)
        if inductive.result != "UNSAT":
            failed_phase = (phase, inductive)
            break
    receipts["safe_prefix_conditional_inductiveness_by_phase"] = phase_receipts
    _write(out / "proof_receipts.partial.json", receipts)
    if failed_phase is not None:
        phase, inductive = failed_phase
        statuses["SAFE_PREFIX_INVARIANT_CONDITIONAL_INDUCTIVENESS"] = _receipt_status(inductive)
        diagnostics: list[dict[str, Any]] = []
        if inductive.result == "SAT":
            diag_env = declare_environment(
                f"verify.ind.p{phase}.diag.env", model, release_count=1
            )
            for clause_name, formula in invariant.phase_inductiveness_clause_counterexamples(
                diag_env, phase, prefix="verify.ind.diag",
                use_p5_summary=(phase == 5),
            ).items():
                clause_receipt = solve_formula(
                    f"SAFE_PREFIX_INDUCTIVE_P{phase}_CLAUSE_{clause_name}",
                    formula, timeout_ms=timeout_ms, capture_model=True,
                )
                if clause_receipt.result == "SAT":
                    diagnostics.append({
                        "clause": clause_name,
                        **clause_receipt.as_dict(include_model=True),
                    })
        receipts["safe_prefix_inductiveness_diagnostics"] = diagnostics
        summary = _fail_summary(
            request, statuses,
            code=f"SAFE_PREFIX_INDUCTIVE_P{phase}_{inductive.result}",
        )
        summary["binding_root_hash"] = recomputed["binding_root_hash"]
        summary["event_window_encoder_version"] = ENCODER_VERSION
        if diagnostics:
            summary["violated_invariant_clauses"] = [row["clause"] for row in diagnostics]
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary
    statuses["SAFE_PREFIX_INVARIANT_CONDITIONAL_INDUCTIVENESS"] = "PASS"

    # 4) Finite two-slot carry-in adequacy meta-theorems.
    _write_progress(out, "CARRY_IN_ADEQUACY", timeout_ms=int(timeout_ms))
    carry_receipts: list[dict[str, Any]] = []
    for obligation in build_two_slot_carry_in_obligations(model, prefix="verify.carry"):
        receipt = solve_formula(
            obligation.obligation_id, obligation.counterexample, timeout_ms=timeout_ms
        )
        row = receipt.as_dict()
        row["explanation"] = obligation.explanation
        carry_receipts.append(row)
        if receipt.result != "UNSAT":
            statuses["CARRY_IN_SUMMARY_ADEQUACY"] = _receipt_status(receipt)
            receipts["carry_in"] = carry_receipts
            summary = _fail_summary(
                request, statuses, code=f"CARRY_IN_ADEQUACY_{receipt.result}:{obligation.obligation_id}"
            )
            _write(out / "proof_receipts.json", receipts)
            _write(out / "proof_summary.json", summary)
            return summary
    receipts["carry_in"] = carry_receipts
    statuses["CARRY_IN_SUMMARY_ADEQUACY"] = "PASS"
    _write(out / "proof_receipts.partial.json", receipts)

    # 5) V9.2 Event layer must be exact semantic compression.  The verifier
    # fresh-checks the exact-minimum/silent-interval/differential obligations
    # before any Event FirstBadWindow is accepted.
    _write_progress(out, "EVENT_REFINEMENT_EQUIVALENCE", timeout_ms=int(timeout_ms))
    event_refinement = prove_event_refinement(
        model, source_root=source_root, timeout_ms=timeout_ms
    )
    receipts["event_refinement"] = event_refinement.as_dict()
    statuses.update(event_refinement.obligation_statuses)
    _write(out / "proof_receipts.partial.json", receipts)
    if event_refinement.status != "PASS":
        summary = _fail_summary(
            request,
            statuses,
            code=event_refinement.failure_code or "V9_2_EVENT_REFINEMENT_UNRESOLVED",
        )
        summary["binding_root_hash"] = recomputed["binding_root_hash"]
        summary["event_window_encoder_version"] = ENCODER_VERSION
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary

    # 6) Per-HI exact Event FirstBadWindows, regenerated here.  Candidate-provided
    # SMT files are not opened or parsed anywhere in this route.  SAT windows
    # are classified immediately so a large encoding is never retained merely
    # to classify it after every other HI task has been solved.
    window_receipts: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    sat_tasks: list[str] = []
    unknown_tasks: list[str] = []
    if concrete_replayer is None:
        concrete_replayer = DeployedRuntimeCounterexampleReplayer(
            request["target_recipe"], model
        )
    symbols_per_full_state = _estimated_symbols_per_state(model)
    for task_index, task in enumerate(model.hi_tasks):
        event_bound = derive_finite_event_bound(model, task.name)
        event_boundary_count = event_bound.finite_event_bound + 1
        # Each potential active macro declares seven zero-time P0..P6 closure
        # states.  This is still tiny compared with D*8 microstep unrolling and
        # retains exact Full-state information to avoid new conservatism.
        controller_exact_instance_count = event_bound.controller_bound
        declared_full_state_upper = (
            event_boundary_count
            + event_bound.finite_event_bound * 7
            + controller_exact_instance_count * 2  # pooled exact-P5 pre/post
            + 3  # terminal P1/P2/P3
        )
        estimated_symbols = declared_full_state_upper * symbols_per_full_state
        _write_progress(
            out, "BUILD_FIRST_BAD_EVENT_WINDOW",
            task=task.name,
            task_index=task_index,
            hi_task_count=len(model.hi_tasks),
            deadline=int(task.deadline),
            finite_event_bound=event_bound.finite_event_bound,
            event_boundary_count=event_boundary_count,
            controller_exact_instance_count=controller_exact_instance_count,
            declared_full_state_upper=declared_full_state_upper,
            estimated_declared_state_symbols=estimated_symbols,
        )
        build_started = perf_counter()
        encoding = build_incremental_event_first_bad_window(model, invariant, task.name)
        build_seconds = perf_counter() - build_started
        _write_progress(
            out, "SOLVE_FIRST_BAD_EVENT_WINDOW_INCREMENTAL_DEPTH",
            task=task.name,
            task_index=task_index,
            hi_task_count=len(model.hi_tasks),
            deadline=int(task.deadline),
            finite_event_bound=event_bound.finite_event_bound,
            event_boundary_count=event_boundary_count,
            controller_exact_instance_count=controller_exact_instance_count,
            declared_full_state_upper=declared_full_state_upper,
            estimated_declared_state_symbols=estimated_symbols,
            build_seconds=round(build_seconds, 6),
            timeout_ms=int(timeout_ms),
            solver_strategy=SOLVER_STRATEGY,
        )
        def _incremental_progress(details: dict[str, Any]) -> None:
            _write_progress(
                out, "SOLVE_FIRST_BAD_EVENT_WINDOW_INCREMENTAL_DEPTH",
                task=task.name,
                task_index=task_index,
                hi_task_count=len(model.hi_tasks),
                deadline=int(task.deadline),
                finite_event_bound=event_bound.finite_event_bound,
                timeout_ms=int(timeout_ms),
                solver_strategy=SOLVER_STRATEGY,
                **details,
            )

        receipt, sat_encoding = solve_incremental_event_window(
            f"FIRST_BAD_EVENT_WINDOW_{task.name}",
            encoding,
            timeout_ms=timeout_ms,
            progress=_incremental_progress,
        )
        statuses[f"FIRST_BAD_EVENT_WINDOW_{task.name}"] = _receipt_status(receipt)
        row = receipt.as_dict()
        row.update({
            "task": task.name,
            "deadline": task.deadline,
            "finite_event_bound": event_bound.finite_event_bound,
            "event_boundary_count": event_boundary_count,
            "controller_exact_instance_count": controller_exact_instance_count,
            "declared_full_state_upper": declared_full_state_upper,
            "estimated_declared_state_symbols": estimated_symbols,
            "build_seconds": round(build_seconds, 6),
            "event_bound": event_bound.as_dict(),
            "source_obligations": list(encoding.source_obligations),
            "event_layer_added_abstractions": list(encoding.event_layer_added_abstractions),
            "exact_p5_in_event_window": encoding.exact_p5_in_event_window,
            "microstep_terminal_fallback_used": encoding.microstep_terminal_fallback_used,
            "incremental_terminal_depth_bmc": True,
            "terminal_stutter_slots_in_solver": 0,
        })
        window_receipts.append(row)
        receipts["hi_event_windows"] = window_receipts
        _write(out / "proof_receipts.partial.json", receipts)

        if receipt.result == "SAT":
            sat_tasks.append(task.name)
            if sat_encoding is None:
                unknown_tasks.append(task.name)
                classifications.append({
                    "status": "UNRESOLVED",
                    "code": "INCREMENTAL_SAT_DEPTH_MATERIALIZATION_MISSING",
                    "target_task": task.name,
                })
                receipts["sat_classification"] = classifications
                _write(out / "proof_receipts.partial.json", receipts)
                del encoding
                continue
            _write_progress(
                out, "CLASSIFY_SAT_EVENT_WINDOW",
                task=task.name,
                deadline=int(task.deadline),
                decisive_depth=int(receipt.decisive_depth) if receipt.decisive_depth is not None else None,
                max_boot_replay_ticks=int(max_boot_replay_ticks),
                timeout_ms=int(timeout_ms),
            )
            classification = classify_sat_event_window(
                sat_encoding, model, invariant,
                concrete_replayer=concrete_replayer,
                timeout_ms=timeout_ms,
                max_boot_ticks=max_boot_replay_ticks,
            )
            classification_row = classification.as_dict()
            classifications.append(classification_row)
            receipts["sat_classification"] = classifications
            _write(out / "proof_receipts.partial.json", receipts)
            if classification.status == "PASS":
                summary = _fail_summary(
                    request, statuses,
                    code="CONCRETE_HI_COUNTEREXAMPLE_VERIFIED",
                    message=str(classification.details.get("target_task")),
                    result=RESULT_CONCRETE_COUNTEREXAMPLE,
                )
                summary["binding_root_hash"] = recomputed["binding_root_hash"]
                summary["counterexample_receipt_hash"] = sha256_object(classifications)
                _write(out / "proof_receipts.json", receipts)
                _write(out / "proof_summary.json", summary)
                _write_progress(out, "COMPLETE", result_status=RESULT_CONCRETE_COUNTEREXAMPLE)
                return summary
        elif receipt.result != "UNSAT":
            unknown_tasks.append(task.name)

        if sat_encoding is not None:
            del sat_encoding
        del encoding

    receipts["hi_event_windows"] = window_receipts
    if classifications:
        receipts["sat_classification"] = classifications

    if unknown_tasks:
        statuses["ALL_HI_EVENT_WINDOWS_UNSAT"] = "UNRESOLVED"
        summary = _fail_summary(
            request, statuses, code="FIRST_BAD_EVENT_WINDOW_SOLVER_UNKNOWN",
            message=",".join(unknown_tasks),
        )
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        _write_progress(out, "COMPLETE", result_status=RESULT_UNRESOLVED,
                        unknown_tasks=unknown_tasks)
        return summary

    if sat_tasks:
        statuses["ALL_HI_EVENT_WINDOWS_UNSAT"] = "FAIL"
        summary = _fail_summary(
            request, statuses, code="SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
            message=",".join(sat_tasks),
        )
        summary["binding_root_hash"] = recomputed["binding_root_hash"]
        summary["counterexample_receipt_hash"] = sha256_object(classifications)
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        _write_progress(out, "COMPLETE", result_status=RESULT_UNRESOLVED, sat_tasks=sat_tasks)
        return summary

    statuses["ALL_HI_EVENT_WINDOWS_UNSAT"] = "PASS"
    summary = _proof_summary(
        request,
        statuses,
        binding_root_hash=recomputed["binding_root_hash"],
        receipts=receipts,
    )
    _write(out / "proof_receipts.json", receipts)
    _write(out / "proof_summary.json", summary)
    _write_progress(out, "COMPLETE", result_status=RESULT_PROVED)
    return summary
