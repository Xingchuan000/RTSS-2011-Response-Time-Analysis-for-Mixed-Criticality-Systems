"""Trusted, verifier-regenerated V9.1 end-to-end proof checker.

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
from formal_toolchain.v9_1.bindings import build_bindings, load_request
from formal_toolchain.v9_1.carry_in import build_two_slot_carry_in_obligations
from formal_toolchain.v9_1.counterexample_replay import (
    DeployedRuntimeCounterexampleReplayer, classify_sat_window,
)
from formal_toolchain.v9_1.constants import (
    PROOF_ROUTE, RESULT_CONCRETE_COUNTEREXAMPLE, RESULT_INVALID, RESULT_PROVED,
    RESULT_UNRESOLVED, SCOPE,
)
from formal_toolchain.v9_1.environment_encoder import declare_environment
from formal_toolchain.v9_1.formula_solver import FormulaReceipt, solve_formula
from formal_toolchain.v9_1.readiness import blocker_rows, proof_pipeline_ready
from formal_toolchain.v9_1.safe_prefix_invariant import SafePrefixInvariant
from formal_toolchain.v9_1.symbolic_state import BoundModel
from formal_toolchain.v9_1.universal_conformance import prove_universal_conformance
from formal_toolchain.v9_1.window_encoder import ENCODER_VERSION, build_first_bad_window


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")




def _write_progress(out: Path, stage: str, **details: Any) -> None:
    _write(out / "verifier_progress.json", {
        "schema_version": "v9_1_verifier_progress_v1",
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
        "schema_version": "v9_1_verified_summary_v2",
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
        "schema_version": "v9_1_verified_summary_v2",
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
        "window_encoder_version": ENCODER_VERSION,
        "obligation_statuses": statuses,
        "proof_receipt_hash": sha256_object(receipts),
    }


def verify_bundle_v9_1(
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
        "schema_version": "v9_1_fresh_proof_receipts_v1",
        "proof_route": PROOF_ROUTE,
        "timeout_ms": int(timeout_ms),
        "candidate_assertions_trusted": False,
    }

    candidate_path = bundle / "candidate_manifest.json"
    bindings_path = bundle / "bindings.json"
    if not candidate_path.is_file() or not bindings_path.is_file():
        summary = _fail_summary(
            request, statuses, code="V9_1_CANDIDATE_BUNDLE_INCOMPLETE", result=RESULT_INVALID
        )
        _write(out / "proof_summary.json", summary)
        return summary
    candidate = _read_json(candidate_path)
    candidate_bindings = _read_json(bindings_path)
    if candidate.get("proof_route") != PROOF_ROUTE or not _candidate_integrity(candidate):
        summary = _fail_summary(
            request, statuses, code="V9_1_CANDIDATE_MANIFEST_INTEGRITY_FAILED", result=RESULT_INVALID
        )
        _write(out / "proof_summary.json", summary)
        return summary

    try:
        recomputed = build_bindings(request_path, source_root=source_root)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        summary = _fail_summary(
            request, statuses, code="V9_1_BINDING_REGENERATION_FAILED",
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
            code="WINDOW_ENCODING_UNRESOLVED",
            message="V9.1 end-to-end proof implementation still has explicit readiness blockers.",
        )
        summary["binding_root_hash"] = recomputed["binding_root_hash"]
        summary["window_encoder_version"] = ENCODER_VERSION
        summary["implementation_gaps"] = blocker_rows()
        _write(out / "proof_summary.json", summary)
        return summary

    try:
        model = BoundModel.from_bindings(recomputed, max_jobs_per_task=2)
    except (ValueError, TypeError, KeyError) as exc:
        summary = _fail_summary(
            request, statuses, code="V9_1_BOUND_MODEL_REGENERATION_FAILED", message=str(exc)
        )
        _write(out / "proof_summary.json", summary)
        return summary
    if any(task.deadline > task.period for task in model.tasks):
        summary = _fail_summary(
            request, statuses, code="V9_1_TWO_SLOT_CARRY_IN_REQUIRES_D_LE_T"
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
            code=conformance.failure_code or "V9_1_UNIVERSAL_CONFORMANCE_UNRESOLVED",
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

    # 2) Safe-prefix invariant: initial and conditional inductiveness.
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
                ind_env, phase, prefix="verify.ind"
            ),
            timeout_ms=timeout_ms,
        )
        phase_receipts.append(inductive.as_dict())
        statuses[f"SAFE_PREFIX_INDUCTIVE_P{phase}"] = _receipt_status(inductive)
        if inductive.result != "UNSAT":
            failed_phase = (phase, inductive)
            break
    receipts["safe_prefix_conditional_inductiveness_by_phase"] = phase_receipts
    _write(out / "proof_receipts.partial.json", receipts)
    if failed_phase is not None:
        phase, inductive = failed_phase
        statuses["SAFE_PREFIX_INVARIANT_CONDITIONAL_INDUCTIVENESS"] = _receipt_status(inductive)
        summary = _fail_summary(
            request, statuses,
            code=f"SAFE_PREFIX_INDUCTIVE_P{phase}_{inductive.result}",
        )
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary
    statuses["SAFE_PREFIX_INVARIANT_CONDITIONAL_INDUCTIVENESS"] = "PASS"

    # 3) Finite two-slot carry-in adequacy meta-theorems.
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
            statuses["FINITE_WINDOW_ENCODING_SOUNDNESS"] = _receipt_status(receipt)
            receipts["carry_in"] = carry_receipts
            summary = _fail_summary(
                request, statuses, code=f"CARRY_IN_ADEQUACY_{receipt.result}:{obligation.obligation_id}"
            )
            _write(out / "proof_receipts.json", receipts)
            _write(out / "proof_summary.json", summary)
            return summary
    receipts["carry_in"] = carry_receipts
    _write(out / "proof_receipts.partial.json", receipts)

    # 4) Per-HI first-bad finite windows, regenerated here.  Candidate-provided
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
    symbols_per_state = _estimated_symbols_per_state(model)
    for task_index, task in enumerate(model.hi_tasks):
        state_count = int(task.deadline) * 8 + 4
        _write_progress(
            out, "BUILD_FIRST_BAD_WINDOW",
            task=task.name, task_index=task_index, hi_task_count=len(model.hi_tasks),
            deadline=int(task.deadline), state_count=state_count,
            estimated_declared_state_symbols=state_count * symbols_per_state,
        )
        build_started = perf_counter()
        encoding = build_first_bad_window(model, invariant, task.name)
        build_seconds = perf_counter() - build_started
        _write_progress(
            out, "SOLVE_FIRST_BAD_WINDOW",
            task=task.name, task_index=task_index, hi_task_count=len(model.hi_tasks),
            deadline=int(task.deadline), state_count=state_count,
            estimated_declared_state_symbols=state_count * symbols_per_state,
            build_seconds=round(build_seconds, 6), timeout_ms=int(timeout_ms),
        )
        receipt = solve_formula(
            f"FIRST_BAD_WINDOW_{task.name}",
            encoding.formula,
            timeout_ms=timeout_ms,
            capture_model=False,
        )
        row = receipt.as_dict()
        row.update({
            "task": task.name,
            "deadline": task.deadline,
            "state_count": state_count,
            "estimated_declared_state_symbols": state_count * symbols_per_state,
            "build_seconds": round(build_seconds, 6),
            "source_obligations": list(encoding.source_obligations),
        })
        window_receipts.append(row)
        receipts["hi_windows"] = window_receipts
        _write(out / "proof_receipts.partial.json", receipts)

        if receipt.result == "SAT":
            sat_tasks.append(task.name)
            _write_progress(
                out, "CLASSIFY_SAT_WINDOW", task=task.name, deadline=int(task.deadline),
                max_boot_replay_ticks=int(max_boot_replay_ticks), timeout_ms=int(timeout_ms),
            )
            classification = classify_sat_window(
                encoding, model, invariant,
                concrete_replayer=concrete_replayer,
                timeout_ms=timeout_ms,
                max_boot_ticks=max_boot_replay_ticks,
            )
            classification_row = classification.as_dict()
            classifications.append(classification_row)
            receipts["sat_classification"] = classifications
            _write(out / "proof_receipts.partial.json", receipts)
            if classification.status == "PASS":
                statuses["FINITE_WINDOW_ENCODING_SOUNDNESS"] = "PASS"
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

        # ``encoding`` can hold hundreds of thousands of symbolic states for a
        # long deadline.  Do not retain it after its receipt/SAT classification.
        del encoding

    receipts["hi_windows"] = window_receipts
    if classifications:
        receipts["sat_classification"] = classifications

    if unknown_tasks:
        statuses["FINITE_WINDOW_ENCODING_SOUNDNESS"] = "UNRESOLVED"
        summary = _fail_summary(
            request, statuses, code="FIRST_BAD_WINDOW_SOLVER_UNKNOWN",
            message=",".join(unknown_tasks),
        )
        _write(out / "proof_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        _write_progress(out, "COMPLETE", result_status=RESULT_UNRESOLVED,
                        unknown_tasks=unknown_tasks)
        return summary

    if sat_tasks:
        statuses["FINITE_WINDOW_ENCODING_SOUNDNESS"] = "PASS"
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

    statuses["FINITE_WINDOW_ENCODING_SOUNDNESS"] = "PASS"
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
