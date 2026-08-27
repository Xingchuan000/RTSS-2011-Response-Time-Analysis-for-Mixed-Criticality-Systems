"""Fresh-process V9.1 verifier.

It has no V8 route selection and no RTA terminal.  The only success status is
DEPLOYED_TREE_PROVED_P0, which requires every V9.1 obligation to be discharged.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.v9_1.bindings import build_bindings, load_request
from formal_toolchain.v9_1.constants import (
    PROOF_ROUTE, RESULT_CONCRETE_COUNTEREXAMPLE, RESULT_INVALID, RESULT_PROVED,
    RESULT_UNRESOLVED, SCOPE,
)
from formal_toolchain.v9_1.encoding_contract import (
    REQUIRED_SOUNDNESS_CLAUSES, WINDOW_ENCODER_IMPLEMENTED, WINDOW_ENCODER_VERSION,
)
from formal_toolchain.v9_1.proof_objects import REQUIRED_CORE_SMT, load_proof_manifest
from formal_toolchain.v9_1.smt_receipts import replay_unsat


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fail_summary(request: dict[str, Any], statuses: dict[str, str], *, code: str,
                  message: str | None = None, result: str = RESULT_UNRESOLVED) -> dict[str, Any]:
    return {
        "schema_version": "v9_1_verified_summary_v1",
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


def _check_soundness_manifest(manifest: dict[str, Any], binding_root_hash: str) -> tuple[bool, str | None]:
    if manifest.get("binding_root_hash") != binding_root_hash:
        return False, "PROOF_OBJECT_BINDING_ROOT_MISMATCH"
    if manifest.get("window_encoder_version") != WINDOW_ENCODER_VERSION:
        return False, "WINDOW_ENCODER_VERSION_MISMATCH"
    coverage = manifest.get("finite_window_soundness_clauses")
    if not isinstance(coverage, dict):
        return False, "FINITE_WINDOW_SOUNDNESS_COVERAGE_MISSING"
    missing = [name for name in REQUIRED_SOUNDNESS_CLAUSES if coverage.get(name) is not True]
    if missing:
        return False, "FINITE_WINDOW_ENCODING_SOUNDNESS_INCOMPLETE"
    return True, None


def _classify_sat_window(proof_dir: Path, task: str) -> tuple[str, str]:
    witness_path = proof_dir / f"counterexample_{task}.json"
    if not witness_path.is_file():
        return RESULT_UNRESOLVED, "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE"
    witness = _read_json(witness_path)
    if not isinstance(witness, dict):
        return RESULT_INVALID, "COUNTEREXAMPLE_WITNESS_INVALID"
    # A JSON assertion is not an independent replay.  Until a source-level replay
    # checker is implemented, never upgrade SAT to a concrete unsafe verdict.
    return RESULT_UNRESOLVED, "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE"


def verify_bundle_v9_1(request_path: Path, bundle: Path, out: Path, *, source_root: Path) -> dict[str, Any]:
    request_path = Path(request_path).resolve()
    bundle = Path(bundle).resolve()
    out = Path(out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    request = load_request(request_path)
    statuses: dict[str, str] = {}

    candidate_path = bundle / "candidate_manifest.json"
    bindings_path = bundle / "bindings.json"
    if not candidate_path.is_file() or not bindings_path.is_file():
        summary = _fail_summary(request, statuses, code="V9_1_CANDIDATE_BUNDLE_INCOMPLETE", result=RESULT_INVALID)
        _write(out / "proof_summary.json", summary)
        return summary
    candidate = _read_json(candidate_path)
    candidate_bindings = _read_json(bindings_path)
    recomputed = build_bindings(request_path, source_root=source_root)
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

    if not WINDOW_ENCODER_IMPLEMENTED:
        statuses.update({
            "POLICY_TIMING_KERNEL_STEP_CONFORMANCE": "UNRESOLVED",
            "TIMING_PROJECTION_PREFIX_REFINEMENT": "UNRESOLVED",
            "FIRST_HI_BAD_PREFIX_REFLECTION": "UNRESOLVED",
            "SAFE_PREFIX_INVARIANT_INITIAL": "UNRESOLVED",
            "SAFE_PREFIX_INVARIANT_CONDITIONAL_INDUCTIVENESS": "UNRESOLVED",
            "FINITE_WINDOW_ENCODING_SOUNDNESS": "UNRESOLVED",
        })
        summary = _fail_summary(
            request, statuses, code="WINDOW_ENCODING_UNRESOLVED",
            message="V9.1 symbolic kernel/window encoder is not implemented in this patch set; refusing to infer safety.")
        summary["binding_root_hash"] = recomputed["binding_root_hash"]
        summary["window_encoder_version"] = WINDOW_ENCODER_VERSION
        _write(out / "proof_summary.json", summary)
        return summary

    proof_dir = bundle / "proof_inputs"
    if not proof_dir.is_dir():
        summary = _fail_summary(request, statuses, code="V9_1_PROOF_INPUTS_MISSING")
        _write(out / "proof_summary.json", summary)
        return summary
    try:
        proof_manifest = load_proof_manifest(proof_dir)
    except ValueError as exc:
        summary = _fail_summary(request, statuses, code="V9_1_PROOF_MANIFEST_INVALID", message=str(exc), result=RESULT_INVALID)
        _write(out / "proof_summary.json", summary)
        return summary
    sound, sound_code = _check_soundness_manifest(proof_manifest, recomputed["binding_root_hash"])
    if not sound:
        statuses["FINITE_WINDOW_ENCODING_SOUNDNESS"] = "FAIL"
        summary = _fail_summary(request, statuses, code=str(sound_code), result=RESULT_INVALID)
        _write(out / "proof_summary.json", summary)
        return summary
    statuses["FINITE_WINDOW_ENCODING_SOUNDNESS"] = "PASS"

    core_map = {
        "kernel_step_conformance": "POLICY_TIMING_KERNEL_STEP_CONFORMANCE",
        "prefix_refinement": "TIMING_PROJECTION_PREFIX_REFINEMENT",
        "first_hi_bad_prefix_reflection": "FIRST_HI_BAD_PREFIX_REFLECTION",
        "safe_prefix_initial": "SAFE_PREFIX_INVARIANT_INITIAL",
        "safe_prefix_conditional_inductiveness": "SAFE_PREFIX_INVARIANT_CONDITIONAL_INDUCTIVENESS",
    }
    receipts: dict[str, Any] = {}
    for key, filename in REQUIRED_CORE_SMT.items():
        receipt = replay_unsat(proof_dir / filename)
        receipts[key] = receipt
        obligation = core_map[key]
        if receipt["status"] == "UNSAT":
            statuses[obligation] = "PASS"
            continue
        statuses[obligation] = receipt["status"]
        if key == "safe_prefix_conditional_inductiveness" and receipt["status"] == "SAT":
            code = "SAFE_PREFIX_INVARIANT_NOT_INDUCTIVE"
        elif receipt["status"] == "UNKNOWN":
            code = "SOLVER_UNKNOWN_OR_TIMEOUT"
        else:
            code = f"{obligation}_FAILED"
        summary = _fail_summary(request, statuses, code=code)
        summary["solver_receipts"] = receipts
        _write(out / "solver_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary

    hi_rows = [row for row in recomputed["taskset"]["ordered_tasks"] if row["criticality"] == "HI"]
    expected = {row["name"]: int(row["deadline"]) for row in hi_rows}
    declared = {row["task"]: row for row in proof_manifest["hi_windows"]}
    if set(declared) != set(expected):
        summary = _fail_summary(request, statuses, code="HI_WINDOW_TASK_SET_MISMATCH", result=RESULT_INVALID)
        _write(out / "proof_summary.json", summary)
        return summary
    for task, deadline in expected.items():
        row = declared[task]
        if int(row["deadline"]) != deadline:
            summary = _fail_summary(request, statuses, code="HI_WINDOW_DEADLINE_MISMATCH", result=RESULT_INVALID)
            _write(out / "proof_summary.json", summary)
            return summary
        receipt = replay_unsat(proof_dir / row["smt2"])
        receipts[f"FirstBadWindow::{task}"] = receipt
        obligation = f"FIRST_HI_MISS_WINDOW_UNSAT::{task}"
        if receipt["status"] == "UNSAT":
            statuses[obligation] = "PASS"
            continue
        statuses[obligation] = receipt["status"]
        if receipt["status"] == "SAT":
            result, code = _classify_sat_window(proof_dir, task)
        elif receipt["status"] == "UNKNOWN":
            result, code = RESULT_UNRESOLVED, "SOLVER_UNKNOWN_OR_TIMEOUT"
        else:
            result, code = RESULT_INVALID, "WINDOW_SMT_INVALID"
        summary = _fail_summary(request, statuses, code=code, result=result)
        summary["solver_receipts"] = receipts
        _write(out / "solver_receipts.json", receipts)
        _write(out / "proof_summary.json", summary)
        return summary

    statuses["POLICY_TIMING_KERNEL_HI_SAFETY_P0"] = "PASS"
    statuses["DEPLOYED_HI_SAFETY_P0"] = "PASS"
    summary = {
        "schema_version": "v9_1_verified_summary_v1",
        "workflow_status": "PROVED",
        "result_status": RESULT_PROVED,
        "proof_route": PROOF_ROUTE,
        "scope": SCOPE,
        "primary_claim": request["primary_claim"],
        "target_id": request["target_id"],
        "target_kind": request["target_kind"],
        "taskset_seed": request["taskset_seed"],
        "tree_variant": request["tree_variant"],
        "binding_root_hash": recomputed["binding_root_hash"],
        "obligation_statuses": statuses,
        "solver_receipts_hash": sha256_object(receipts),
        "certificate": {
            "proof_route": PROOF_ROUTE,
            "scope": SCOPE,
            "result": RESULT_PROVED,
        },
    }
    _write(out / "solver_receipts.json", receipts)
    _write(out / "proof_summary.json", summary)
    return summary
