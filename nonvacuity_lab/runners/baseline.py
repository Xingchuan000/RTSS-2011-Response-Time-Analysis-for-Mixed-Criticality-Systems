from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..subprocess_runner import ordinary_prove_command, run_command
from ..validation.proof_request_clean import assert_experiment_blind_request, IsolationError


def run_baseline(
    *,
    seed_dir: Path,
    tree_variant: str,
    source_root: Path,
    output_dir: Path,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    proof_dir = output_dir / "proof"
    receipt = run_command(
        ordinary_prove_command(
            seed_dir=seed_dir,
            tree_variant=tree_variant,
            source_root=source_root,
            output_dir=proof_dir,
        ),
        cwd=source_root,
        log_dir=output_dir / "logs",
        timeout_seconds=timeout_seconds,
    )
    result = _read_proof_result(proof_dir)
    request = proof_dir / "request" / "proof_request.json"
    request_guard = _verify_request_is_blind(request)
    return {
        "schema_version": "baseline_run_v1",
        "receipt": receipt,
        "proof_result": result,
        "request_guard": request_guard,
        "passed": result.get("result_status") == "DEPLOYED_TREE_PROVED" and request_guard["status"] == "PASS",
    }


def _read_proof_result(proof_dir: Path) -> dict[str, Any]:
    path = proof_dir / "proof_result.json"
    if not path.is_file():
        return {
            "workflow_status": "FAILED",
            "result_status": "PROOF_BUNDLE_INVALID",
            "failure_code": "PROOF_RESULT_MISSING",
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def _verify_request_is_blind(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "FAIL", "reason": "proof request missing"}
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        assert_experiment_blind_request(raw)
    except IsolationError as exc:
        return {"status": "FAIL", "forbidden_fields": [str(exc)]}
    return {
        "status": "PASS",
        "forbidden_fields": [],
    }
