"""Untrusted candidate-bundle compiler for the V9.1 route."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.v9_1.bindings import build_bindings, load_request
from formal_toolchain.v9_1.constants import PROOF_ROUTE, SCOPE
from formal_toolchain.v9_1.encoding_contract import (
    REQUIRED_SOUNDNESS_CLAUSES, WINDOW_ENCODER_IMPLEMENTED, WINDOW_ENCODER_VERSION,
)
from formal_toolchain.v9_1.proof_objects import copy_proof_inputs


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compile_request_v9_1(request_path: Path, out: Path, *, source_root: Path) -> dict[str, Any]:
    request_path = Path(request_path).resolve()
    out = Path(out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    request = load_request(request_path)
    bindings = build_bindings(request_path, source_root=source_root)
    _write(out / "bindings.json", bindings)

    workspace_root = request_path.parent.parent
    formal_inputs_dir = (workspace_root / request["formal_inputs_dir"]).resolve()
    proof_inputs = copy_proof_inputs(formal_inputs_dir, out)
    gaps = []
    if not WINDOW_ENCODER_IMPLEMENTED:
        gaps.append({
            "code": "WINDOW_ENCODING_UNRESOLVED",
            "component": "formal_toolchain.v9_1 symbolic Policy-Timing Kernel / first-miss window encoder",
            "required_encoder_version": WINDOW_ENCODER_VERSION,
        })
    if proof_inputs.get("status") != "COPIED":
        gaps.append({"code": proof_inputs.get("failure_code", "V9_1_PROOF_INPUTS_MISSING"),
                     "component": "safe-prefix/kernel/window proof objects"})

    manifest = {
        "schema_version": "v9_1_candidate_bundle_v1",
        "proof_route": PROOF_ROUTE,
        "scope": SCOPE,
        "target_id": request["target_id"],
        "taskset_seed": request["taskset_seed"],
        "binding_root_hash": bindings["binding_root_hash"],
        "proof_inputs": proof_inputs,
        "window_encoder": {
            "version": WINDOW_ENCODER_VERSION,
            "implemented": WINDOW_ENCODER_IMPLEMENTED,
            "required_soundness_clauses": list(REQUIRED_SOUNDNESS_CLAUSES),
        },
        "implementation_gaps": gaps,
    }
    manifest["candidate_root_hash"] = sha256_object(manifest)
    _write(out / "candidate_manifest.json", manifest)
    return {
        "workflow_status": "CANDIDATE_COMPILED",
        "proof_route": PROOF_ROUTE,
        "binding_root_hash": bindings["binding_root_hash"],
        "candidate_root_hash": manifest["candidate_root_hash"],
        "implementation_gap_count": len(gaps),
    }
