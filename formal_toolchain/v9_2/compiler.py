"""Untrusted candidate-bundle compiler for the single V9.2 route."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.v9_2.bindings import build_bindings, load_request
from formal_toolchain.v9_2.constants import PROOF_ROUTE, SCOPE
from formal_toolchain.v9_2.encoding_contract import (
    EVENT_WINDOW_ENCODER_IMPLEMENTED,
    EVENT_WINDOW_ENCODER_VERSION,
    REQUIRED_SOUNDNESS_CLAUSES,
)
from formal_toolchain.v9_2.readiness import blocker_rows


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compile_request_v9_2(request_path: Path, out: Path, *, source_root: Path) -> dict[str, Any]:
    request_path = Path(request_path).resolve()
    out = Path(out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    request = load_request(request_path)
    bindings = build_bindings(request_path, source_root=source_root)
    _write(out / "bindings.json", bindings)

    gaps = blocker_rows() if not EVENT_WINDOW_ENCODER_IMPLEMENTED else []
    manifest = {
        "schema_version": "v9_2_candidate_bundle_v1",
        "proof_route": PROOF_ROUTE,
        "scope": SCOPE,
        "target_id": request["target_id"],
        "taskset_seed": request["taskset_seed"],
        "binding_root_hash": bindings["binding_root_hash"],
        # Candidate bundles transport identity only.  Trusted formulas are
        # rebuilt from request + source by the verifier.
        "candidate_assertions_trusted": False,
        "event_window_encoder": {
            "version": EVENT_WINDOW_ENCODER_VERSION,
            "implemented": EVENT_WINDOW_ENCODER_IMPLEMENTED,
            "required_soundness_clauses": list(REQUIRED_SOUNDNESS_CLAUSES),
            "event_layer_added_abstractions": [],
            "exact_p5_in_event_window": True,
        },
        "implementation_gaps": gaps,
    }
    _write(out / "candidate_manifest.json", manifest)
    return {
        "workflow_status": "CANDIDATE_COMPILED",
        "proof_route": PROOF_ROUTE,
        "binding_root_hash": bindings["binding_root_hash"],
        "implementation_gap_count": len(gaps),
    }


__all__ = ["compile_request_v9_2"]
