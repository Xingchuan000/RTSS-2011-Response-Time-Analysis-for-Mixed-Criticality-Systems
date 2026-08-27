"""Strict V9.1 proof-object interface.

The compiler may copy proof-producing artifacts, but only the verifier decides
whether their machine-checkable obligations are sufficient.  No status flag from
the candidate bundle is trusted.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file, sha256_object

PROOF_INPUT_DIRNAME = "v9_1_proofs"
PROOF_MANIFEST = "proof_manifest.json"
REQUIRED_CORE_SMT = {
    "kernel_step_conformance": "kernel_step_conformance.smt2",
    "prefix_refinement": "prefix_refinement.smt2",
    "first_hi_bad_prefix_reflection": "first_hi_bad_prefix_reflection.smt2",
    "safe_prefix_initial": "safe_prefix_initial.smt2",
    "safe_prefix_conditional_inductiveness": "safe_prefix_conditional_inductiveness.smt2",
}


def copy_proof_inputs(formal_inputs_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    source = Path(formal_inputs_dir) / PROOF_INPUT_DIRNAME
    destination = Path(candidate_dir) / "proof_inputs"
    if destination.exists():
        shutil.rmtree(destination)
    if not source.is_dir():
        return {"status": "MISSING", "failure_code": "V9_1_PROOF_INPUTS_MISSING"}
    shutil.copytree(source, destination, symlinks=False)
    files = {
        str(path.relative_to(destination)): sha256_file(path)
        for path in sorted(destination.rglob("*")) if path.is_file()
    }
    return {"status": "COPIED", "files": files, "content_hash": sha256_object(files)}


def load_proof_manifest(proof_dir: Path) -> dict[str, Any]:
    path = Path(proof_dir) / PROOF_MANIFEST
    if not path.is_file():
        raise ValueError("V9.1 proof_manifest.json missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "v9_1_proof_manifest_v1":
        raise ValueError("invalid V9.1 proof manifest schema")
    if data.get("safe_prefix_inductiveness_condition") != "NOT NewHIMiss(z,z')":
        raise ValueError("safe-prefix inductiveness must be conditional on NOT NewHIMiss")
    if data.get("carry_in_summary") not in {"NONE", "EXPLICIT"}:
        raise ValueError("carry_in_summary must be NONE or EXPLICIT")
    windows = data.get("hi_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("V9.1 proof manifest must contain per-HI window entries")
    seen: set[str] = set()
    for row in windows:
        if not isinstance(row, dict) or not isinstance(row.get("task"), str):
            raise ValueError("invalid HI window row")
        if row["task"] in seen:
            raise ValueError("duplicate HI window task")
        seen.add(row["task"])
        if not isinstance(row.get("deadline"), int) or row["deadline"] <= 0:
            raise ValueError("HI window deadline must be a positive integer")
        if not isinstance(row.get("smt2"), str):
            raise ValueError("HI window smt2 path missing")
    return data
