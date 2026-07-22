from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file, sha256_object


def load_verified_theory_statement(theory_dir: Path, theorem_id: str) -> dict[str, Any]:
    manifest = json.loads(
        (theory_dir / "theory_manifest.json").read_text(encoding="utf-8")
    )
    if theorem_id not in set(manifest.get("required_theorems", [])):
        raise ValueError(f"theorem is not required by current library: {theorem_id}")

    statement_path = theory_dir / "statements" / f"{theorem_id}.json"
    statement = json.loads(statement_path.read_text(encoding="utf-8"))
    if statement.get("theorem_id") != theorem_id:
        raise ValueError("theory statement id mismatch")

    statement_payload = {
        key: statement[key]
        for key in (
            "theorem_id",
            "exact_statement",
            "conclusion",
            "source_reference",
            "assurance_level",
            "version",
        )
    }
    assumption_payload = {
        "theorem_id": statement["theorem_id"],
        "assumptions": statement["assumptions"],
        "version": statement["version"],
    }
    if statement["statement_hash"] != sha256_object(statement_payload):
        raise ValueError("theory statement hash mismatch")
    if statement["assumption_hash"] != sha256_object(assumption_payload):
        raise ValueError("theory assumption hash mismatch")

    declared = json.loads(
        (theory_dir / "hashes.json").read_text(encoding="utf-8")
    )["statements"].get(theorem_id)
    if declared != {
        "statement_hash": statement["statement_hash"],
        "assumption_hash": statement["assumption_hash"],
    }:
        raise ValueError("theory hashes.json mismatch")

    proof_object = statement.get("proof_object")
    if proof_object:
        proof_path = theory_dir / proof_object["path"]
        if sha256_file(proof_path) != proof_object["sha256"]:
            raise ValueError("theory proof object hash mismatch")

    return statement
