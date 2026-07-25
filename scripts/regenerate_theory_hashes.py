from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_toolchain.core.hashing import sha256_file, sha256_json_file, sha256_object, sha256_text_file_normalized

STATEMENTS = ROOT / "formal_toolchain" / "theory" / "statements"
OUTPUT = ROOT / "formal_toolchain" / "theory" / "hashes.json"
THEORY_ROOT = ROOT / "formal_toolchain" / "theory"


def _refresh_source_bound_proof(backend_name: str, proof: dict[str, object]) -> bool:
    """Refresh source bindings using cross-platform canonical text hashes."""

    source_files: tuple[str, ...] | None = None
    if backend_name == "casewise-prefix-induction-v1":
        from formal_toolchain.theory.backends.casewise_prefix_induction import SOURCE_FILES
        source_files = tuple(SOURCE_FILES)
    elif backend_name in {
        "arrival-batch-decomposition-v1",
        "event-handler-decomposition-v1",
        "finite-release-fold-v1",
    }:
        from formal_toolchain.theory.backends.handler_decomposition import SOURCE_FILES
        source_files = tuple(SOURCE_FILES)

    if source_files is not None:
        proof["source_binding_hash_mode"] = "canonical_text_v1"
        proof["source_bindings"] = {
            name: sha256_text_file_normalized(ROOT / name)
            for name in source_files
        }
        return True

    if backend_name == "finite-hi-bad-prefix-z3-v1":
        from formal_toolchain.theory.backends.finite_hi_bad_prefix import current_n6_source_bindings
        proof["source_binding_hash_mode"] = "canonical_text_v1"
        proof["source_bindings"] = current_n6_source_bindings()
        return True

    if backend_name == "reference-prefix-extension-z3-v3":
        from formal_toolchain.theory.backends.reference_prefix_extension import current_prefix_extension_source_bindings
        proof["source_binding_hash_mode"] = "canonical_text_v1"
        proof["source_bindings"] = current_prefix_extension_source_bindings()
        return True

    return False


def main() -> None:
    result: dict[str, dict[str, object]] = {}
    for path in sorted(STATEMENTS.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        theorem_id = str(row["theorem_id"])

        proof_object = row.get("proof_object")
        if isinstance(proof_object, dict):
            proof_path = (THEORY_ROOT / str(proof_object["path"])).resolve(strict=True)
            if THEORY_ROOT.resolve() not in proof_path.parents:
                raise ValueError(f"PROOF_OBJECT_ESCAPES_THEORY_ROOT:{theorem_id}")
            if proof_path.suffix.lower() == ".json":
                proof_payload = json.loads(proof_path.read_text(encoding="utf-8"))
                refreshed = _refresh_source_bound_proof(
                    str(proof_object.get("backend", "")), proof_payload
                )
                if refreshed:
                    proof_path.write_text(
                        json.dumps(proof_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                proof_object["hash_mode"] = "canonical_json_v1"
                proof_object["sha256"] = sha256_json_file(proof_path)
            else:
                proof_object["hash_mode"] = "raw_bytes_v1"
                proof_object["sha256"] = sha256_file(proof_path)

        row["statement_hash"] = sha256_object({
            "theorem_id": theorem_id,
            "exact_statement": row["exact_statement"],
            "conclusion": row["conclusion"],
            "source_reference": row["source_reference"],
            "assurance_level": row["assurance_level"],
            "version": row["version"],
        })
        row["assumption_hash"] = sha256_object({
            "theorem_id": theorem_id,
            "assumptions": row.get("assumptions", []),
            "premise_obligation_ids": row.get("premise_obligation_ids", []),
            "version": row["version"],
        })
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result[theorem_id] = {
            "statement_hash": row["statement_hash"],
            "assumption_hash": row["assumption_hash"],
        }

    OUTPUT.write_text(
        json.dumps(
            {
                "schema_version": "theory_hashes_v3",
                "statements": result,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
