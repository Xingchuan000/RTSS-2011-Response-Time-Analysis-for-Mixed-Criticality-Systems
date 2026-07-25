from __future__ import annotations

import json
from pathlib import Path

from formal_toolchain.core.hashing import sha256_file, sha256_file_by_mode
from formal_toolchain.theory.loader import verify_theory_library_for_route


def test_canonical_json_hash_ignores_windows_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    text = '{\n  "b": [2, 3],\n  "a": 1\n}\n'
    lf.write_bytes(text.encode("utf-8"))
    crlf.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))

    assert sha256_file(lf) != sha256_file(crlf)
    assert sha256_file_by_mode(lf, "canonical_json_v1") == sha256_file_by_mode(
        crlf, "canonical_json_v1"
    )


def test_json_theory_proof_objects_use_canonical_hash_mode() -> None:
    theory_root = Path(__file__).resolve().parents[3] / "formal_toolchain" / "theory"
    for statement_path in sorted((theory_root / "statements").glob("*.json")):
        statement = json.loads(statement_path.read_text(encoding="utf-8"))
        proof_object = statement.get("proof_object")
        if not isinstance(proof_object, dict):
            continue
        proof_path = theory_root / proof_object["path"]
        expected_mode = (
            "canonical_json_v1" if proof_path.suffix.lower() == ".json" else "raw_bytes_v1"
        )
        assert proof_object.get("hash_mode") == expected_mode
        assert proof_object["sha256"] == sha256_file_by_mode(proof_path, expected_mode)


def test_protected_prefix_theory_library_still_verifies() -> None:
    theory_root = Path(__file__).resolve().parents[3] / "formal_toolchain" / "theory"
    result = verify_theory_library_for_route(theory_root, "protected_prefix")
    assert result["status"] == "PASS", result
