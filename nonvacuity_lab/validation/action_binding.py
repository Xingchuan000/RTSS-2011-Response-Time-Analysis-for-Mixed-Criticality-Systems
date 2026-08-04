from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

from ..canonical import file_hash


def verify_compiled_action_binding(bundle_dir: Path, expected_ratio: Fraction) -> dict:
    path = Path(bundle_dir) / "proof_context" / "action_semantics.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    actual = Fraction(str(frozen["increase_ratio"]))
    if actual != expected_ratio:
        raise ValueError(f"compiled ratio {actual} != expected {expected_ratio}")
    return {"expected_ratio": str(expected_ratio), "compiled_ratio": str(actual), "action_semantics_sha256": file_hash(path)}
