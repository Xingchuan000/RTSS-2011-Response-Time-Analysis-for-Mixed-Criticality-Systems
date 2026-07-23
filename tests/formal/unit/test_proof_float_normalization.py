from __future__ import annotations

import json
from pathlib import Path

import pytest

from formal_toolchain.core.hashing import proof_safe_value, sha256_proof_object


def test_binary64_is_explicitly_normalized_before_proof_hash() -> None:
    value = {"ratio": 0.02, "nested": [1.0, 2]}
    normalized = proof_safe_value(value)
    assert normalized == {"ratio": "0.02", "nested": ["1", 2]}
    assert sha256_proof_object(value) == sha256_proof_object(normalized)


def test_nonfinite_float_is_rejected() -> None:
    with pytest.raises(ValueError):
        proof_safe_value({"bad": float("nan")})


def test_raw_seed_request_with_float_config_can_be_proof_hashed(tmp_path: Path) -> None:
    request = {
        "schema_version": "proof_request_v2",
        "target_recipe": {"kwargs": {"runtime_args": {"ratio": 0.02}}},
    }
    path = tmp_path / "proof_request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_proof_object({"request": loaded})
    assert len(digest) == 64
