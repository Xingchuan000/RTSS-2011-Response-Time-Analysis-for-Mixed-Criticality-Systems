"""Stable observation schema fingerprinting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence


def observation_schema_fingerprint(
    *,
    observation_mode: str,
    feature_names: Sequence[str],
) -> str:
    payload = {
        "schema_version": "observation_schema_v1",
        "observation_mode": observation_mode,
        "observation_dim": len(feature_names),
        "feature_names": list(feature_names),
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
