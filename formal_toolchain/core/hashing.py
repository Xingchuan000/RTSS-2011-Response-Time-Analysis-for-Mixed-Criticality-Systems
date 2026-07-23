"""canonical object、文件和目录的 SHA-256 工具。"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .canonical_json import canonical_bytes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_object(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def proof_safe_value(value: Any) -> Any:
    """Normalize runtime diagnostics before they enter a proof-object hash.

    Binary64 values are represented by the shortest round-trippable decimal
    string.  This preserves the exact Python float value while keeping
    canonical JSON free of platform-dependent JSON numbers.
    """

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("proof object 禁止 NaN 和 Inf")
        return format(value, ".17g")
    if isinstance(value, Mapping):
        return {str(key): proof_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [proof_safe_value(item) for item in value]
    return value


def sha256_proof_object(value: Any) -> str:
    """Hash an object after explicit proof-boundary normalization."""

    return sha256_object(proof_safe_value(value))
