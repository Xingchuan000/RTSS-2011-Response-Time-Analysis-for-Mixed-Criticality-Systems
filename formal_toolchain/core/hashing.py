"""canonical object、文件和目录的 SHA-256 工具。"""

from __future__ import annotations

import hashlib
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
