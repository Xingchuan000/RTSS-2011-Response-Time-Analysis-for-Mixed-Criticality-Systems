"""outer bundle root 的纯函数接口。"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def compute_outer_bundle_root(preimage: Mapping[str, Any]) -> str:
    """只对调用方提供的 root preimage 计算 SHA-256，不读取 summary/report。"""

    return sha256_object(dict(preimage))
