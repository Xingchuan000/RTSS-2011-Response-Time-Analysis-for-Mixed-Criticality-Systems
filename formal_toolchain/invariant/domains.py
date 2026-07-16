"""Phase H01：消费 Phase F 已认证预算域。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.hashing import sha256_object


def consume_budget_domain(certificate: Mapping[str, Any], expected_context_hash: str | None = None) -> dict[str, Any]:
    if certificate.get("status") != "PASS" or certificate.get("schema_version") != "budget_domain_v1":
        raise ValueError("budget_domain_certificate 缺失或未通过")
    if not isinstance(certificate.get("context_hash"), str):
        raise ValueError("budget domain 缺少 context_hash")
    if expected_context_hash is not None and certificate.get("context_hash") != expected_context_hash:
        raise ValueError("budget domain context 不一致")
    tasks = certificate.get("tasks")
    if not isinstance(tasks, Mapping) or not tasks:
        raise ValueError("budget domain 必须包含有限 task 域")
    for name, domain in tasks.items():
        values = domain.get("finite_integer_domain")
        if not isinstance(values, list) or not values or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError(f"{name} 没有有限整数 budget domain")
        if domain.get("initial") not in values or domain.get("code_lower") > domain.get("initial"):
            raise ValueError(f"{name} budget domain 不满足初始值和 HI lower")
    return {"status": "PASS", "schema_version": "consumed_budget_domain_v1",
            "source_hash": sha256_object(certificate), "tasks": dict(tasks)}
