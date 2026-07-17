"""Phase H01：消费 Phase F 已认证预算域。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.hashing import sha256_object


SUPPORTED_SCHEMAS = {"budget_domain_v1", "budget_domain_v2"}


def consume_budget_domain(certificate: Mapping[str, Any], expected_context_hash: str | None = None) -> dict[str, Any]:
    if certificate.get("status") != "PASS" or certificate.get("schema_version") not in SUPPORTED_SCHEMAS:
        raise ValueError("budget_domain_certificate 缺失或未通过")
    if not isinstance(certificate.get("context_hash"), str):
        raise ValueError("budget domain 缺少 context_hash")
    if expected_context_hash is not None and certificate.get("context_hash") != expected_context_hash:
        raise ValueError("budget domain context 不一致")
    tasks = certificate.get("tasks")
    if not isinstance(tasks, Mapping) or not tasks:
        raise ValueError("budget domain 必须包含有限 task 域")
    for name, domain in tasks.items():
        interval = domain.get("integer_interval")
        if not isinstance(interval, Mapping):
            raise ValueError(f"{name} 缺少 integer_interval")
        lower = int(interval["lower"])
        upper = int(interval["upper"])
        initial = int(domain["initial"])
        if lower > upper or not lower <= initial <= upper:
            raise ValueError(f"{name} budget interval 非法")
    return {"status": "PASS", "schema_version": "consumed_budget_domain_v1",
            "source_hash": sha256_object(certificate), "tasks": dict(tasks)}


def task_interval(consumed: Mapping[str, Any], task_name: str) -> tuple[int, int]:
    row = consumed["tasks"][task_name]
    interval = row["integer_interval"]
    return int(interval["lower"]), int(interval["upper"])
