"""把机器状态投影为稳定、无结论偷换的报告数据。"""

from __future__ import annotations

from typing import Any


def status_projection(obligation: dict[str, Any]) -> dict[str, Any]:
    """仅复制状态证据字段，不把 candidate 或 HOUT 指标变成 PASS。"""
    return {"id": obligation.get("id", obligation.get("obligation_id")),
            "status": obligation.get("obligation_status", obligation.get("status")),
            "failure": obligation.get("failure"), "evidence": obligation.get("evidence")}
