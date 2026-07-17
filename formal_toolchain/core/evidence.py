"""逐 obligation 的原始证据合同。

证据对象是 candidate compiler 与 fresh verifier 之间唯一允许传递的
语义结果形态。这里不提供任何默认 PASS；调用方如果还没有真实 checker，
必须显式构造 ``UNRESOLVED``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


VALID_STATUSES = frozenset({"PASS", "FAIL", "UNRESOLVED", "NOT_APPLICABLE"})


@dataclass(frozen=True)
class RawEvidence:
    """一个 obligation 对应的一份、不可复用为其他命题的原始证据。"""

    obligation_id: str
    status: str
    route: str | None
    code: str | None
    witness: Mapping[str, Any]
    checker_input_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid evidence status: {self.status}")
        if not self.obligation_id:
            raise ValueError("evidence obligation_id 不能为空")

    def to_dict(self) -> dict[str, Any]:
        """返回不含 dataclass 私有状态的 canonical-friendly 对象。"""

        return {
            "obligation_id": self.obligation_id,
            "status": self.status,
            "route": self.route,
            "code": self.code,
            "witness": dict(self.witness),
            "checker_input_hashes": dict(self.checker_input_hashes),
        }


def unresolved(obligation_id: str, code: str,
               *, route: str = "UNRESOLVED",
               witness: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """构造 fail-closed 证据，供未实现或输入不足的 checker 使用。"""

    return RawEvidence(
        obligation_id=obligation_id,
        status="UNRESOLVED",
        route=route,
        code=code,
        witness=dict(witness or {}),
        checker_input_hashes={},
    ).to_dict()


def normalize_evidence(obligation_id: str, raw: Mapping[str, Any] | None,
                       *, missing_code: str = "EVIDENCE_MISSING") -> dict[str, Any]:
    """把旧模块返回的 evidence 规范化为单 obligation 对象。

    该函数只保留已有状态，不把缺少状态的对象提升为 PASS；旧对象只能被
    视为 UNRESOLVED，避免历史摘要继续为最终 claim 授权。
    """

    if not isinstance(raw, Mapping):
        return unresolved(obligation_id, missing_code)
    status = raw.get("status", raw.get("obligation_status"))
    if status not in VALID_STATUSES:
        return unresolved(obligation_id, "EVIDENCE_STATUS_MISSING")
    return RawEvidence(
        obligation_id=obligation_id,
        status=str(status),
        route=raw.get("route"),
        code=raw.get("code"),
        witness=dict(raw),
        checker_input_hashes=dict(raw.get("checker_input_hashes", {})),
    ).to_dict()
