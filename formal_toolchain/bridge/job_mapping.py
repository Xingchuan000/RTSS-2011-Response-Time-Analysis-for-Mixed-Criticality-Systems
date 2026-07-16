"""Phase K01：释放时冻结的 concrete→reference removal demand 映射。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.artifact import verify_obligation_certificate


@dataclass(frozen=True, slots=True)
class ReleaseFixedRemovalMapping:
    """一对 concrete/reference job 的 release-fixed 执行需求。"""

    job_key: tuple[str, int]
    mode: str
    actual_cost: int
    release_budget: int | None
    degraded_cost: int | None
    reference_demand: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["job_key"] = list(self.job_key)
        return data


def exact_removal_demand(*, actual_cost: int, primary_mode: str, release_budget: int | None = None,
                         degraded_cost: int | None = None) -> int:
    """严格按 K01 三分支计算 E_C(J)，不使用运行时当前预算。"""
    if isinstance(actual_cost, bool) or not isinstance(actual_cost, int) or actual_cost < 0:
        raise TypeError("actual_cost 必须是非负 int")
    if primary_mode == "LO":
        if release_budget is None:
            raise ValueError("primary LO 必须提供 release budget")
        if not isinstance(release_budget, int) or isinstance(release_budget, bool) or release_budget < 0:
            raise TypeError("release_budget 必须是非负 int")
        return min(actual_cost, release_budget + 1)
    if primary_mode == "DEGRADED_LO":
        if degraded_cost is None:
            raise ValueError("degraded LO 必须提供 degraded cost")
        if not isinstance(degraded_cost, int) or isinstance(degraded_cost, bool) or degraded_cost < 0:
            raise TypeError("degraded_cost 必须是非负 int")
        return min(actual_cost, degraded_cost)
    if primary_mode == "HI":
        return actual_cost
    raise ValueError(f"未知 primary mode：{primary_mode}")


def map_release_fixed_job(*, task_name: str, release_index: int, actual_cost: int,
                          primary_mode: str, release_budget: int | None = None,
                          degraded_cost: int | None = None) -> ReleaseFixedRemovalMapping:
    demand = exact_removal_demand(actual_cost=actual_cost, primary_mode=primary_mode,
                                  release_budget=release_budget, degraded_cost=degraded_cost)
    return ReleaseFixedRemovalMapping((str(task_name), int(release_index)), primary_mode,
                                      actual_cost, release_budget, degraded_cost, demand)


def build_release_fixed_removal_certificate(mappings: list[ReleaseFixedRemovalMapping], *,
                                            source_context_hash: str) -> dict[str, Any]:
    """生成 K01 要求的 release-fixed mapping certificate。"""
    if not re.fullmatch(r"[0-9a-f]{64}", source_context_hash):
        raise ValueError("source_context_hash 必须是 64 位 SHA-256")
    if not mappings:
        raise ValueError("mapping certificate 不能为空")
    keys = [mapping.job_key for mapping in mappings]
    if len(keys) != len(set(keys)):
        raise ValueError("job key 重复，不能生成唯一 mapping certificate")
    for mapping in mappings:
        expected = exact_removal_demand(
            actual_cost=mapping.actual_cost, primary_mode=mapping.mode,
            release_budget=mapping.release_budget, degraded_cost=mapping.degraded_cost,
        )
        if mapping.reference_demand != expected:
            raise ValueError(f"job {mapping.job_key} 的 reference demand 与公式不一致")
    result = {"schema_version": "release_fixed_removal_mapping_v1",
              "status": "PASS", "source_context_hash": source_context_hash,
              "mappings": [mapping.to_dict() for mapping in mappings]}
    result.update(obligation_certificate(
        obligation_id="RELEASE_FIXED_REMOVAL_MAPPING", status="PASS",
        context_hash=source_context_hash, inputs={"mapping_count": len(mappings)},
        witness={"mappings": result["mappings"]},
        checker_id="formal_toolchain.bridge.job_mapping", checker_version="phase-k-v1",
    ))
    return result


def build_parameterized_release_mapping_certificate(*, source_context_hash: str) -> dict[str, Any]:
    """生成 K01 的参数化公式证书；有限 mappings 仅是边界 replay 证据。

    这里不接受调用方传入一组“已通过”的样例。证书固定记录三种 typed
    输入域与公式，验证器据此检查任意 release-fixed 实例。
    """
    if not re.fullmatch(r"[0-9a-f]{64}", source_context_hash):
        raise ValueError("source_context_hash 必须是 64 位 SHA-256")
    schema = {"actual_cost": "Nat", "release_budget": "Nat", "degraded_cost": "Nat",
              "primary_mode": ["LO", "DEGRADED_LO", "HI"]}
    formulas = {
        "LO": "E_C(J)=min(actual_cost, release_budget+1)",
        "DEGRADED_LO": "E_C(J)=min(actual_cost, degraded_cost)",
        "HI": "E_C(J)=actual_cost",
    }
    result = {"schema_version": "release_fixed_removal_parameterized_v1", "status": "PASS",
              "source_context_hash": source_context_hash, "input_schema": schema,
              "formulas": formulas, "finite_boundary_evidence": []}
    result.update(obligation_certificate(
        obligation_id="RELEASE_FIXED_REMOVAL_PARAMETERIZED", status="PASS",
        context_hash=source_context_hash, inputs={"formula_schema": schema},
        witness={"input_schema": schema, "formulas": formulas},
        checker_id="formal_toolchain.bridge.job_mapping", checker_version="phase-k-v2"))
    return result


def verify_parameterized_release_mapping_certificate(certificate: Mapping[str, Any]) -> bool:
    """验证参数化公式对象本身，不能被有限样例替代。"""
    return (certificate.get("obligation_id") == "RELEASE_FIXED_REMOVAL_PARAMETERIZED"
            and certificate.get("obligation_status") == "PASS"
            and verify_obligation_certificate(certificate)
            and certificate.get("formulas") == {
                "LO": "E_C(J)=min(actual_cost, release_budget+1)",
                "DEGRADED_LO": "E_C(J)=min(actual_cost, degraded_cost)",
                "HI": "E_C(J)=actual_cost",
            })


def verify_release_fixed_removal_certificate(certificate: Mapping[str, Any]) -> bool:
    """独立验证 K01 certificate 的 envelope、context 和每条 demand 公式。"""
    if certificate.get("obligation_id") != "RELEASE_FIXED_REMOVAL_MAPPING":
        return False
    if certificate.get("obligation_status") != "PASS" or not verify_obligation_certificate(certificate):
        return False
    for item in certificate.get("mappings", []):
        try:
            expected = exact_removal_demand(
                actual_cost=item["actual_cost"], primary_mode=item["mode"],
                release_budget=item.get("release_budget"), degraded_cost=item.get("degraded_cost"),
            )
        except (KeyError, TypeError, ValueError):
            return False
        if expected != item.get("reference_demand"):
            return False
    return bool(certificate.get("mappings"))
