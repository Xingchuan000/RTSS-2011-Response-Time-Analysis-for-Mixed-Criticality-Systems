"""Phase I 独立 mapping verifier。

本模块不调用 ``build_reference_taskset``，而是从原始 code task、certified
envelope、xf 和 context 输入重新计算全部 reference 数值及 context preimage。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.contexts import build_reference_context
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.task_mapping import ReferenceTaskset


def _degraded_cost_independent(code_c_lo: Any, *, xf: float | int) -> int:
    """Verifier 自己实现 ties-to-even/clamp，避免共享 production helper。"""
    if isinstance(code_c_lo, bool) or not isinstance(code_c_lo, int) or code_c_lo <= 0:
        raise ValueError("code_c_lo invalid")
    if isinstance(xf, bool) or not isinstance(xf, (int, float)):
        raise TypeError("xf invalid")
    if isinstance(xf, float) and not __import__("math").isfinite(xf):
        raise ValueError("xf invalid")
    return int(max(1, min(code_c_lo, round(xf * code_c_lo))))


def _integer_independent(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be int")
    return value


def _certified_upper(envelope: Mapping[str, Any], name: str) -> int:
    if envelope.get("schema_version") not in {"certified_envelope_v1", "certified_envelope_v2", "certified_envelope_v3"} or envelope.get("status") != "PASS":
        raise ValueError("certified envelope status/schema invalid")
    if envelope.get("schema_version") == "certified_envelope_v2" and envelope.get("trust_level") not in {None, "VERIFIED"}:
        raise ValueError("certified envelope trust level invalid")
    if envelope.get("schema_version") == "certified_envelope_v3":
        if envelope.get("method") != "single_action_safety_polytope_projection":
            raise ValueError("certified envelope method invalid")
        required = ("safety_polytope_hash", "coordinate_upper_witness_hash",
                    "action_transition_hash", "mask_fallback_hash")
        if any(not isinstance(envelope.get(field), str) for field in required):
            raise ValueError("certified envelope structural binding missing")
    preservation = envelope.get("preservation_certificate")
    if not isinstance(preservation, Mapping) or preservation.get("obligation_status") != "PASS":
        raise ValueError("preservation certificate invalid")
    if sha256_object(dict(preservation)) != envelope.get("preservation_certificate_hash"):
        raise ValueError("preservation certificate hash invalid")
    upper = envelope.get("upper")
    active = envelope.get("active_release_budget_upper")
    if not isinstance(upper, Mapping) or not isinstance(active, Mapping) or name not in upper or name not in active:
        raise ValueError(f"certified upper missing for {name}")
    value = upper[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or active[name] != value:
        raise ValueError(f"certified upper invalid for {name}")
    return value


def verify_reference_mapping(*, reference: ReferenceTaskset, ordered_tasks: Sequence[Any],
                             budget_by_task: Mapping[str, Mapping[str, Any]],
                             certified_envelope: Mapping[str, Any], xf: float | int,
                             semantic_context_hash: str,
                             effective_runtime_config_hash: str) -> dict[str, Any]:
    """独立重算 Phase I mapping；任何 mismatch 都是 FAIL。"""
    try:
        envelope_hash = sha256_object(dict(certified_envelope))
        expected: list[dict[str, Any]] = []
        names: list[str] = []
        for index, code in enumerate(ordered_tasks):
            name = str(code.name)
            names.append(name)
            budget = budget_by_task[name]
            if budget.get("certified_envelope_hash") != envelope_hash:
                raise ValueError(f"{name}: certified envelope hash mismatch")
            b_bar = _certified_upper(certified_envelope, name)
            if "b_bar" in budget and budget["b_bar"] != b_bar:
                raise ValueError(f"{name}: provenance b_bar mismatch")
            criticality = getattr(code.criticality, "value", str(code.criticality))
            if criticality == "LO":
                cdeg = _degraded_cost_independent(_integer_independent(code.c_lo, f"{name}.c_lo"), xf=xf)
                c_lo, c_hi = max(b_bar + 1, cdeg), cdeg
            elif criticality == "HI":
                c_lo = _integer_independent(code.c_lo, f"{name}.c_lo")
                c_hi = _integer_independent(code.c_hi, f"{name}.c_hi")
            else:
                raise ValueError(f"{name}: criticality invalid")
            period = _integer_independent(code.period, f"{name}.period")
            deadline = _integer_independent(code.deadline, f"{name}.deadline")
            if period <= 0 or deadline < 0:
                raise ValueError(f"{name}: period/deadline invalid")
            expected.append({"name": name, "period": period,
                             "deadline": deadline, "c_lo": c_lo, "c_hi": c_hi,
                             "criticality": criticality, "priority_index": index,
                             "code_c_lo": int(code.c_lo), "code_c_hi": int(code.c_hi),
                             "degraded_cost": cdeg if criticality == "LO" else None})
        code_records = [{"name": str(code.name), "priority_index": index,
                         "criticality": getattr(code.criticality, "value", str(code.criticality)),
                         "period": _integer_independent(code.period, f"{code.name}.period"), "deadline": _integer_independent(code.deadline, f"{code.name}.deadline"),
                         "code_c_lo": _integer_independent(code.c_lo, f"{code.name}.c_lo"), "code_c_hi": _integer_independent(code.c_hi, f"{code.name}.c_hi")}
                        for index, code in enumerate(ordered_tasks)]
        code_fp = sha256_object({"tasks": code_records, "priority_order": names})
        ref_fp = sha256_object({"schema_version": "reference_taskset_v1",
                                "tasks": expected, "priority_order": names})
        context = build_reference_context(
            semantic_context_hash=semantic_context_hash,
            certified_envelope_hash=envelope_hash,
            code_taskset_fingerprint=code_fp, priority_order=names, xf=str(xf),
            effective_runtime_config_hash=effective_runtime_config_hash,
            reference_taskset_fingerprint=ref_fp,
        )
        actual = [asdict(item) for item in reference.tasks]
        if actual != expected or reference.source_context_hash != context["hash"]:
            raise ValueError("reference taskset 数值或 context preimage mismatch")
        return obligation_certificate(
            obligation_id="CODE_REFERENCE_UPPER_BOUND_MAPPING", status="PASS",
            context_hash=context["hash"], inputs={"code_taskset_fingerprint": code_fp,
            "certified_envelope_hash": envelope_hash}, witness={"reference": expected,
            "context_preimage": context}, checker_id=__name__, checker_version="phase-i-v2",
        )
    except (KeyError, TypeError, ValueError) as exc:
        # 失败分支仍使用可追溯 context；无法计算 context 时返回诊断对象，不授予 PASS。
        return {"artifact_schema_version": "reference_mapping_diagnostic_v1",
                "obligation_id": "CODE_REFERENCE_UPPER_BOUND_MAPPING",
                "obligation_status": "FAIL", "failure": {"route": "MODEL_CONFORMANCE_FAILED",
                "code": "REFERENCE_MAPPING_MISMATCH", "message": str(exc)}}
