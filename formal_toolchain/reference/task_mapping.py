"""Phase I：把实际代码任务确定性地映射为 C-AMC-sem 参考任务。

本模块只做计划中定义的数值映射。它不读取 HOUT、不重排任务，也不从
默认配置推断预算；缺失的 LO budget 信息会直接报错（fail closed）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True, slots=True)
class ReferenceTask:
    """参考模型中的一个不可变任务记录。"""

    name: str
    period: int
    deadline: int
    c_lo: int
    c_hi: int
    criticality: str
    priority_index: int
    code_c_lo: int
    code_c_hi: int
    degraded_cost: int | None = None
    offset: int = 0

    def __post_init__(self) -> None:
        if self.criticality not in {"LO", "HI"}:
            raise ValueError(f"{self.name}: criticality must be LO or HI")
        if self.period <= 0:
            raise ValueError(f"{self.name}: period must be positive")
        if not 0 < self.deadline <= self.period:
            raise ValueError(f"{self.name}: constrained deadline requires 0 < D <= T")
        if not 0 <= self.offset < self.period:
            raise ValueError(f"{self.name}: offset must satisfy 0 <= offset < period")
        if self.priority_index < 0:
            raise ValueError(f"{self.name}: priority index must be non-negative")
        if self.c_lo <= 0 or self.c_hi <= 0:
            raise ValueError(f"{self.name}: reference WCETs must be positive")
        if self.code_c_lo < 0 or self.code_c_hi < 0:
            raise ValueError(f"{self.name}: code WCETs must be non-negative")
        if self.criticality == "LO" and self.c_hi > self.c_lo:
            raise ValueError(f"{self.name}: LO task requires C_HI <= C_LO")
        if self.degraded_cost is not None and self.degraded_cost < 0:
            raise ValueError(f"{self.name}: degraded_cost must be non-negative")


@dataclass(frozen=True, slots=True)
class ReferenceTaskset:
    """固定优先级、单一 context 下的 canonical reference taskset。"""

    tasks: tuple[ReferenceTask, ...]
    source_context_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("reference taskset 不能为空")
        if not isinstance(self.source_context_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_context_hash):
            raise ValueError("reference taskset 必须绑定 64 位 certified reference context hash")
        if tuple(task.priority_index for task in self.tasks) != tuple(range(len(self.tasks))):
            raise ValueError("reference priority_index 必须连续且与 task 顺序一致")

    @property
    def priority_order(self) -> tuple[str, ...]:
        return tuple(task.name for task in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        value = {"schema_version": "reference_taskset_v2",
                "tasks": [asdict(task) for task in self.tasks],
                "priority_order": list(self.priority_order),
                "periodic_language_is_sporadic_sub_language": True,
                "source_context_hash": self.source_context_hash}
        value["fingerprint"] = sha256_object(value)
        return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} 必须是 int，不能使用 float/bool")
    return value


def _certified_upper(envelope: Mapping[str, Any], task_name: str, *, allow_unverified_candidate: bool = False) -> int:
    """读取唯一可信的 ``B̄`` 来源，并验证 Phase H 认证边界。

    ``budget_by_task`` 里的字段只能作为 provenance 交叉检查，绝不能成为
    数值来源；否则调用方可以通过修改 provenance 绕过 Phase H 的上界。
    """
    is_candidate_view = envelope.get("trust_level") == "CANDIDATE_UNVERIFIED" and envelope.get("not_a_certified_envelope") is True
    if is_candidate_view and not allow_unverified_candidate:
        raise ValueError("candidate envelope 不能作为 trusted reference 输入")
    if not is_candidate_view and (envelope.get("schema_version") not in {"certified_envelope_v1", "certified_envelope_v2", "certified_envelope_v3"} or envelope.get("status") != "PASS"):
        raise ValueError("certified envelope 必须是 PASS 的 certified_envelope_v1/v2/v3")
    if envelope.get("schema_version") == "certified_envelope_v3":
        if envelope.get("method") != "single_action_safety_polytope_projection":
            raise ValueError("certified envelope v3 method invalid")
        required = ("safety_polytope_hash", "coordinate_upper_witness_hash",
                    "action_transition_hash", "mask_fallback_hash")
        if any(not isinstance(envelope.get(field), str) for field in required):
            raise ValueError("certified envelope v3 缺少 structural binding")
    preservation = envelope.get("preservation_certificate")
    preservation_hash = envelope.get("preservation_certificate_hash")
    if is_candidate_view:
        return _integer(envelope["upper"][task_name], f"candidate_envelope.upper[{task_name}]")
    if not isinstance(preservation, Mapping) or preservation.get("obligation_status") != "PASS":
        raise ValueError("certified envelope 缺少 PASS preservation certificate")
    if sha256_object(dict(preservation)) != preservation_hash:
        raise ValueError("certified envelope preservation certificate hash mismatch")
    upper = envelope.get("upper")
    active_upper = envelope.get("active_release_budget_upper")
    if not isinstance(upper, Mapping) or not isinstance(active_upper, Mapping):
        raise ValueError("certified envelope 缺少 upper/active_release_budget_upper")
    if task_name not in upper or task_name not in active_upper:
        raise ValueError(f"certified envelope 缺少 task {task_name} 的 upper")
    value = _integer(upper[task_name], f"certified_envelope.upper[{task_name}]")
    active_value = _integer(active_upper[task_name], f"certified_envelope.active_release_budget_upper[{task_name}]")
    if value < 0 or active_value != value:
        raise ValueError(f"task {task_name} 的 active upper 与 certified upper 不一致")
    return value


def degraded_cost(code_c_lo: int, *, xf: float | int) -> int:
    """按 Python ``round`` 的 ties-to-even 计算 degraded LO 成本。"""
    c = _integer(code_c_lo, "code_c_lo")
    if c <= 0:
        raise ValueError("code_c_lo 必须为正整数")
    if isinstance(xf, bool) or not isinstance(xf, (int, float)):
        raise TypeError("xf 必须是数值")
    if isinstance(xf, float) and not math.isfinite(xf):
        raise ValueError("xf 必须是有限数")
    value = max(1, min(c, round(xf * c)))
    return int(value)


def build_reference_taskset(
    ordered_tasks: Sequence[Any],
    budget_by_task: Mapping[str, Mapping[str, Any]],
    *, xf: float | int, certified_envelope: Mapping[str, Any] | None = None,
    semantic_context_hash: str | None = None,
    effective_runtime_config_hash: str | None = None,
    source_context_hash: str | None = None,
    allow_unverified_candidate: bool = False,
) -> ReferenceTaskset:
    """严格执行 I01/I02，保持输入顺序和所有时序参数不变。"""
    if not ordered_tasks:
        raise ValueError("reference taskset 不能为空")
    if source_context_hash is not None:
        raise ValueError("reference context hash 必须由 preimage 自动计算，不能由调用方传入")
    if not isinstance(semantic_context_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", semantic_context_hash):
        raise ValueError("semantic_context_hash 必须是 64 位 SHA-256")
    if not isinstance(effective_runtime_config_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", effective_runtime_config_hash):
        raise ValueError("effective_runtime_config_hash 必须是 64 位 SHA-256")
    if not isinstance(certified_envelope, Mapping):
        raise ValueError("reference taskset 必须接收实际 certified envelope 对象")
    certified_envelope_hash = sha256_object(dict(certified_envelope))
    result: list[ReferenceTask] = []
    names: set[str] = set()
    for index, task in enumerate(ordered_tasks):
        name = str(task.name)
        if name in names:
            raise ValueError(f"task name 重复：{name}")
        names.add(name)
        if name not in budget_by_task:
            raise ValueError(f"缺失 task {name} 的 budget provenance")
        budget = budget_by_task[name]
        code_lo = _integer(task.c_lo, f"{name}.c_lo")
        code_hi = _integer(task.c_hi, f"{name}.c_hi")
        period = _integer(task.period, f"{name}.period")
        deadline = _integer(task.deadline, f"{name}.deadline")
        if period <= 0 or not 0 < deadline <= period:
            raise ValueError(f"{name}: expected constrained deadline 0 < D <= T")
        offset = _integer(getattr(task, "offset", 0), f"{name}.offset")
        if offset != 0:
            raise ValueError(f"{name}: P0 runtime only supports zero release offset (got {offset})")
        if not 0 <= offset < period:
            raise ValueError(f"{name}: invalid periodic release offset")
        # I02 的 B̄_i 只来自 Phase H certified envelope；budget_floor 和
        # provenance.b_bar 都只是绑定检查，不能改变认证后的数值。
        if budget.get("certified_envelope_hash") != certified_envelope_hash:
            raise ValueError(f"task {name} 的 budget provenance 未绑定传入的 certified envelope")
        b_bar = _certified_upper(certified_envelope, name,
                                 allow_unverified_candidate=allow_unverified_candidate)
        if "b_bar" in budget and budget["b_bar"] != b_bar:
            raise ValueError(f"task {name} 的 provenance b_bar 与 certified upper 不一致")
        crit = getattr(task.criticality, "value", str(task.criticality))
        if crit == "LO":
            deg = degraded_cost(code_lo, xf=xf)
            ref_lo, ref_hi = max(b_bar + 1, deg), deg
            if ref_hi > ref_lo:
                raise ValueError(f"LO reference task {name} 不满足 CrefHI<=CrefLO")
        elif crit == "HI":
            deg = None
            ref_lo, ref_hi = code_lo, code_hi
            if ref_lo > ref_hi:
                raise ValueError(f"HI reference task {name} 不满足 CrefLO<=CrefHI")
        else:
            raise ValueError(f"未知 criticality：{crit}")
        result.append(ReferenceTask(name, period, deadline,
                                    ref_lo, ref_hi, crit, index, code_lo, code_hi, deg, offset))
    code_records = [{"name": str(task.name), "priority_index": index,
                     "criticality": getattr(task.criticality, "value", str(task.criticality)),
                     "period": _integer(task.period, f"{task.name}.period"), "deadline": _integer(task.deadline, f"{task.name}.deadline"),
                     "code_c_lo": _integer(task.c_lo, f"{task.name}.c_lo"), "code_c_hi": _integer(task.c_hi, f"{task.name}.c_hi")}
                    for index, task in enumerate(ordered_tasks)]
    priority = [record["name"] for record in code_records]
    code_fingerprint = sha256_object({"tasks": code_records, "priority_order": priority})
    reference_records = [asdict(task) for task in result]
    reference_fingerprint = sha256_object({"schema_version": "reference_taskset_v2",
                                           "tasks": reference_records,
                                           "priority_order": priority})
    from formal_toolchain.core.contexts import build_reference_context
    context = build_reference_context(
        semantic_context_hash=semantic_context_hash,
        certified_envelope_hash=certified_envelope_hash,
        code_taskset_fingerprint=code_fingerprint,
        priority_order=priority, xf=str(xf),
        effective_runtime_config_hash=effective_runtime_config_hash,
        reference_taskset_fingerprint=reference_fingerprint,
    )
    return ReferenceTaskset(tuple(result), context["hash"])


def validate_reference_mapping(reference: ReferenceTaskset, ordered_tasks: Sequence[Any], *,
                               budget_by_task: Mapping[str, Mapping[str, Any]],
                               certified_envelope: Mapping[str, Any], xf: float | int,
                               semantic_context_hash: str,
                               effective_runtime_config_hash: str) -> dict[str, Any]:
    """调用独立 verifier，禁止只做 immutable 字段的局部检查。"""
    from formal_toolchain.verifier.reference_mapping_verifier import verify_reference_mapping
    return verify_reference_mapping(
        reference=reference, ordered_tasks=ordered_tasks, budget_by_task=budget_by_task,
        certified_envelope=certified_envelope, xf=xf,
        semantic_context_hash=semantic_context_hash,
        effective_runtime_config_hash=effective_runtime_config_hash,
    )
