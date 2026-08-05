"""从已构造的 target 导出 effective runtime config 及 provenance。

这里不读取默认配置来猜测最终值；调用方传入的 environment、env 和 runtime
config 对象按明确优先级展开，wrapper/env 的字段会覆盖底层 config 的同名值。
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

REQUIRED_EFFECTIVE_FIELDS = frozenset({
    "semantics", "drop_lo_jobs_on_hi_switch", "c_amc_sem_lo_degradation_ratio",
    "c_amc_sem_primary_on_switch_time", "stop_at_first_miss", "capture_trace",
    "capture_debug_events", "agent_period", "action_space", "budget_increase_ratio",
    "budget_decrease_ratio", "budget_floor_ratio", "forbid_decreasing_hi_budgets",
    "mask_detail_mode", "enable_deploy_cap_mask", "deploy_cap_mask_ratio",
    "deploy_cap_mask_criticality", "observation_mode", "check_safety",
})


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        # effective config 允许配置浮点，但必须保留为字符串，避免 proof object float。
        return format(value, ".17g")
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _plain(item) for key, item in vars(value).items() if not key.startswith("_")}
    return repr(value)


def export_effective_config(runtime_config: Any, environment: Any = None) -> dict[str, Any]:
    """导出最终可观察字段，每个字段附带来源对象和来源属性。"""
    records: dict[str, dict[str, Any]] = {}
    base = _plain(runtime_config)
    for name, value in base.items():
        records[name] = {"value": value, "origin": "runtime_config", "source_object": type(runtime_config).__name__,
                         "final_origin": "runtime_config", "source_attribute": name, "source_chain": ["runtime_config"]}
    # env 是 wrapper 的最终视图：只覆盖已存在的 config 字段或计划明确的 wrapper 字段。
    if environment is not None:
        env_values = _plain(environment)
        if isinstance(env_values, dict):
            for name, value in env_values.items():
                if name in records or name in {"agent_period", "action_space", "observation_mode",
                                                "mask_detail_mode", "budget_increase_ratio", "budget_decrease_ratio",
                                                "budget_floor_ratio", "forbid_decreasing_hi_budgets",
                                                "budget_rounding_mode", "min_budget_delta", "processor_overhead",
                                                "enable_deploy_cap_mask", "deploy_cap_mask_ratio",
                                                "deploy_cap_mask_criticality", "capture_trace", "capture_debug_events",
                                                "check_safety"}:
                    previous = records.get(name, {})
                    chain = list(previous.get("source_chain", ["runtime_config"])) + ["environment_wrapper"]
                    records[name] = {"value": value, "origin": "environment_wrapper", "source_object": type(environment).__name__,
                                     "final_origin": "environment_wrapper",
                                     "source_attribute": name, "source_chain": chain}
    return {"schema_version": "effective_runtime_config_v1", "fields": records}


def export_formal_target_config(target: Any) -> dict[str, Any]:
    """严格从 FormalTarget 的 runtime_config + environment 导出配置。

    缺字段或 wrapper 覆盖来源不唯一时返回 UNRESOLVED，避免把 dataclass 默认值
    冒充 effective deployment config。
    """
    provenance = getattr(target, "provenance", {}) or {}
    conflicts = provenance.get("source_conflicts", [])
    if conflicts:
        return {"status": "UNRESOLVED", "failure": {"code": "EFFECTIVE_CONFIG_SOURCE_CONFLICT",
                "route": "MODEL_CONFORMANCE_FAILED", "fields": list(conflicts)}}
    result = export_effective_config(target.runtime_config, target.environment)
    missing = sorted(REQUIRED_EFFECTIVE_FIELDS - set(result["fields"]))
    if missing:
        return {"status": "UNRESOLVED", "failure": {"code": "EFFECTIVE_CONFIG_FIELD_MISSING",
                "route": "MODEL_CONFORMANCE_FAILED", "fields": missing}, "config": result}
    metadata = provenance.get("feature_metadata")
    if isinstance(metadata, dict) and metadata.get("feature_names") is not None:
        observed = list(getattr(target, "feature_names", ()))
        if list(metadata["feature_names"]) != observed:
            return {"status": "UNRESOLVED", "failure": {"code": "FEATURE_METADATA_MISMATCH",
                    "route": "MODEL_CONFORMANCE_FAILED"}, "config": result}
    result["status"] = "PASS"
    return result
