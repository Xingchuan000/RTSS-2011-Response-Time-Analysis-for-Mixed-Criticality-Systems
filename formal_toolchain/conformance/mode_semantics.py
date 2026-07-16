"""Phase F04：C-AMC-sem 模式转换的显式合同检查。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def check_mode_semantics(*, effective_config: Mapping[str, Any], micro_scenarios: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] = ()) -> dict[str, Any]:
    required = {"semantics", "c_amc_sem_primary_on_switch_time"}
    missing = sorted(required - set(effective_config))
    if missing:
        raise ValueError(f"effective config 缺少模式字段: {missing}")
    if effective_config["semantics"] != "C_AMC_SEM":
        raise ValueError("P0 目标必须使用 C_AMC_SEM")
    required = {"completion_at_deadline", "deadline_observe_only", "primary_lo_b_plus_one",
                "hi_nontruncation", "idle_recovery", "c_amc_single_switch", "inherited_hi_entry"}
    scenario_items = list(micro_scenarios.values()) if isinstance(micro_scenarios, Mapping) else list(micro_scenarios)
    observed = set(micro_scenarios) if isinstance(micro_scenarios, Mapping) else {
        str(item.get("name")) if isinstance(item, Mapping) else "" for item in scenario_items}
    # 微场景必须来自真实 runtime 执行结果；缺失时保持 UNRESOLVED，不能把
    # 理论声明或空列表当作 conformance evidence。
    if not scenario_items:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "MODE_MICRO_SCENARIOS_MISSING", "required": sorted(required)}}
    evidence_fields = {"event_sequence", "state_snapshots", "assertions", "final_state"}
    if (not required <= observed or
        any(item.get("status") != "PASS" or not evidence_fields <= set(item) or
            not item.get("event_sequence") or not item.get("state_snapshots") or
            not isinstance(item.get("assertions"), Mapping) or
            any(value is not True for value in item["assertions"].values())
            for item in scenario_items if isinstance(item, Mapping))):
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "MODE_MICRO_SCENARIOS_INCOMPLETE", "missing": sorted(required - observed)}}
    expected_events = {
        "completion_at_deadline": ("job_completion",),
        "deadline_observe_only": ("deadline_miss",),
        "primary_lo_b_plus_one": ("budget_overrun",),
        "hi_nontruncation": ("c_amc_sem_mode_switch", "job_completion"),
        "idle_recovery": ("c_amc_sem_mode_switch", "job_completion"),
        "c_amc_single_switch": ("c_amc_sem_mode_switch", "job_arrival"),
        "inherited_hi_entry": ("c_amc_sem_mode_switch", "job_arrival"),
    }
    for name, item in (micro_scenarios.items() if isinstance(micro_scenarios, Mapping) else enumerate(scenario_items)):
        if name in expected_events and not all(event in item.get("event_sequence", []) for event in expected_events[name]):
            return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                    "failure": {"code": "MODE_EVENT_ORDER_EVIDENCE_MISSING", "scenario": name}}
    if not any(item.get("assertions", {}).get("inherited_hi_entry") is True for item in scenario_items if isinstance(item, Mapping)):
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "INHERITED_HI_ENTRY_EVIDENCE_MISSING"}}
    return {"status": "PASS", "schema_version": "mode_semantics_conformance_v1",
            "switch_trigger": "abnormal_hi_arrival", "hi_mode_persistent": True,
            "idle_recovery": True, "same_batch_frozen": True,
            "primary_on_switch_time": effective_config["c_amc_sem_primary_on_switch_time"],
            "micro_scenarios": len(scenario_items)}
