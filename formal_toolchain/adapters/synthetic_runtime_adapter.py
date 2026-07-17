"""合成 target 的显式 runtime adapter。

这个类只服务于 ``target_kind=SYNTHETIC_P0`` 的测试输入。真实 target 不会
进入本模块；真实 target 必须在自己的 factory 中提供 ``AMCRealRuntimeAdapter``。
"""

from __future__ import annotations

from typing import Any, Mapping

from amc_py.rl.actions import build_budget_action_space
from formal_toolchain.core.hashing import sha256_object


class SyntheticP0RuntimeAdapter:
    """把既有 synthetic runtime 原语包装成 FormalRuntimeAdapter。"""

    def __init__(self, target: Any) -> None:
        from formal_toolchain.adapters.synthetic_policy import build_runtime_adapter

        self.target = target
        self.actions = build_budget_action_space(
            target.ordered_tasks,
            action_space=str(target.runtime_config.action_space),
            budget_increase_ratio=float(target.runtime_config.budget_increase_ratio),
            budget_decrease_ratio=float(target.runtime_config.budget_decrease_ratio),
        )
        self._runtime = build_runtime_adapter(target, self.actions)

    def build_runtime_state_from_budget_vector(self, budget_by_task: Mapping[str, int]) -> dict[str, Any]:
        names = [str(task.name) for task in self.target.ordered_tasks]
        return {
            "budgets": {name: int(budget_by_task[name]) for name in names},
            "initial_budgets": {name: int(self.target.provenance["budget_by_task"][name]["initial_runtime_budget"]) for name in names},
            "floors": {name: int(self.target.provenance["budget_by_task"][name].get("budget_floor", 1)) for name in names},
            "caps": {name: int(self.target.provenance["budget_by_task"][name]["budget_cap"]) for name in names},
            "config": self.target.runtime_config,
            "tasks": self.target.ordered_tasks,
            "feature_names": self.target.feature_names,
            "action_definitions": self.target.action_definitions,
        }

    def quantization_input_state(self, runtime_state: Mapping[str, Any]) -> Mapping[str, Any]:
        return runtime_state

    def extract_observation(self, runtime_state: Mapping[str, Any]):
        return tuple(self._runtime["evaluate"](runtime_state)["observation"])

    def valid_action_mask(self, runtime_state: Mapping[str, Any]):
        from formal_toolchain.adapters.synthetic_policy import mask_and_reasons
        mask, reasons = mask_and_reasons(runtime_state, self.actions, self.target.ordered_tasks)
        return tuple(mask), tuple(reasons)

    def apply_action(self, runtime_state: Mapping[str, Any], action_id: int | None):
        if action_id is None:
            return dict(runtime_state["budgets"])
        return self._runtime["apply"](runtime_state, int(action_id))

    def common_transition_witnesses(self):
        return ()

    def export_observation_contract(self):
        return {"feature_names": list(self.target.feature_names), "total_on_p0_states": True,
                "nan_rejected_or_normalized": True, "inf_rejected_or_normalized": True,
                "clip_before_quantization": True, "history_initialization_defined": True,
                "preclosed_state_read": True}

    def export_mask_contract(self):
        return {"action_ids": list(range(len(self.actions))), "shared_with_step": True,
                "explicit_noop": False, "fallback": "implicit_none_when_no_valid_action"}

    def export_action_contract(self):
        return {"action_ids": [int(action.action_id) for action in self.actions],
                "action_definitions": [dict(item) for item in self.target.action_definitions]}

    def export_source_binding_targets(self):
        return {"adapter_kind": "SYNTHETIC_P0"}

    def export_initial_state_contract(self):
        names = [str(task.name) for task in self.target.ordered_tasks]
        return {
            "status": "PASS",
            "current_time": 0,
            "mode": "LO",
            "running_job": None,
            "active_jobs": [],
            "ready_jobs": [],
            "service_in_progress": False,
            "runtime_budgets": {
                name: int(self.target.provenance["budget_by_task"][name]["initial_runtime_budget"])
                for name in names
            },
            "quiescent": True,
        }

    def export_boot_transition_contract(self):
        initial = self.export_initial_state_contract()
        return {
            "status": "PASS",
            "initial_state_hash": sha256_object(initial),
            "boot_time": 0,
            "mode_after_boot": "LO",
            "first_release_batch_defined": True,
            "no_service_before_boot_closure": True,
            "initial_runtime_budget_snapshot": dict(initial["runtime_budgets"]),
        }
