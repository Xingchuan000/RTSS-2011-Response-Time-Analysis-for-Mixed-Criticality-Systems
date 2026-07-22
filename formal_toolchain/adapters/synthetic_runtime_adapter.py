"""合成 target 的显式 runtime adapter。

这个类只服务于 ``target_kind=SYNTHETIC_P0`` 的测试输入。真实 target 不会
进入本模块；真实 target 必须在自己的 factory 中提供 ``AMCRealRuntimeAdapter``。
"""

from __future__ import annotations

from typing import Any, Mapping

from amc_py.amc import build_design_r_lo_map
from amc_py.rl.actions import build_budget_action_space
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.invariant.safety_polytope import rebuild_expected_rows


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
        budget_info = self.target.provenance["budget_by_task"]
        return {
            "budgets": {name: int(budget_by_task[name]) for name in names},
            "initial_budgets": {
                name: int(budget_info[name]["initial_runtime_budget"])
                for name in names
            },
            "floors": {
                name: int(budget_info[name].get("budget_floor", 1))
                for name in names
            },
            "caps": {
                name: int(budget_info[name]["action_hard_upper"])
                for name in names
            },
            "source_base_budgets": {
                name: int(budget_info[name]["source_base_budget"])
                for name in names
            },
            "config": self.target.runtime_config,
            "tasks": self.target.ordered_tasks,
            "feature_names": self.target.feature_names,
            "action_definitions": self.target.action_definitions,
        }

    def actual_cost_for(self, task: Any, release_index: int) -> int:
        name = str(getattr(task, "name", task))
        budget_info = self.target.provenance["budget_by_task"]
        if name not in budget_info:
            raise KeyError(f"unknown synthetic task: {name}")
        return int(budget_info[name]["initial_runtime_budget"])

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
                "explicit_noop": False, "implicit_noop_when_all_invalid": True,
                "check_safety": True, "fallback": "implicit_none_when_no_valid_action"}

    def export_action_contract(self):
        return {"action_ids": [int(action.action_id) for action in self.actions],
                "action_definitions": [dict(item) for item in self.target.action_definitions]}

    def export_source_binding_targets(self):
        return {"adapter_kind": "SYNTHETIC_P0"}

    def export_initial_state_contract(self):
        names = [str(task.name) for task in self.target.ordered_tasks]
        budget_info = self.target.provenance["budget_by_task"]
        return {
            "status": "PASS",
            "current_time": 0,
            "mode": "LO",
            "running_job": None,
            "active_jobs": [],
            "ready_jobs": [],
            "service_in_progress": False,
            "runtime_budgets": {
                name: int(budget_info[name]["initial_runtime_budget"])
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

    def export_budget_safety_polytope(self) -> dict[str, Any]:
        design_r_lo = build_design_r_lo_map(self.target.ordered_tasks)
        rows = rebuild_expected_rows(self.target.ordered_tasks, design_r_lo=design_r_lo, check_lo_tasks=True)
        return {
            "status": "PASS",
            "schema_version": "budget_safety_polytope_v1",
            "task_order": [str(task.name) for task in self.target.ordered_tasks],
            "rows": rows,
            "design_r_lo": {str(name): int(value) for name, value in design_r_lo.items()},
            "check_lo_tasks": True,
            "candidate_positive_lower": {
                str(task.name): (int(task.c_lo) if task.criticality.value == "HI" else 1)
                for task in self.target.ordered_tasks
            },
            "production_checker_type": "SyntheticP0RuntimeAdapter",
            "check_safety": True,
        }
