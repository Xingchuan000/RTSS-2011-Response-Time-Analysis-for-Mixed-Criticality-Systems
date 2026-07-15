"""target、feature、action 三种顺序的严格一致性检查。"""

from __future__ import annotations

from typing import Any

from formal_toolchain.adapters.amc_taskset import derive_action_task_order, derive_feature_task_order, export_taskset
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact


def check_order_consistency(ordered_tasks: list[str], feature_task_order: list[str],
                            action_task_order: list[str]) -> dict[str, Any]:
    if ordered_tasks != feature_task_order or ordered_tasks != action_task_order:
        return {"obligation_status": "FAIL", "failure": {
            "code": "FEATURE_TASK_PRIORITY_ORDER_MISMATCH",
            "route": "MODEL_CONFORMANCE_FAILED",
            "ordered_tasks": ordered_tasks,
            "feature_task_order": feature_task_order,
            "action_task_order": action_task_order,
        }}
    return {"obligation_status": "PASS"}


def preflight_formal_target(target: Any, artifact_dir) -> dict[str, Any]:
    """严格比较实际 target、feature、action 三种顺序和基础 fingerprint。"""
    expected_state_dim = len(target.feature_names)
    expected_action_dim = len(target.action_definitions)
    inventory = inspect_tree_artifact(artifact_dir, expected_state_dim=expected_state_dim,
                                      expected_action_dim=expected_action_dim,
                                      expected_seed=None)
    taskset = export_taskset(target.ordered_tasks, target.provenance.get("budget_by_task"))
    feature_order = derive_feature_task_order(target.feature_names)
    action_order = derive_action_task_order(target.action_definitions)
    ordered = [task["name"] for task in taskset["ordered_tasks"]]
    result = check_order_consistency(ordered, feature_order, action_order)
    result.update({"taskset": taskset, "feature_task_order": feature_order,
                   "action_task_order": action_order, "inventory": inventory})
    if result["obligation_status"] != "PASS":
        return result
    if inventory.get("feature_names") != list(target.feature_names):
        return {"obligation_status": "FAIL", "failure": {"code": "ARTIFACT_TARGET_FEATURE_MISMATCH",
                "route": "MODEL_CONFORMANCE_FAILED"}, **result}
    if inventory.get("action_definitions") != list(target.action_definitions):
        return {"obligation_status": "FAIL", "failure": {"code": "ARTIFACT_TARGET_ACTION_MISMATCH",
                "route": "MODEL_CONFORMANCE_FAILED"}, **result}
    if inventory.get("metadata_taskset_seed") is not None and target.provenance.get("taskset_seed") not in (None, inventory["metadata_taskset_seed"]):
        return {"obligation_status": "FAIL", "failure": {"code": "TASKSET_FINGERPRINT_MISMATCH", "route": "MODEL_CONFORMANCE_FAILED"}, **result}
    return result
