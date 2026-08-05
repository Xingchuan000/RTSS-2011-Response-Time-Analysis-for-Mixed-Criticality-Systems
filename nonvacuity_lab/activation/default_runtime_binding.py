"""Small artifact-backed concrete replay used by automatic activation.

This adapter deliberately avoids importing the user's ordinary or overlay
source into the lab process.  It replays the integer tree and the certified
single-budget action contract directly from the resolved artifacts.  That is
sufficient for research negative controls and keeps the replay deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json
import math
from types import SimpleNamespace
from typing import Any


@dataclass
class ControllerState:
    task_budgets: dict[str, int]
    feature_values: dict[int, int]

    def clone(self) -> "ControllerState":
        return ControllerState(dict(self.task_budgets), dict(self.feature_values))


@dataclass
class StepObservation:
    leaf_id: int
    raw_top1_action_id: int
    raw_top1_valid: bool
    selected_action_id: int | None
    action_applied: bool
    post_invariants_hold: bool
    budgets_before: dict[str, int]
    budgets_after: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "leaf_id": self.leaf_id,
            "raw_top1_action_id": self.raw_top1_action_id,
            "raw_top1_valid": self.raw_top1_valid,
            "selected_action_id": self.selected_action_id,
            "action_applied": self.action_applied,
            "post_invariants_hold": self.post_invariants_hold,
            "budgets_before": dict(self.budgets_before),
            "budgets_after": dict(self.budgets_after),
        }


class StateBinding:
    def __init__(self, budgets: dict[str, int]):
        self._budgets = dict(budgets)

    def load_canonical_empty_controller_state(self) -> ControllerState:
        return ControllerState(dict(self._budgets), {})


class ArtifactControllerRuntime:
    def __init__(
        self,
        *,
        tree: dict,
        actions: dict[int, dict],
        task_contracts: dict[str, dict],
        selection_semantics: str,
        apply_unchecked: bool,
        rounding_mode: str,
        disabled_guard: str | None = None,
    ) -> None:
        self.tree = tree
        self.actions = actions
        self.task_contracts = task_contracts
        self.selection_semantics = selection_semantics
        self.apply_unchecked = apply_unchecked
        self.rounding_mode = rounding_mode
        self.disabled_guard = disabled_guard
        self._nodes = _tree_index(tree)

    def controller_step(self, state: ControllerState) -> StepObservation:
        leaf = _evaluate_leaf(self.tree, self._nodes, state.feature_values)
        ranking = [int(item) for item in leaf.get("action_ranking", leaf.get("ranking", []))]
        if not ranking:
            ranking = [int(leaf.get("raw_action_id", leaf.get("action_id")))]
        legal = {action_id: self._candidate(state.task_budgets, action_id)[1] for action_id in ranking}
        raw = ranking[0]
        if self.selection_semantics == "raw_top1":
            selected = raw
        elif self.selection_semantics == "top1_valid_else_noop":
            selected = raw if legal[raw] else None
        elif self.selection_semantics == "all_invalid_force_top1":
            selected = next((aid for aid in ranking if legal[aid]), raw)
        else:
            selected = next((aid for aid in ranking if legal[aid]), None)
        before = dict(state.task_budgets)
        after = dict(before)
        applied = False
        if selected is not None:
            candidate, selected_legal = self._candidate(before, selected)
            if selected_legal or self.apply_unchecked or self.disabled_guard:
                after = candidate
                applied = after != before
        post = self._invariants(after)
        return StepObservation(
            leaf_id=int(leaf.get("leaf_id", leaf.get("node_id", leaf.get("id")))),
            raw_top1_action_id=raw,
            raw_top1_valid=bool(legal[raw]),
            selected_action_id=selected,
            action_applied=applied,
            post_invariants_hold=post,
            budgets_before=before,
            budgets_after=after,
        )

    def _candidate(self, budgets: dict[str, int], action_id: int) -> tuple[dict[str, int], bool]:
        action = self.actions[action_id]
        if bool(action.get("is_noop")):
            return dict(budgets), True
        task = str(action.get("task_id", action.get("target_task", action.get("increase_task") or (action.get("decrease_tasks") or [None])[0])))
        direction = str(action.get("operation", action.get("direction", "increase" if action.get("increase_task") else "decrease")))
        ratio = float(action.get("ratio", action.get("increase_ratio" if direction == "increase" else "decrease_ratio", 0.02)))
        old = int(budgets[task])
        raw = old * (1.0 + ratio if direction == "increase" else 1.0 - ratio)
        if self.rounding_mode == "nearest":
            value = int(round(raw))
        else:
            value = math.ceil(raw) if direction == "increase" else math.floor(raw)
        minimum_delta = int(action.get("minimum_increment", 1))
        value = max(value, old + minimum_delta) if direction == "increase" else min(value, old - minimum_delta)
        candidate = dict(budgets)
        candidate[task] = value
        contract = self.task_contracts[task]
        legal = int(contract["floor"]) <= value <= int(contract["upper"])
        if str(contract["criticality"]) == "HI":
            legal = legal and value >= int(contract["reference"])
        if self.disabled_guard is not None:
            # The selected B4 guard is treated as the only blocking predicate;
            # post-invariant checking remains active below.
            legal = True
        return candidate, legal

    def _invariants(self, budgets: dict[str, int]) -> bool:
        for task, value in budgets.items():
            contract = self.task_contracts[task]
            if value < int(contract["floor"]) or value > int(contract["upper"]):
                return False
            if str(contract["criticality"]) == "HI" and value < int(contract["reference"]):
                return False
        return True


def build_default_runtime_binding(
    *, clean_source_root: Path, overlay_source_root: Path,
    resolved_target: dict, binding: dict,
):
    del clean_source_root, overlay_source_root
    tree = _read(resolved_target["tree_path"])
    action_raw = _read(binding.get("action_definitions_path") or binding["action_definitions"])
    task_raw = _read(binding.get("taskset_path") or binding["taskset"])
    actions = _normalize_actions(action_raw, float(binding.get("default_action_ratio", 0.02)))
    contracts = _normalize_tasks(task_raw)
    initial = {name: int(item["reference"]) for name, item in contracts.items()}
    overlay_semantics = str(binding.get("overlay_semantics", "raw_top1"))
    clean = ArtifactControllerRuntime(
        tree=tree, actions=actions, task_contracts=contracts,
        selection_semantics="ranked_first_valid", apply_unchecked=False,
        rounding_mode="ceil_floor",
    )
    overlay = ArtifactControllerRuntime(
        tree=tree, actions=actions, task_contracts=contracts,
        selection_semantics=overlay_semantics,
        apply_unchecked=bool(binding.get("overlay_unchecked_apply", overlay_semantics in {"raw_top1", "all_invalid_force_top1"})),
        rounding_mode=str(binding.get("overlay_rounding_mode", "ceil_floor")),
        disabled_guard=binding.get("disabled_guard"),
    )
    return SimpleNamespace(
        state_binding=StateBinding(initial), clean_runtime=clean, overlay_runtime=overlay,
    )


def _read(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding="utf-8"))
    return copy.deepcopy(value)


def _normalize_tasks(raw: dict) -> dict[str, dict]:
    tasks = raw.get("tasks", raw.get("ordered_tasks", []))
    result = {}
    for task in tasks:
        name = str(task.get("task_id", task.get("name")))
        reference = int(task.get("reference_budget", task.get("initial_runtime_budget", task.get("code_c_lo", 0))))
        result[name] = {
            "criticality": str(getattr(task["criticality"], "value", task["criticality"])).upper(),
            "reference": reference,
            "floor": int(task.get("minimum_budget", task.get("budget_floor", reference))),
            "upper": int(task.get("certified_upper_bound", task.get("action_hard_upper", task.get("code_c_hi", reference)))),
        }
    return result


def _normalize_actions(raw: Any, default_ratio: float) -> dict[int, dict]:
    if isinstance(raw, dict) and isinstance(raw.get("actions"), list):
        rows = raw["actions"]
    elif isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            row = dict(value)
            row.setdefault("action_id", int(key))
            rows.append(row)
    else:
        raise ValueError("action definitions must be an array or action map")
    result = {}
    for item in rows:
        action = dict(item)
        action.setdefault("ratio", default_ratio)
        result[int(action["action_id"])] = action
    if not result:
        raise ValueError("action definitions are empty")
    return result


def _tree_index(tree: dict) -> dict[int, dict]:
    result = {}
    for item in tree.get("nodes", []):
        result[int(item.get("node_id", item.get("id")))] = item
    for item in tree.get("leaves", []):
        result[int(item.get("node_id", item.get("leaf_id", item.get("id"))))] = item
    return result


def _evaluate_leaf(tree: dict, nodes: dict[int, dict], features: dict[int, int]) -> dict:
    node_id = int(tree.get("root_node_id", tree.get("root_id", tree.get("root", 0))))
    while True:
        node = nodes[node_id]
        if "action_ranking" in node or "leaf_id" in node or "feature_index" not in node:
            return node
        value = int(features.get(int(node["feature_index"]), 0))
        threshold = int(node.get("threshold_int", node.get("threshold")))
        node_id = int(node.get("left_child", node.get("left"))) if value <= threshold else int(node.get("right_child", node.get("right")))
