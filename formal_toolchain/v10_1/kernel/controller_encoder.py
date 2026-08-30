"""Symbolic deployed-controller decision for the V10.1 P5 phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import z3

from .action_encoder import encode_budget_after_selected_action, encode_first_valid_leaf_cases
from .mask_encoder import encode_action_mask, encode_safety_margin_min
from .numeric_encoder import NumericEncoding, encode_v11_full_10d_observation
from .symbolic_state import BoundModel, SymbolicKernelState
from .tree_encoder import TreeEncoding, encode_tree_leaf_and_ranking


@dataclass(frozen=True, slots=True)
class ControllerEncoding:
    enabled: z3.BoolRef
    observation: NumericEncoding
    tree: TreeEncoding
    mask: tuple[z3.BoolRef, ...]
    candidates: tuple[dict[str, z3.ArithRef], ...]
    selected_action: z3.ArithRef
    budget_after: dict[str, z3.ArithRef]
    constraints: tuple[z3.BoolRef, ...]


@dataclass(frozen=True, slots=True)
class ControllerPolicyCase:
    leaf_id: int
    selected_action: int

    @property
    def case_id(self) -> str:
        return f"LEAF_{self.leaf_id}__ACTION_{self.selected_action}"


@dataclass(frozen=True, slots=True)
class ControllerCaseEncoding:
    case: ControllerPolicyCase
    budget_after: dict[str, z3.ArithRef]
    constraints: tuple[z3.BoolRef, ...]


def _tree_maps(tree: Any):
    return (
        {int(node.node_id): node for node in tree.nodes},
        {int(leaf.node_id): leaf for leaf in tree.leaves},
    )


def _leaf_paths(tree: Any) -> dict[int, tuple[tuple[int, int, bool], ...]]:
    nodes, leaves = _tree_maps(tree)
    result: dict[int, tuple[tuple[int, int, bool], ...]] = {}

    def visit(node_id: int, path: tuple[tuple[int, int, bool], ...]) -> None:
        if node_id in leaves:
            result[node_id] = path
            return
        node = nodes[node_id]
        step = (int(node.feature_index), int(node.threshold_int))
        visit(int(node.left_child), path + ((step[0], step[1], True),))
        visit(int(node.right_child), path + ((step[0], step[1], False),))

    visit(int(tree.root_node_id), ())
    return result


def enumerate_controller_policy_cases(model: BoundModel) -> tuple[ControllerPolicyCase, ...]:
    if model.tree is None or model.noop_id is None:
        raise ValueError("V10_1_CONTROLLER_POLICY_UNBOUND")
    _, leaves = _tree_maps(model.tree)
    rows: list[ControllerPolicyCase] = []
    for leaf_id in sorted(leaves):
        ranking = tuple(int(value) for value in leaves[leaf_id].action_ranking)
        for action in ranking:
            rows.append(ControllerPolicyCase(leaf_id, action))
            if action == int(model.noop_id):
                break
    return tuple(rows)


def encode_controller_policy_case(
    state: SymbolicKernelState,
    model: BoundModel,
    case: ControllerPolicyCase,
) -> ControllerCaseEncoding:
    """Exact deployed controller restricted to one leaf/FirstValid outcome."""

    if model.tree is None or model.noop_id is None:
        raise ValueError("V10_1_CONTROLLER_POLICY_UNBOUND")
    paths = _leaf_paths(model.tree)
    _, leaves = _tree_maps(model.tree)
    if case.leaf_id not in paths:
        raise ValueError("V10_1_CONTROLLER_CASE_LEAF_UNKNOWN")
    leaf = leaves[case.leaf_id]
    ranking = tuple(int(value) for value in leaf.action_ranking)
    if case.selected_action not in ranking:
        raise ValueError("V10_1_CONTROLLER_CASE_ACTION_NOT_RANKED")
    selected_pos = ranking.index(case.selected_action)
    noop_pos = ranking.index(int(model.noop_id))
    if selected_pos > noop_pos:
        raise ValueError("V10_1_CONTROLLER_CASE_AFTER_NOOP")

    path = paths[case.leaf_id]
    used_features = tuple(sorted({index for index, _, _ in path}))
    safety_margin_index = 10 * len(model.tasks) + 7
    safety_margin = (
        encode_safety_margin_min(state.budgets, model)
        if safety_margin_index in used_features else z3.RealVal(1)
    )
    base = str(state.t)
    observation = encode_v11_full_10d_observation(
        state,
        model,
        safety_margin=safety_margin,
        prefix=f"{base}.p5.case.{case.leaf_id}.q",
        active_feature_indices=used_features,
    )
    path_constraints: list[z3.BoolRef] = []
    for index, threshold, goes_left in path:
        predicate = observation.quantized[index] <= int(threshold)
        path_constraints.append(predicate if goes_left else z3.Not(predicate))

    mask, candidates, mask_constraints = encode_action_mask(
        state.budgets, model.action_definitions, model
    )
    first_valid = [mask[case.selected_action]]
    first_valid.extend(z3.Not(mask[action]) for action in ranking[:selected_pos])
    constraints = tuple(
        list(observation.constraints)
        + path_constraints
        + list(mask_constraints)
        + first_valid
    )
    return ControllerCaseEncoding(
        case=case,
        budget_after=dict(candidates[case.selected_action]),
        constraints=constraints,
    )


def encode_controller_decision(state: SymbolicKernelState, model: BoundModel) -> ControllerEncoding:
    """Encode observation -> CART -> mask -> ranked FirstValid -> budget update.

    All auxiliary variables are named from the state timestamp symbol so every
    unrolled P5 occurrence remains independent in one solver instance.
    """

    if model.tree is None or model.action_dim <= 0 or model.noop_id is None:
        raise ValueError("V10_1_P5_POLICY_ARTIFACT_UNBOUND")
    if int(model.tree.action_dim) != model.action_dim:
        raise ValueError("V10_1_P5_TREE_ACTION_DIMENSION_MISMATCH")
    if int(model.tree.state_dim) != len(model.feature_names):
        raise ValueError("V10_1_P5_TREE_STATE_DIMENSION_MISMATCH")
    if len(model.action_definitions) != model.action_dim:
        raise ValueError("V10_1_P5_ACTION_ALPHABET_UNBOUND")

    base = str(state.t)
    enabled = (state.t % model.agent_period) == 0
    used_features = tuple(sorted({int(node.feature_index) for node in model.tree.nodes}))
    safety_margin_index = 10 * len(model.tasks) + 7
    safety_margin = (
        encode_safety_margin_min(state.budgets, model)
        if safety_margin_index in used_features
        else z3.RealVal(1)
    )
    observation = encode_v11_full_10d_observation(
        state,
        model,
        safety_margin=safety_margin,
        prefix=f"{base}.p5.q",
        active_feature_indices=used_features,
    )
    tree = encode_tree_leaf_and_ranking(observation.quantized, model.tree, prefix=f"{base}.p5.tree")
    mask, candidates, mask_constraints = encode_action_mask(
        state.budgets, model.action_definitions, model
    )
    selected, selector_constraints = encode_first_valid_leaf_cases(
        tree.leaf_cases,
        mask,
        action_dim=model.action_dim,
        noop_id=model.noop_id,
        name=f"{base}.p5.selected_action",
    )
    budget_after, budget_update_constraints = encode_budget_after_selected_action(
        selected,
        candidates,
        state.budgets,
        action_dim=model.action_dim,
        prefix=f"{base}.p5.budget_after",
    )
    constraints = tuple(
        list(observation.constraints)
        + list(tree.constraints)
        + list(mask_constraints)
        + list(selector_constraints)
        + list(budget_update_constraints)
    )
    return ControllerEncoding(
        enabled=enabled,
        observation=observation,
        tree=tree,
        mask=mask,
        candidates=candidates,
        selected_action=selected,
        budget_after=budget_after,
        constraints=constraints,
    )


__all__ = [
    "ControllerCaseEncoding", "ControllerEncoding", "ControllerPolicyCase",
    "encode_controller_decision", "encode_controller_policy_case",
    "enumerate_controller_policy_cases",
]
