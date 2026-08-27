"""Symbolic deployed-controller decision for the V9.1 P5 phase."""

from __future__ import annotations

from dataclasses import dataclass

import z3

from .action_encoder import encode_budget_after_selected_action, encode_first_valid_explicit_noop
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


def encode_controller_decision(state: SymbolicKernelState, model: BoundModel) -> ControllerEncoding:
    """Encode observation -> CART -> mask -> ranked FirstValid -> budget update.

    All auxiliary variables are named from the state timestamp symbol so every
    unrolled P5 occurrence remains independent in one solver instance.
    """

    if model.tree is None or model.action_dim <= 0 or model.noop_id is None:
        raise ValueError("V9_1_P5_POLICY_ARTIFACT_UNBOUND")
    if int(model.tree.action_dim) != model.action_dim:
        raise ValueError("V9_1_P5_TREE_ACTION_DIMENSION_MISMATCH")
    if int(model.tree.state_dim) != len(model.feature_names):
        raise ValueError("V9_1_P5_TREE_STATE_DIMENSION_MISMATCH")
    if len(model.action_definitions) != model.action_dim:
        raise ValueError("V9_1_P5_ACTION_ALPHABET_UNBOUND")

    base = str(state.t)
    enabled = (state.t % model.agent_period) == 0
    safety_margin = encode_safety_margin_min(state.budgets, model)
    observation = encode_v11_full_10d_observation(
        state,
        model,
        safety_margin=safety_margin,
        prefix=f"{base}.p5.q",
    )
    tree = encode_tree_leaf_and_ranking(observation.quantized, model.tree, prefix=f"{base}.p5.tree")
    mask, candidates, mask_constraints = encode_action_mask(
        state.budgets, model.action_definitions, model
    )
    selected, selector_constraints = encode_first_valid_explicit_noop(
        tree.ranking,
        mask,
        action_dim=model.action_dim,
        noop_id=model.noop_id,
        name=f"{base}.p5.selected_action",
    )
    budget_after = encode_budget_after_selected_action(
        selected,
        candidates,
        state.budgets,
        action_dim=model.action_dim,
    )
    constraints = tuple(
        list(observation.constraints)
        + list(tree.constraints)
        + list(mask_constraints)
        + list(selector_constraints)
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


__all__ = ["ControllerEncoding", "encode_controller_decision"]
