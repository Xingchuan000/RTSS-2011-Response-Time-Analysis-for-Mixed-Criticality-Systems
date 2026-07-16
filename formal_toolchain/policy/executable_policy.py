"""Phase G05：runtime state 到 deployed action 的可执行语义链。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from amc_py.viper.integer_tree import IntegerTreeModel, evaluate_integer_tree
from amc_py.models import Task
from amc_py.rl.actions import BudgetAction
from .quantization import replay_quantize
from .mask_fallback import select_first_valid
from .actions import replay_action


def replay_deployed_policy(runtime_state: Mapping[str, Any], target: Any,
                           tree: IntegerTreeModel, config: dict[str, Any], *,
                           actions: Sequence[BudgetAction]) -> dict[str, Any]:
    """从 runtime state 完整重放 observation→tree→mask→action。"""
    from formal_toolchain.adapters.synthetic_policy import build_runtime_adapter
    adapter = build_runtime_adapter(target, actions)
    evidence = adapter["evaluate"](runtime_state)
    observation = evidence["observation"]
    valid_mask = evidence["mask"]
    if len(observation) != tree.state_dim or len(valid_mask) != tree.action_dim:
        raise ValueError("runtime observation/mask 维度与 artifact 不一致")
    quantized = tuple(replay_quantize(value, config)[0] for value in observation)
    evaluation = evaluate_integer_tree(tree, quantized)
    selected = select_first_valid(evaluation.action_ranking, valid_mask, action_dim=tree.action_dim)
    after = None
    if selected is not None:
        after = adapter["apply"](runtime_state, selected)
    return {"status": "PASS", "quantized": quantized, "leaf_id": evaluation.leaf_id,
            "ranking": evaluation.action_ranking, "selected_action": selected,
            "mask": valid_mask, "mask_reasons": evidence["reasons"],
            "implicit_noop": selected is None, "budget_after": after}
