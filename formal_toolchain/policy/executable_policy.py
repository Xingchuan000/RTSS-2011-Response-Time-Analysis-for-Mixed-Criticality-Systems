"""Phase G05：runtime state 到 deployed action 的可执行语义链。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from amc_py.viper.integer_tree import IntegerTreeModel, evaluate_integer_tree
from amc_py.models import Task
from amc_py.rl.actions import BudgetAction
from .quantization import replay_quantize
from .mask_fallback import select_by_semantics
from .actions import replay_action


def replay_deployed_policy(runtime_state: Mapping[str, Any], target: Any,
                           tree: IntegerTreeModel, config: dict[str, Any], *,
                           actions: Sequence[BudgetAction]) -> dict[str, Any]:
    """从 runtime state 完整重放 observation→tree→mask→action。"""
    adapter = target.runtime_adapter
    if adapter is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "FORMAL_RUNTIME_ADAPTER_MISSING"}}
    evidence = {
        "observation": tuple(adapter.extract_observation(runtime_state)),
        "mask": tuple(adapter.valid_action_mask(runtime_state)[0]),
        "reasons": tuple(adapter.valid_action_mask(runtime_state)[1]),
    }
    observation = evidence["observation"]
    valid_mask = evidence["mask"]
    if len(observation) != tree.state_dim or len(valid_mask) != tree.action_dim:
        raise ValueError("runtime observation/mask 维度与 artifact 不一致")
    quantized = tuple(replay_quantize(value, config)[0] for value in observation)
    evaluation = evaluate_integer_tree(tree, quantized)
    mask_contract = adapter.export_mask_contract()
    selection_semantics = str(mask_contract.get("selection", "ranked_first_valid"))
    noop_ids = tuple(int(value) for value in mask_contract.get("explicit_noop_action_ids", ()))
    explicit_noop_action_id = noop_ids[0] if mask_contract.get("explicit_noop") and len(noop_ids) == 1 else None
    selected = select_by_semantics(
        evaluation.action_ranking, valid_mask, action_dim=tree.action_dim,
        selection_semantics=selection_semantics,
        explicit_noop_action_id=explicit_noop_action_id,
    )
    after = None
    if selected is not None:
            after = adapter.apply_action(runtime_state, selected)
    return {"status": "PASS", "quantized": quantized, "leaf_id": evaluation.leaf_id,
            "ranking": evaluation.action_ranking, "selected_action": selected,
            "mask": valid_mask, "mask_reasons": evidence["reasons"],
            "implicit_noop": selected is None, "budget_after": after}
