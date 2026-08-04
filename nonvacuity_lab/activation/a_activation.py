from __future__ import annotations


def evaluate_mask_contained_activation(witness: dict, *, ordinary_runtime, mutated_tree_path) -> dict:
    observation = ordinary_runtime.replay_single_controller_decision(state=witness["state"], tree_path=mutated_tree_path)
    activated = (
        observation.leaf_id == witness["leaf_id"]
        and observation.raw_top1_action_id == witness["action_id"]
        and observation.raw_top1_valid is False
        and ((observation.selected_rank is not None and observation.selected_rank > 1) or observation.implicit_noop)
    )
    return {"activated": activated, "observation": observation.to_json()}
