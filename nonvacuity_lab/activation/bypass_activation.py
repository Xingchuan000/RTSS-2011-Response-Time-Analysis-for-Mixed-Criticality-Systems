from __future__ import annotations


def evaluate_bypass_activation(pair_receipt, witness, base_runtime, overlay_runtime) -> dict:
    base = base_runtime.replay_single_controller_decision(witness["state"])
    mutated = overlay_runtime.replay_single_controller_decision(witness["state"])
    activated = (
        base.raw_top1_action_id == pair_receipt["action_id"]
        and base.raw_top1_valid is False
        and mutated.selected_action_id == pair_receipt["action_id"]
        and mutated.action_applied is True
        and mutated.post_invariants_hold is False
    )
    return {"activated": activated, "base": base.to_json(), "mutated": mutated.to_json()}
