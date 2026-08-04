from __future__ import annotations


class ActivationError(ValueError):
    pass


def require_all_invalid_witness(witness, ordinary_runtime, overlay_runtime):
    base = ordinary_runtime.replay_single_controller_decision(witness["state"])
    mutated = overlay_runtime.replay_single_controller_decision(witness["state"])
    if not base.all_invalid:
        raise ActivationError("witness is not an all-invalid state")
    if not base.implicit_noop:
        raise ActivationError("ordinary policy did not take implicit noop")
    if mutated.selected_action_id != base.raw_top1_action_id:
        raise ActivationError("mutation did not force raw top-1")
    if not mutated.action_applied:
        raise ActivationError("forced action was not applied")
    return {"activated": True, "base": base.to_json(), "mutated": mutated.to_json()}
