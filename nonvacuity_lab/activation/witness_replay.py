from __future__ import annotations


class ActivationReplayError(ValueError):
    pass


def materialize_runtime_state(model_json: dict, binding):
    factory = getattr(binding, "load_canonical_empty_controller_state", None)
    if factory is None and isinstance(binding, dict):
        factory = binding.get("load_canonical_empty_controller_state")
    if not callable(factory):
        raise ActivationReplayError("binding must provide canonical runtime state factory")
    state = factory()
    for task_id, value in model_json.get("budgets", {}).items():
        state.task_budgets[task_id] = int(value)
    return state


def replay_symbolic_witness(model_json, *, binding, clean_runtime, overlay_runtime, expected):
    state = materialize_runtime_state(model_json, binding)
    base = clean_runtime.controller_step(state.clone())
    mutated = overlay_runtime.controller_step(state.clone())
    checks = {
        "leaf_matches": base.leaf_id == expected["leaf_id"],
        "raw_action_matches": base.raw_top1_action_id == expected["action_id"],
        "base_legality_matches": base.raw_top1_valid is False,
    }
    if expected.get("formula_kind") == "B_RAW_TOP1_BREAKS_INVARIANT":
        checks.update({
            "mutated_selects_raw": mutated.selected_action_id == expected["action_id"],
            "mutated_applies_raw": mutated.action_applied is True,
            "mutated_breaks_invariant": mutated.post_invariants_hold is False,
        })
    activated = all(checks.values())
    return {
        "status": "MATCHED" if activated else "ACTIVATION_MODEL_RUNTIME_MISMATCH",
        "activated": activated,
        "checks": checks,
        "base_observation": base.to_json(),
        "mutated_observation": mutated.to_json(),
    }
