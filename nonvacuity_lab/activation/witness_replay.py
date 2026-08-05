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
    if hasattr(state, "feature_values"):
        state.feature_values.update({int(key): int(value) for key, value in model_json.get("features", {}).items()})
    return state


def replay_symbolic_witness(model_json, *, binding, clean_runtime, overlay_runtime, expected):
    state = materialize_runtime_state(model_json, binding)
    base = clean_runtime.controller_step(state.clone())
    mutated = overlay_runtime.controller_step(state.clone())
    formula = expected.get("formula_kind")
    checks = {
        "leaf_matches": base.leaf_id == expected["leaf_id"],
        "raw_action_matches": base.raw_top1_action_id == expected["action_id"],
    }
    if formula in {"A_MASK_REJECT", "B2_NO_FIRST_VALID_DIFFERENCE", "B_RAW_TOP1_BREAKS_INVARIANT", "B3_ALL_INVALID", "B4_GUARD_NECESSITY"}:
        checks["base_legality_matches"] = base.raw_top1_valid is False
    if formula == "A_MASK_REJECT":
        checks.update({
            "clean_avoids_raw": base.selected_action_id != expected["action_id"],
            "overlay_noops_on_invalid_top1": mutated.selected_action_id is None,
        })
    elif formula == "B2_NO_FIRST_VALID_DIFFERENCE":
        checks.update({
            "clean_selects_lower_valid": (
                base.selected_action_id is not None
                and base.selected_action_id != expected["action_id"]
            ),
            "overlay_noops_on_invalid_top1": mutated.selected_action_id is None,
            "selection_differs": base.selected_action_id != mutated.selected_action_id,
        })
    elif formula == "B_RAW_TOP1_BREAKS_INVARIANT":
        checks.update({
            "mutated_selects_raw": mutated.selected_action_id == expected["action_id"],
            "mutated_applies_raw": mutated.action_applied is True,
            "mutated_breaks_invariant": mutated.post_invariants_hold is False,
        })
    elif formula == "B3_ALL_INVALID":
        checks.update({
            "mutated_forces_action": mutated.selected_action_id is not None,
            "mutated_applies_action": mutated.action_applied is True,
        })
    elif formula == "B4_GUARD_NECESSITY":
        checks.update({
            "mutated_guard_removed": mutated.action_applied is True,
            "mutated_breaks_invariant": mutated.post_invariants_hold is False,
        })
    elif formula == "C2_ROUNDING_DIFFERENCE":
        checks["rounding_changes_budget"] = base.budgets_after != mutated.budgets_after
    activated = all(checks.values())
    return {
        "status": "MATCHED" if activated else "ACTIVATION_MODEL_RUNTIME_MISMATCH",
        "activated": activated,
        "checks": checks,
        "base_observation": base.to_json(),
        "mutated_observation": mutated.to_json(),
    }
