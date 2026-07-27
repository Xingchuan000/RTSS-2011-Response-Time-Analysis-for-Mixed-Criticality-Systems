"""Evaluate a concrete leaf/action witness without mutation-aware checkers."""

from __future__ import annotations

from typing import Any, Mapping


def evaluate_policy_witness(witness: Mapping[str, Any]) -> dict[str, Any]:
    state = _number_map(witness.get("state", {}), "state")
    lower = _number_map(witness.get("lower", {}), "lower")
    upper = _number_map(witness.get("upper", {}), "upper")
    ranking = [int(item) for item in witness.get("ranking", ())]
    actions = witness.get("actions", {})
    if not state or not ranking or not isinstance(actions, Mapping):
        raise ValueError("witness 必须包含 state/ranking/actions")
    guard_satisfied = all(_constraint_holds(state, item) for item in witness.get("leaf_guard", ()))
    baseline_validity: dict[int, bool] = {}
    post_states: dict[int, dict[str, float]] = {}
    reject_reasons: dict[int, list[str]] = {}
    for action_id in ranking:
        raw_action = actions.get(str(action_id), actions.get(action_id))
        if not isinstance(raw_action, Mapping):
            raise ValueError(f"缺少 action definition: {action_id}")
        post = dict(state)
        for key, delta in _number_map(raw_action.get("delta", {}), "delta").items():
            post[key] = post.get(key, 0.0) + delta
        post_states[action_id] = post
        reasons = _invariant_reasons(post, lower, upper)
        for guard in raw_action.get("guards", ()):
            if not _constraint_holds(state, guard):
                reasons.append(str(guard.get("name", "ACTION_GUARD")))
        reject_reasons[action_id] = reasons
        baseline_validity[action_id] = not reasons
    first_valid = next((action for action in ranking if baseline_validity[action]), None)
    top1 = ranking[0]
    all_invalid = not any(baseline_validity.values())
    disabled_guard = witness.get("disabled_guard")
    guard_ablation_valid = None
    other_rejects_after_ablation: list[str] = []
    if disabled_guard is not None:
        other_rejects_after_ablation = [
            reason for reason in reject_reasons[top1] if reason != str(disabled_guard)
        ]
        guard_ablation_valid = not other_rejects_after_ablation
    return {
        "schema_version": "policy_witness_v1",
        "leaf_id": _optional_int(witness.get("leaf_id")),
        "action_id": top1,
        "leaf_guard_satisfied": guard_satisfied,
        "baseline_top1_valid": baseline_validity[top1],
        "baseline_reject_reasons": reject_reasons[top1],
        "baseline_first_valid_action": first_valid,
        "baseline_selected_rank": (
            ranking.index(first_valid) if first_valid is not None else None
        ),
        "all_invalid": all_invalid,
        "mutated_raw_top1_action": top1,
        "mutated_post_state": post_states[top1],
        "mutated_post_invariant_violation": bool(
            _invariant_reasons(post_states[top1], lower, upper)
        ),
        "guard_ablation_valid": guard_ablation_valid,
        "other_rejects_after_ablation": other_rejects_after_ablation,
        "activated_b1": (
            guard_satisfied
            and not baseline_validity[top1]
            and first_valid is not None
            and bool(_invariant_reasons(post_states[top1], lower, upper))
        ),
        "activated_b3": guard_satisfied and all_invalid,
        "activated_b4": (
            guard_satisfied
            and disabled_guard in reject_reasons[top1]
            and guard_ablation_valid is True
        ),
    }


def _constraint_holds(state: Mapping[str, float], constraint: Mapping[str, Any]) -> bool:
    field = str(constraint["field"])
    value = state[field]
    target = float(constraint["value"])
    op = str(constraint["op"])
    return {
        "<=": value <= target,
        "<": value < target,
        ">=": value >= target,
        ">": value > target,
        "==": value == target,
        "!=": value != target,
    }.get(op, False)


def _invariant_reasons(
    state: Mapping[str, float],
    lower: Mapping[str, float],
    upper: Mapping[str, float],
) -> list[str]:
    result = []
    for key, value in state.items():
        if key in lower and value < lower[key]:
            result.append(f"{key}:LOWER")
        if key in upper and value > upper[key]:
            result.append(f"{key}:UPPER")
    return result


def _number_map(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必须为 object")
    return {str(key): float(item) for key, item in value.items()}


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
