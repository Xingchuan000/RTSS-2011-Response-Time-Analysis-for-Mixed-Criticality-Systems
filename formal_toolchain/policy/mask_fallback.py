"""Phase G04：mask、ranked first-valid 与 implicit noop。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def evaluate_synthetic_mask(state: Mapping[str, Any], action_definitions: Sequence[Mapping[str, Any]],
                            *, forbid_decreasing_hi_budgets: bool = True) -> tuple[tuple[bool, ...], tuple[str, ...]]:
    """按 synthetic action schema 独立重算 mask/reason，不消费调用方 mask。"""
    budgets = state.get("budgets")
    criticality = state.get("criticality")
    if not isinstance(budgets, Mapping) or not isinstance(criticality, Mapping):
        raise ValueError("synthetic mask state 缺少 budgets/criticality")
    mask: list[bool] = []; reasons: list[str] = []
    for definition in action_definitions:
        task = definition.get("target_task") or definition.get("task_name")
        direction = definition.get("direction", "increase")
        valid = isinstance(task, str) and task in budgets
        reason = "accepted"
        if not valid:
            reason = "unknown_target"
        elif direction == "decrease" and forbid_decreasing_hi_budgets and criticality.get(task) == "HI":
            valid = False; reason = "hi_decrease_guard"
        elif direction == "decrease" and int(budgets[task]) <= int(state.get("floor", {}).get(task, 1)):
            valid = False; reason = "budget_floor"
        mask.append(valid); reasons.append(reason)
    return tuple(mask), tuple(reasons)


def select_first_valid(ranking: Sequence[int], valid_mask: Sequence[bool], *, action_dim: int) -> int | None:
    if len(valid_mask) != action_dim:
        raise ValueError("mask length 必须等于 action_dim")
    if len(ranking) != action_dim or tuple(sorted(ranking)) != tuple(range(action_dim)):
        raise ValueError("ranking 不完整或含越界 action；这属于 artifact invalid")
    for action_id in ranking:
        if bool(valid_mask[action_id]):
            return int(action_id)
    return None


def build_mask_fallback_certificate(rankings: Sequence[Sequence[int]], masks: Sequence[Sequence[bool]], *, action_dim: int,
                                    runtime_reasons: Sequence[Sequence[str]] | None = None,
                                    synthetic_cases: Sequence[Mapping[str, Any]] | None = None,
                                    runtime_mask_evaluator: Any | None = None) -> dict[str, Any]:
    if (runtime_reasons is None or synthetic_cases is None or runtime_mask_evaluator is None or
        len(runtime_reasons) != len(masks) or len(synthetic_cases) != len(masks)):
        return {"status": "UNRESOLVED", "route": "POLICY_CONTRACT_VIOLATION",
                "failure": {"code": "MASK_REJECT_REASON_EVIDENCE_MISSING"}}
    if any(len(mask) != action_dim or len(reason) != action_dim for mask, reason in zip(masks, runtime_reasons)):
        return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                "failure": {"code": "MASK_REASON_LENGTH_MISMATCH"}}
    selected = [select_first_valid(ranking, mask, action_dim=action_dim) for ranking, mask in zip(rankings, masks)]
    for case, mask, reasons in zip(synthetic_cases, masks, runtime_reasons):
        runtime_mask, runtime_reason = runtime_mask_evaluator(case["state"], case["action_definitions"],
                                                              case.get("forbid_decreasing_hi_budgets", True))
        if tuple(mask) != tuple(runtime_mask) or tuple(reasons) != tuple(runtime_reason):
            return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {"code": "RUNTIME_MASK_EVIDENCE_MISMATCH"}}
        formal_mask, formal_reasons = evaluate_synthetic_mask(case["state"], case["action_definitions"],
                                                               forbid_decreasing_hi_budgets=case.get("forbid_decreasing_hi_budgets", True))
        if tuple(mask) != formal_mask or tuple(reasons) != formal_reasons:
            return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {"code": "MASK_REASON_DIFFERENTIAL_MISMATCH"}}
    return {"status": "PASS", "schema_version": "mask_fallback_v1", "action_dim": action_dim,
            "cases": [{"ranking": list(r), "mask": list(m), "selected_action": a, "reject_reasons": list(reason)}
                      for r, m, a, reason in zip(rankings, masks, selected, runtime_reasons)],
            "implicit_noop": any(item is None for item in selected), "selection": "first_valid"}
