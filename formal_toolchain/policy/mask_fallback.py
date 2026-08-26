"""Phase G04：mask、ranked first-valid 与 implicit noop。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from formal_toolchain.core.hashing import sha256_object


def evaluate_synthetic_mask(state: Mapping[str, Any], action_definitions: Sequence[Mapping[str, Any]],
                            *, forbid_decreasing_hi_budgets: bool = True) -> tuple[tuple[bool, ...], tuple[str, ...]]:
    """按 synthetic action schema 独立重算 mask/reason，不消费调用方 mask。"""
    budgets = state.get("budgets")
    criticality = state.get("criticality")
    if not isinstance(budgets, Mapping) or not isinstance(criticality, Mapping):
        raise ValueError("synthetic mask state 缺少 budgets/criticality")
    floors = state.get("floors", state.get("floor", {}))
    caps = state.get("caps", {})
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
        elif direction == "increase" and task in caps and int(budgets[task]) >= int(caps[task]):
            valid = False; reason = "budget_upper_bound"
        elif direction == "decrease" and int(budgets[task]) <= int(floors.get(task, 1)):
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



def select_by_semantics(ranking: Sequence[int], valid_mask: Sequence[bool], *, action_dim: int,
                        selection_semantics: str = "ranked_first_valid",
                        explicit_noop_action_id: int | None = None) -> int | None:
    if len(valid_mask) != action_dim:
        raise ValueError("mask length 必须等于 action_dim")
    if len(ranking) != action_dim or tuple(sorted(ranking)) != tuple(range(action_dim)):
        raise ValueError("ranking 不完整或含越界 action；这属于 artifact invalid")
    raw_top1 = int(ranking[0])
    if explicit_noop_action_id is not None and not (0 <= explicit_noop_action_id < action_dim):
        raise ValueError("explicit_noop_action_id 越界")
    if selection_semantics == "ranked_first_valid":
        first = select_first_valid(ranking, valid_mask, action_dim=action_dim)
        # V7 A8: if the mask contract is malformed and even the total explicit
        # noop is reported invalid, fail operationally to the same action id
        # rather than reintroducing legacy implicit ``None`` semantics.
        return explicit_noop_action_id if first is None and explicit_noop_action_id is not None else first
    if selection_semantics == "raw_top1":
        return raw_top1
    if selection_semantics == "top1_valid_else_noop":
        return raw_top1 if bool(valid_mask[raw_top1]) else None
    if selection_semantics == "all_invalid_force_top1":
        first = select_first_valid(ranking, valid_mask, action_dim=action_dim)
        return raw_top1 if first is None else first
    raise ValueError("UNSUPPORTED_POLICY_SELECTION_SEMANTICS")

def build_mask_fallback_certificate(rankings: Sequence[Sequence[int]], masks: Sequence[Sequence[bool]], *, action_dim: int,
                                    runtime_reasons: Sequence[Sequence[str]] | None = None,
                                    synthetic_cases: Sequence[Mapping[str, Any]] | None = None,
                                    runtime_mask_evaluator: Any | None = None,
                                    formal_mask_evaluator: Any | None = None) -> dict[str, Any]:
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
        if formal_mask_evaluator is None:
            formal_mask, formal_reasons = evaluate_synthetic_mask(
                case["state"], case["action_definitions"],
                forbid_decreasing_hi_budgets=case.get("forbid_decreasing_hi_budgets", True))
        else:
            formal_mask, formal_reasons = formal_mask_evaluator(
                case["state"], case["action_definitions"],
                case.get("forbid_decreasing_hi_budgets", True))
        if tuple(mask) != formal_mask or tuple(reasons) != formal_reasons:
            return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {"code": "MASK_REASON_DIFFERENTIAL_MISMATCH"}}
    return {"status": "PASS", "schema_version": "mask_fallback_v1", "action_dim": action_dim,
            "cases": [{"ranking": list(r), "mask": list(m), "selected_action": a, "reject_reasons": list(reason)}
                      for r, m, a, reason in zip(rankings, masks, selected, runtime_reasons)],
            "implicit_noop": any(item is None for item in selected), "selection": "first_valid"}


def build_parametric_mask_fallback_certificate(
    *,
    rankings: Mapping[int, Sequence[int]],
    action_dim: int,
    mask_contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected_actions = tuple(range(action_dim))
    leaves: list[dict[str, Any]] = []

    if mask_contract.get("shared_with_step") is not True:
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "failure": {"code": "MASK_STEP_SHARED_SEMANTICS_UNVERIFIED"},
        }

    selection_semantics = str(mask_contract.get("selection", "ranked_first_valid"))
    if selection_semantics not in {
        "ranked_first_valid", "raw_top1", "top1_valid_else_noop",
        "all_invalid_force_top1",
    }:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "failure": {"code": "UNSUPPORTED_POLICY_SELECTION_SEMANTICS"}}
    explicit_noop = bool(mask_contract.get("explicit_noop", False))
    noop_ids = tuple(int(value) for value in mask_contract.get("explicit_noop_action_ids", ()))
    # V7 A6/A7: an explicit noop is a normal ranked candidate.  Certification
    # must therefore use ranked first-valid semantics; top1-only variants can
    # skip a valid explicit noop that appears before later budget actions.
    if explicit_noop and selection_semantics != "ranked_first_valid":
        return {
            "status": "FAIL",
            "route": "MODEL_CONFORMANCE_FAILED",
            "failure": {
                "code": "EXPLICIT_NOOP_REQUIRES_RANKED_FIRST_VALID",
                "selection_semantics": selection_semantics,
            },
        }
    if explicit_noop:
        if len(noop_ids) != 1 or mask_contract.get("explicit_noop_always_valid") is not True:
            return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                    "failure": {"code": "EXPLICIT_NOOP_IDENTITY_UNRESOLVED"}}
        if noop_ids[0] not in expected_actions:
            return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                    "failure": {"code": "EXPLICIT_NOOP_ACTION_ID_INVALID"}}
        if (
            int(mask_contract.get("defensive_fallback_action_id", -1)) != noop_ids[0]
            or mask_contract.get("defensive_fallback_same_explicit_noop") is not True
        ):
            return {
                "status": "FAIL",
                "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "EXPLICIT_NOOP_DEFENSIVE_FALLBACK_UNRESOLVED"},
            }

    for leaf_id, ranking_value in sorted(rankings.items()):
        ranking = tuple(int(value) for value in ranking_value)
        if tuple(sorted(ranking)) != expected_actions:
            return {
                "status": "FAIL",
                "route": "PROOF_BUNDLE_INVALID",
                "failure": {
                    "code": "RANKING_NOT_COMPLETE",
                    "leaf_id": int(leaf_id),
                },
            }

        regions = []
        if selection_semantics == "ranked_first_valid":
            for position, action_id in enumerate(ranking):
                regions.append({"rank_position": position, "selected_action": int(action_id),
                                "predicate": {"selected_valid": int(action_id), "preceding_invalid": list(ranking[:position])}})
                if explicit_noop and action_id == noop_ids[0]:
                    break
            if not explicit_noop:
                regions.append({"rank_position": len(ranking), "selected_action": None, "predicate": {"all_invalid": list(ranking)}})
            coverage_rule = "first_true_including_total_explicit_noop" if explicit_noop else "first_true_or_none"
        elif selection_semantics == "raw_top1":
            regions.append({"rank_position": 0, "selected_action": int(ranking[0]),
                            "predicate": {"unconditional_raw_top1": int(ranking[0])}})
            coverage_rule = "raw_top1_unconditional"
        elif selection_semantics == "top1_valid_else_noop":
            regions.append({"rank_position": 0, "selected_action": int(ranking[0]),
                            "predicate": {"selected_valid": int(ranking[0])}})
            regions.append({"rank_position": action_dim, "selected_action": None,
                            "predicate": {"selected_invalid": int(ranking[0])}})
            coverage_rule = "top1_valid_else_none"
        else:
            for position, action_id in enumerate(ranking):
                regions.append({"rank_position": position, "selected_action": int(action_id),
                                "predicate": {"selected_valid": int(action_id), "preceding_invalid": list(ranking[:position])}})
            regions.append({"rank_position": 0, "selected_action": int(ranking[0]),
                            "predicate": {"all_invalid_force_raw_top1": list(ranking)}})
            coverage_rule = "first_true_else_raw_top1"
        leaves.append({"leaf_id": int(leaf_id), "ranking": list(ranking), "regions": regions,
                       "coverage_rule": coverage_rule, "pairwise_disjoint": True, "total": True})

    return {
        "status": "PASS",
        "schema_version": "mask_fallback_v2",
        "selection": selection_semantics,
        "action_dim": action_dim,
        "leaves": leaves,
        "implicit_noop": (
            not explicit_noop
            and selection_semantics in {"ranked_first_valid", "top1_valid_else_noop"}
        ),
        "explicit_noop_action_ids": list(noop_ids),
        "explicit_noop_mask_total": explicit_noop,
        "defensive_fallback_action_id": noop_ids[0] if explicit_noop else None,
        "defensive_fallback_same_explicit_noop": explicit_noop,
        "fallback_unreachable_under_total_explicit_noop_mask": explicit_noop,
        "universal_over_runtime_masks": True,
        "mask_contract_hash": sha256_object(dict(mask_contract)),
    }
