"""Phase H04：逐 leaf/rank/action 检查 deployed policy 的预算不变量。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from itertools import product

from amc_py.rl.actions import action_violates_hi_decrease_guard
from formal_toolchain.policy.actions import replay_action
from formal_toolchain.policy.mask_fallback import select_first_valid
from formal_toolchain.adapters.synthetic_policy import mask_and_reasons


def check_deployed_policy_preservation(candidate: Mapping[str, Any], actions: Sequence[Any],
                                       tasks: Sequence[Any], *, leaves: Sequence[Any] = (),
                                       selected_cases: Sequence[Mapping[str, Any]] | None = None,
                                       forbid_decreasing_hi_budgets: bool = True) -> dict[str, Any]:
    if candidate.get("status") != "PASS":
        raise ValueError("deployed preservation 必须消费已通过 candidate envelope")
    if not leaves or selected_cases is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "DEPLOYED_LEAF_REGION_EVIDENCE_MISSING"}}
    states = {tuple(sorted((str(name), int(value)) for name, value in case["runtime_state"]["budgets"].items()))
              for case in selected_cases}
    expected_case_count = len(leaves) * len(states) * len(actions)
    if len(selected_cases) < expected_case_count:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "DEPLOYED_REGION_COVERAGE_INCOMPLETE",
                             "expected": expected_case_count, "actual": len(selected_cases)}}
    if not all({"leaf_id", "rank_position", "action_id", "valid", "mask_reasons", "ranking",
                "mask", "runtime_state", "action_definitions"} <= set(case) for case in selected_cases):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "DEPLOYED_REGION_EVIDENCE_INCOMPLETE"}}
    region_keys = {(case["leaf_id"], case["rank_position"], case["action_id"],
                    tuple(sorted((str(name), int(value)) for name, value in case["runtime_state"]["budgets"].items())))
                   for case in selected_cases}
    if len(region_keys) != len(selected_cases) or any(case["leaf_id"] not in leaves for case in selected_cases):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "DEPLOYED_REGION_EVIDENCE_DUPLICATE_OR_OUT_OF_SCOPE"}}
    action_ids = {int(action.action_id) for action in actions}
    for case in selected_cases:
        ranking = case["ranking"]
        if tuple(sorted(ranking)) != tuple(sorted(action_ids)) or not isinstance(case["mask_reasons"], (tuple, list)):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "failure": {"code": "DEPLOYED_REGION_RANKING_INVALID"}}
        formal_mask, formal_reasons = mask_and_reasons(case["runtime_state"], actions, tasks)
        if tuple(case["mask"]) != formal_mask or tuple(case["mask_reasons"]) != formal_reasons:
            return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {"code": "DEPLOYED_MASK_RECOMPUTE_MISMATCH", "leaf_id": case["leaf_id"],
                                "rank_position": case["rank_position"]}}
        first = select_first_valid(ranking, formal_mask, action_dim=len(action_ids))
        expected_valid = first == case["action_id"]
        if bool(case["valid"]) != expected_valid:
            return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {"code": "DEPLOYED_SELECTED_REGION_MISMATCH", "leaf_id": case["leaf_id"],
                                "rank_position": case["rank_position"], "expected": expected_valid}}
        if case["valid"] is False and not case["mask_reasons"]:
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "failure": {"code": "MASKED_REGION_REASON_MISSING"}}
    lower = candidate["lower"]
    upper = candidate["upper"]
    finite_domains = [tuple(range(int(lower[str(task.name)]), int(upper[str(task.name)]) + 1)) for task in tasks]
    if any(len(values) == 0 for values in finite_domains) or len(list(product(*finite_domains))) > 10000:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "DEPLOYED_FINITE_DOMAIN_TOO_LARGE"}}
    bases = tuple({str(task.name): int(value) for task, value in zip(tasks, values)}
                  for values in product(*finite_domains))
    witnesses = []
    selected_action_ids = {int(case["action_id"]) for case in selected_cases if case["valid"] and case["action_id"] is not None}
    region_groups: dict[tuple[int, tuple[tuple[str, int], ...]], list[Mapping[str, Any]]] = {}
    for case in selected_cases:
        key = (int(case["leaf_id"]), tuple(sorted((str(name), int(value))
                                                    for name, value in case["runtime_state"]["budgets"].items())))
        region_groups.setdefault(key, []).append(case)
    for action in actions:
        if action.action_id not in selected_action_ids:
            witnesses.append({"action_id": int(action.action_id), "checked": False, "masked": True})
            continue
        if getattr(action, "is_constraint_guided_pair", False) or getattr(action, "is_residual_ranked", False):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "failure": {"code": "DYNAMIC_ACTION_TARGET_UNSUPPORTED"}}
        if action_violates_hi_decrease_guard(action, tasks, forbid_decreasing_hi_budgets):
            # 该动作属于 mask invalid 区域，不是 deployed policy 的可选后继。
            witnesses.append({"action_id": int(action.action_id), "checked": False,
                              "masked_reason": "hi_decrease_guard"})
            continue
        checked_states = 0
        for base in bases:
            updates = replay_action(action, base, tasks)
            after = dict(base); after.update(updates)
            for name, value in after.items():
                if value < lower[name] or value > upper[name]:
                    return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                            "failure": {"code": "DEPLOYED_POLICY_PRESERVATION", "action_id": action.action_id,
                                        "budget_before": base, "budget_after": after, "task": name}}
            checked_states += 1
        witnesses.append({"action_id": int(action.action_id), "checked": True,
                          "finite_domain_states": checked_states})
    noop_groups = sum(1 for group in region_groups.values() if not any(case["valid"] for case in group))
    selected_positions = sorted({int(case["rank_position"]) for case in selected_cases if case["valid"]})
    return {"status": "PASS", "schema_version": "deployed_policy_preservation_v1",
            "leaf_count": len(leaves), "budget_state_count": len(states),
            "selected_action_ids": sorted(selected_action_ids), "first_valid_positions": selected_positions,
            "noop_state_count": noop_groups, "witnesses": witnesses,
            "implicit_noop_checked": noop_groups > 0}
