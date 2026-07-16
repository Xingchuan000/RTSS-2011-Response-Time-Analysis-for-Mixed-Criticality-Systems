"""Phase H02：有限预算域上的 candidate envelope 合成。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from amc_py.rl.actions import action_violates_hi_decrease_guard
from formal_toolchain.policy.actions import replay_action
from .domains import consume_budget_domain


def synthesize_candidate_envelope(domain_certificate: Mapping[str, Any], actions: Sequence[Any],
                                  tasks: Sequence[Any], *, context_hash: str | None = None,
                                  max_states: int = 100_000,
                                  forbid_decreasing_hi_budgets: bool = True) -> dict[str, Any]:
    """在已认证有限域内逐动作检查 upper/lower；不使用默认 envelope。"""
    if context_hash is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "CANDIDATE_CONTEXT_MISSING"}}
    domain = consume_budget_domain(domain_certificate, context_hash)
    if not actions or not tasks:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "CANDIDATE_INPUT_EMPTY"}}
    if any(getattr(action, "is_constraint_guided_pair", False) or getattr(action, "is_residual_ranked", False)
           for action in actions):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "DYNAMIC_ACTION_DOMAIN_UNSUPPORTED"}}
    state_count = 1
    for row in domain["tasks"].values():
        state_count *= len(row["finite_integer_domain"])
    # 当前第一轮 verifier 只接受可完整枚举的 finite domain；超过预算不能
    # 用 cap/default 伪造证明，后续应由计划规定的 SMT backend 接管。
    if state_count > max_states:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "FINITE_DOMAIN_TOO_LARGE", "state_count": state_count,
                             "max_states": max_states}}
    upper = {name: int(row["runtime_deploy_cap"]) for name, row in domain["tasks"].items()}
    # P0 的预算不变量是：LO 允许从 0 起定义域（运行时正整数约束另由
    # action primitive 保持），HI 才要求 B_i >= C_i^code,LO。
    lower = {name: (int(row["code_lower"]) if next(task for task in tasks if str(task.name) == name).criticality.value == "HI" else 0)
             for name, row in domain["tasks"].items()}
    witnesses = []
    from itertools import product
    names = list(domain["tasks"])
    values_by_name = [domain["tasks"][name]["finite_integer_domain"] for name in names]
    for vector in product(*values_by_name):
        budgets = dict(zip(names, vector))
        for action in actions:
            if action_violates_hi_decrease_guard(action, tasks, forbid_decreasing_hi_budgets):
                continue
            updates = replay_action(action, budgets, tasks)
            after = dict(budgets); after.update(updates)
            for name, value in after.items():
                if value < lower[name] or value > upper[name]:
                    return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                            "failure": {"code": "CANDIDATE_ENVELOPE_VIOLATION", "action_id": action.action_id,
                                        "budget_before": budgets, "budget_after": after, "task": name,
                                        "upper": upper[name], "lower": lower[name]}}
        witnesses.append({"action_id": int(action.action_id), "checked": True})
    return {"status": "PASS", "schema_version": "candidate_envelope_v1", "method": "finite_domain_enumeration",
            "lower": lower, "upper": upper, "active_release_budget_upper": dict(upper),
            "domain_certificate_hash": domain["source_hash"], "witnesses": witnesses}
