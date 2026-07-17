"""Phase H02：candidate envelope 合成。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

from amc_py.rl.actions import action_violates_hi_decrease_guard
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.invariant.domains import consume_budget_domain
from formal_toolchain.invariant.safety_polytope import (
    derive_componentwise_upper,
    normalize_rows,
    rebuild_expected_rows,
    verify_production_rows,
    vector_satisfies_rows,
)
from formal_toolchain.policy.actions import replay_action


def synthesize_candidate_envelope(
    domain_certificate: Mapping[str, Any],
    actions: Sequence[Any],
    tasks: Sequence[Any],
    *,
    context_hash: str | None = None,
    runtime_adapter: Any | None = None,
    max_states: int = 100_000,
) -> dict[str, Any]:
    """在真实 runtime 上走结构化安全多面体；tiny synthetic 才保留穷举。"""
    if context_hash is None:
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "failure": {"code": "CANDIDATE_CONTEXT_MISSING"},
        }
    domain = consume_budget_domain(domain_certificate, context_hash)
    if not actions or not tasks:
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "failure": {"code": "CANDIDATE_INPUT_EMPTY"},
        }
    if any(
        getattr(action, "is_constraint_guided_pair", False)
        or getattr(action, "is_residual_ranked", False)
        for action in actions
    ):
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "failure": {"code": "DYNAMIC_ACTION_DOMAIN_UNSUPPORTED"},
        }
    if runtime_adapter is not None:
        production_polytope = runtime_adapter.export_budget_safety_polytope()
        if production_polytope.get("status") == "PASS":
            return _synthesize_from_safety_polytope(
                domain=domain,
                actions=actions,
                tasks=tasks,
                production_polytope=production_polytope,
                context_hash=context_hash,
            )
    return _synthesize_by_complete_enumeration(
        domain=domain,
        actions=actions,
        tasks=tasks,
        context_hash=context_hash,
        max_states=max_states,
    )


def _synthesize_from_safety_polytope(
    *,
    domain: Mapping[str, Any],
    actions: Sequence[Any],
    tasks: Sequence[Any],
    production_polytope: Mapping[str, Any],
    context_hash: str | None,
) -> dict[str, Any]:
    verified = verify_production_rows(production_polytope, tasks)
    if verified.get("status") != "PASS":
        return verified

    names = [str(task.name) for task in tasks]
    formal_lower = {
        name: int(domain["tasks"][name]["formal_lower"])
        for name in names
    }
    candidate_lower = {
        name: int(domain["tasks"][name]["candidate_positive_lower"])
        for name in names
    }
    hard_upper = {
        name: int(domain["tasks"][name]["action_hard_upper"])
        for name in names
    }

    derived = derive_componentwise_upper(
        rows=verified["rows"],
        task_order=names,
        candidate_lower=candidate_lower,
        action_hard_upper=hard_upper,
    )
    if derived.get("status") != "PASS":
        return derived

    initial = {
        name: int(domain["tasks"][name]["initial"])
        for name in names
    }
    if not vector_satisfies_rows(initial, verified["rows"]):
        return {
            "status": "FAIL",
            "route": "MODEL_CONFORMANCE_FAILED",
            "failure": {"code": "INITIAL_BUDGET_OUTSIDE_SAFETY_POLYTOPE"},
        }

    upper = dict(derived["upper"])
    return {
        "status": "PASS",
        "schema_version": "candidate_envelope_v2",
        "method": "single_action_safety_polytope_projection",
        "context_hash": context_hash,
        "lower": formal_lower,
        "candidate_positive_lower": candidate_lower,
        "upper": upper,
        "active_release_budget_upper": dict(upper),
        "action_hard_upper": hard_upper,
        "safety_polytope_hash": verified["row_hash"],
        "safety_polytope_rows": verified["rows"],
        "coordinate_upper_witnesses": derived["witnesses"],
        "initial_budget": initial,
        "initial_budget_in_polytope": True,
        "proof_rule": (
            "mask_valid_non_noop_implies_candidate_in_polytope; "
            "polytope_implies_componentwise_envelope"
        ),
    }


def _synthesize_by_complete_enumeration(
    *,
    domain: Mapping[str, Any],
    actions: Sequence[Any],
    tasks: Sequence[Any],
    context_hash: str | None,
    max_states: int,
) -> dict[str, Any]:
    state_count = 1
    names = [str(task.name) for task in tasks]
    values_by_name: list[list[int]] = []
    for name in names:
        interval = domain["tasks"][name]["integer_interval"]
        lower = int(interval["lower"])
        upper = int(interval["upper"])
        values = list(range(lower, upper + 1))
        state_count *= len(values)
        values_by_name.append(values)
    if state_count > max_states:
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "failure": {
                "code": "FINITE_DOMAIN_TOO_LARGE",
                "state_count": state_count,
                "max_states": max_states,
            },
        }

    upper = {name: int(row["action_hard_upper"]) for name, row in domain["tasks"].items()}
    lower = {name: int(row["formal_lower"]) for name, row in domain["tasks"].items()}
    witnesses: list[dict[str, Any]] = []
    for vector in product(*values_by_name):
        budgets = dict(zip(names, vector))
        for action in actions:
            if action_violates_hi_decrease_guard(action, tasks, True):
                continue
            updates = replay_action(action, budgets, tasks)
            after = dict(budgets)
            after.update(updates)
            for name, value in after.items():
                if value < lower[name] or value > upper[name]:
                    return {
                        "status": "FAIL",
                        "route": "POLICY_CONTRACT_VIOLATION",
                        "failure": {
                            "code": "CANDIDATE_ENVELOPE_VIOLATION",
                            "action_id": int(action.action_id),
                            "budget_before": budgets,
                            "budget_after": after,
                            "task": name,
                            "upper": upper[name],
                            "lower": lower[name],
                        },
                    }
        witnesses.append({
            "action_ids": [int(action.action_id) for action in actions],
            "checked": True,
        })

    return {
        "status": "PASS",
        "schema_version": "candidate_envelope_v1",
        "method": "finite_domain_enumeration",
        "lower": lower,
        "upper": upper,
        "active_release_budget_upper": dict(upper),
        "candidate_positive_lower": {name: int(row["candidate_positive_lower"]) for name, row in domain["tasks"].items()},
        "action_hard_upper": upper,
        "domain_certificate_hash": domain["source_hash"],
        "witnesses": witnesses,
        "context_hash": context_hash,
    }
