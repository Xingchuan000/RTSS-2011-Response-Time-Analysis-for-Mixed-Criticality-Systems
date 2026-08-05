"""Fail-closed role contracts for coherent semantic source mutations.

The non-vacuity lab must never treat a one-sided runtime edit as a faithful
mutation of the deployed PPP semantics.  Each named semantic change therefore
has an explicit set of source roles that must be patched together.
"""

from __future__ import annotations

from collections.abc import Iterable


_REQUIRED_ROLES: dict[str, frozenset[str]] = {
    "raw_top1_selection": frozenset({
        "DEPLOYED_SELECTION", "DEPLOYED_APPLY",
        "FROZEN_SELECTION", "FROZEN_APPLY",
    }),
    "top1_valid_else_noop": frozenset({
        "DEPLOYED_SELECTION", "FROZEN_SELECTION",
    }),
    "all_invalid_force_top1": frozenset({
        "DEPLOYED_SELECTION", "DEPLOYED_APPLY",
        "FROZEN_SELECTION", "FROZEN_APPLY",
    }),
    "guard_removal": frozenset({"DEPLOYED_GUARD", "FROZEN_GUARD"}),
    "rounding_to_nearest": frozenset({"DEPLOYED_APPLY", "FORMAL_SEMANTIC_MIRROR"}),
    "deadline_cleanup_removes_unfinished_job": frozenset({
        "DEPLOYED_IMPLEMENTATION", "FORMAL_SEMANTIC_MIRROR",
    }),
    "hi_job_truncate": frozenset({
        "DEPLOYED_IMPLEMENTATION", "FORMAL_SEMANTIC_MIRROR",
    }),
    "event_order_changed": frozenset({
        "DEPLOYED_IMPLEMENTATION", "FORMAL_SEMANTIC_MIRROR",
    }),
    "controller_overhead_changed": frozenset({
        "DEPLOYED_IMPLEMENTATION", "FORMAL_SEMANTIC_MIRROR",
    }),
    "nonquiescent_recovery_changed": frozenset({
        "DEPLOYED_IMPLEMENTATION", "FORMAL_SEMANTIC_MIRROR",
    }),
    "unstable_demand_reads": frozenset({
        "DEPLOYED_IMPLEMENTATION", "FORMAL_SEMANTIC_MIRROR",
    }),
}


def required_roles(semantic_change_id: str | None) -> frozenset[str]:
    """Return the role contract for one semantic mutation.

    Unknown research mutations retain the historical minimum requirement of a
    deployed implementation patch.  Named paper mutations use stricter,
    mutation-specific contracts above.
    """

    key = str(semantic_change_id or "").strip()
    return _REQUIRED_ROLES.get(key, frozenset({"DEPLOYED_IMPLEMENTATION"}))


def missing_roles(semantic_change_id: str | None, roles: Iterable[str]) -> tuple[str, ...]:
    present = {str(role) for role in roles}
    return tuple(sorted(required_roles(semantic_change_id) - present))


def validate_roles(semantic_change_id: str | None, roles: Iterable[str]) -> None:
    missing = missing_roles(semantic_change_id, roles)
    if missing:
        raise ValueError(
            "COHERENT_PATCH_REQUIRED_ROLES_MISSING:"
            f"semantic_change_id={semantic_change_id}:missing={list(missing)}"
        )
