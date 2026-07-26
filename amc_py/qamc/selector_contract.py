"""Explicit checkpoint-selector compatibility contracts."""

from __future__ import annotations

from dataclasses import dataclass

from amc_py.runtime_models import RuntimeSemantics


@dataclass(frozen=True, slots=True)
class SelectorContract:
    required_fields: frozenset[str]
    forbidden_fields: frozenset[str] = frozenset()
    allowed_semantics: frozenset[RuntimeSemantics] | None = None


SELECTOR_CONTRACTS = {
    "qos_recovery_stable": SelectorContract(
        required_fields=frozenset(
            {
                "hi_deadline_misses_sum",
                "mode_changes_mean",
                "baseline_mode_changes_mean",
                "lc_service_loss_mean",
                "policy_action_hist_json",
                "policy_action_is_increase_sum_json",
                "policy_action_safe_recovery_decrease_sum_json",
                "policy_action_over_increase_sum_json",
            }
        ),
        forbidden_fields=frozenset(
            {
                "lo_degraded_completion_ratio_mean",
                "lo_degraded_release_ratio_mean",
            }
        ),
    ),
}


def selector_is_compatible(name: str, semantics: RuntimeSemantics) -> bool:
    contract = SELECTOR_CONTRACTS.get(name)
    if contract is None:
        return False
    return contract.allowed_semantics is None or semantics in contract.allowed_semantics


def validate_selector_row(
    *,
    name: str,
    semantics: RuntimeSemantics,
    row: dict[str, object],
) -> None:
    if not selector_is_compatible(name, semantics):
        raise ValueError(f"SELECTOR_NOT_COMPATIBLE:{name}:{semantics.value}")
    contract = SELECTOR_CONTRACTS[name]
    missing = sorted(field for field in contract.required_fields if field not in row)
    if missing:
        raise ValueError("SELECTOR_REQUIRED_FIELDS_MISSING:" + ",".join(missing))
    present_forbidden = sorted(
        field
        for field in contract.forbidden_fields
        if row.get(field) not in (None, "")
    )
    if present_forbidden:
        raise ValueError(
            "SELECTOR_USES_SEMANTICALLY_INVALID_FIELDS:"
            + ",".join(present_forbidden)
        )


__all__ = [
    "SELECTOR_CONTRACTS",
    "SelectorContract",
    "selector_is_compatible",
    "validate_selector_row",
]
