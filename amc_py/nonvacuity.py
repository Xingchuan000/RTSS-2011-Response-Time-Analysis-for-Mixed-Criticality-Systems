"""Non-vacuity experiment profiles.

The production/default profile is ``off``.  Every mutation is opt-in and is
serialized into the formal request so candidate generation and fresh
verification reconstruct the same runtime semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PROFILE_OFF = "off"

SUPPORTED_PROFILES = (
    PROFILE_OFF,
    "b1_mask_bypass",
    "b2_no_first_valid",
    "b3_all_invalid_force_top1",
    "b4_disable_guard",
    "c1_action_step",
    "c2_round_nearest",
    "c2_min_increment_2",
    "c3_retroactive_release_budget",
    "e1_deadline_cleanup_remove",
    "e2_hi_budget_cap_truncate",
    "e3_arrival_before_deadline",
    "e4_controller_overhead",
    "e5_recover_without_quiescence",
    "e6_unstable_demand_reads",
)

SUPPORTED_DISABLED_GUARDS = (
    "hi_decrease",
    "deploy_cap",
    "budget_floor",
    "residual_guard",
    "safety_checker",
)


@dataclass(frozen=True, slots=True)
class NonvacuitySettings:
    profile: str = PROFILE_OFF
    selection_semantics: str = "ranked_first_valid"
    step_guard_semantics: str = "checked"
    disabled_guards: tuple[str, ...] = ()
    action_ratio_override: float | None = None
    rounding_mode: str = "ceil_floor"
    min_budget_delta: int = 1
    retroactive_release_budget: bool = False
    deadline_cleanup_remove: bool = False
    hi_budget_cap_truncate: bool = False
    arrival_before_deadline: bool = False
    controller_overhead_ticks: int = 0
    recover_without_quiescence: bool = False
    unstable_demand_reads: bool = False

    @property
    def enabled(self) -> bool:
        return self.profile != PROFILE_OFF

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "enabled": self.enabled,
            "selection_semantics": self.selection_semantics,
            "step_guard_semantics": self.step_guard_semantics,
            "disabled_guards": list(self.disabled_guards),
            "action_ratio_override": self.action_ratio_override,
            "rounding_mode": self.rounding_mode,
            "min_budget_delta": self.min_budget_delta,
            "retroactive_release_budget": self.retroactive_release_budget,
            "deadline_cleanup_remove": self.deadline_cleanup_remove,
            "hi_budget_cap_truncate": self.hi_budget_cap_truncate,
            "arrival_before_deadline": self.arrival_before_deadline,
            "controller_overhead_ticks": self.controller_overhead_ticks,
            "recover_without_quiescence": self.recover_without_quiescence,
            "unstable_demand_reads": self.unstable_demand_reads,
        }


def _normalise_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(params or {})
    disabled = raw.get("disabled_guards", ())
    if isinstance(disabled, str):
        disabled = (disabled,)
    raw["disabled_guards"] = tuple(str(value) for value in disabled)
    return raw


def resolve_nonvacuity_settings(
    profile: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> NonvacuitySettings:
    """Resolve one experiment profile into explicit runtime semantics.

    ``None`` and the empty string both resolve to ``off``.  Unknown profiles or
    parameters fail closed; there is no permissive fallback.
    """

    name = str(profile or PROFILE_OFF).strip().lower()
    if name not in SUPPORTED_PROFILES:
        raise ValueError(f"UNSUPPORTED_NONVACUITY_PROFILE:{name}")
    raw = _normalise_params(params)

    settings = NonvacuitySettings(profile=name)
    if name == PROFILE_OFF:
        if any(value not in (None, (), [], {}, "") for value in raw.values()):
            raise ValueError("NONVACUITY_PARAMS_REQUIRE_NON_OFF_PROFILE")
        return settings

    if name == "b1_mask_bypass":
        settings = NonvacuitySettings(
            profile=name,
            selection_semantics="raw_top1",
            step_guard_semantics="unchecked_apply",
        )
    elif name == "b2_no_first_valid":
        settings = NonvacuitySettings(
            profile=name,
            selection_semantics="top1_or_noop",
            step_guard_semantics="checked",
        )
    elif name == "b3_all_invalid_force_top1":
        settings = NonvacuitySettings(
            profile=name,
            selection_semantics="first_valid_else_top1",
            step_guard_semantics="unchecked_if_invalid",
        )
    elif name == "b4_disable_guard":
        disabled = tuple(sorted(set(raw.get("disabled_guards", ()))))
        if not disabled:
            raise ValueError("B4_REQUIRES_DISABLED_GUARD")
        unknown = sorted(set(disabled) - set(SUPPORTED_DISABLED_GUARDS))
        if unknown:
            raise ValueError(f"UNSUPPORTED_DISABLED_GUARD:{','.join(unknown)}")
        settings = NonvacuitySettings(profile=name, disabled_guards=disabled)
    elif name == "c1_action_step":
        ratio = float(raw.get("action_ratio", 0.05))
        if not (0.0 < ratio < 1.0):
            raise ValueError("NONVACUITY_ACTION_RATIO_OUT_OF_RANGE")
        settings = NonvacuitySettings(profile=name, action_ratio_override=ratio)
    elif name == "c2_round_nearest":
        settings = NonvacuitySettings(profile=name, rounding_mode="nearest")
    elif name == "c2_min_increment_2":
        delta = int(raw.get("min_budget_delta", 2))
        if delta < 2:
            raise ValueError("NONVACUITY_MIN_BUDGET_DELTA_MUST_BE_AT_LEAST_2")
        settings = NonvacuitySettings(profile=name, min_budget_delta=delta)
    elif name == "c3_retroactive_release_budget":
        settings = NonvacuitySettings(profile=name, retroactive_release_budget=True)
    elif name == "e1_deadline_cleanup_remove":
        settings = NonvacuitySettings(profile=name, deadline_cleanup_remove=True)
    elif name == "e2_hi_budget_cap_truncate":
        settings = NonvacuitySettings(profile=name, hi_budget_cap_truncate=True)
    elif name == "e3_arrival_before_deadline":
        settings = NonvacuitySettings(profile=name, arrival_before_deadline=True)
    elif name == "e4_controller_overhead":
        ticks = int(raw.get("controller_overhead_ticks", 1))
        if ticks <= 0:
            raise ValueError("NONVACUITY_CONTROLLER_OVERHEAD_MUST_BE_POSITIVE")
        settings = NonvacuitySettings(profile=name, controller_overhead_ticks=ticks)
    elif name == "e5_recover_without_quiescence":
        settings = NonvacuitySettings(profile=name, recover_without_quiescence=True)
    elif name == "e6_unstable_demand_reads":
        settings = NonvacuitySettings(profile=name, unstable_demand_reads=True)
    return settings


def formal_expected_failure(settings: NonvacuitySettings) -> tuple[str, str] | None:
    """Return the first P0-level expected rejection for model mutations."""

    if settings.deadline_cleanup_remove:
        return "DEADLINE_OBSERVATION", "DEADLINE_CLEANUP_REMOVES_JOB"
    if settings.hi_budget_cap_truncate:
        return "HI_NONTRUNCATION", "HI_BUDGET_CAP_TRUNCATES_JOB"
    if settings.arrival_before_deadline:
        return "DEADLINE_BOUNDARY_ORDER", "ARRIVAL_PRECEDES_DEADLINE_OBSERVATION"
    if settings.controller_overhead_ticks > 0:
        return "OVERHEAD_PROFILE", "PROCESSOR_OVERHEAD_NOT_ZERO"
    if settings.recover_without_quiescence:
        return "MODE_SEMANTICS_CONFORMANCE", "RECOVERY_WITHOUT_QUIESCENCE"
    if settings.unstable_demand_reads:
        return "DEMAND_ORACLE_BATCH_CONTRACT", "DEMAND_ORACLE_NOT_KEY_STABLE"
    if settings.retroactive_release_budget:
        return "ACTIVE_RELEASE_BUDGET_INVARIANT", "ACTIVE_RELEASE_BUDGET_RETROACTIVELY_MUTATED"
    return None


__all__ = [
    "PROFILE_OFF",
    "SUPPORTED_PROFILES",
    "SUPPORTED_DISABLED_GUARDS",
    "NonvacuitySettings",
    "resolve_nonvacuity_settings",
    "formal_expected_failure",
]
