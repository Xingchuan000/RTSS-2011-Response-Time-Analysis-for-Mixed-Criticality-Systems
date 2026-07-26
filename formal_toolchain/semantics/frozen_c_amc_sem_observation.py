"""Frozen v11 observation schema used by the C-AMC-sem/P0 proof route.

The proof binds the exported tree feature artifact to this stable schema.  The
mutable training/runtime observation implementation is intentionally outside
of the blocking proof hash and may contain q-AMC-specific features.
"""

from __future__ import annotations


FORMAL_OBSERVATION_CONTRACT_VERSION = "c_amc_sem_p0_v11_observation_v1"

V11_PER_TASK_FEATURE_NAMES: tuple[str, ...] = (
    "budget_norm",
    "recent_cost_norm",
    "ema_cost_norm",
    "max_cost_k_norm",
    "overrun_ema",
    "risk",
    "surplus",
    "criticality",
    "priority_norm",
    "util_budget",
)

V11_GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "total_budget_util",
    "hi_budget_util",
    "lo_budget_util",
    "recent_mode_change_rate",
    "recent_lo_cancel_rate",
    "recent_hi_overrun_rate",
    "recent_lo_overrun_rate",
    "safety_margin_min",
)


def _build_v11_family_observation(
    *,
    ordered_task_names: list[str],
    per_task_values: dict[str, dict[str, float]],
    global_values: dict[str, float],
) -> tuple[float, ...]:
    """Flatten task-structured values in the frozen feature order."""

    values: list[float] = []
    for task_name in ordered_task_names:
        task_values = per_task_values[task_name]
        for feature_name in V11_PER_TASK_FEATURE_NAMES:
            values.append(float(task_values[feature_name]))
    for feature_name in V11_GLOBAL_FEATURE_NAMES:
        values.append(float(global_values[feature_name]))
    return tuple(values)


def build_v11_full_10d_observation(
    *,
    ordered_task_names: list[str],
    per_task_values: dict[str, dict[str, float]],
    global_values: dict[str, float],
) -> tuple[float, ...]:
    """Canonical v11_full_10d entrypoint."""

    return _build_v11_family_observation(
        ordered_task_names=ordered_task_names,
        per_task_values=per_task_values,
        global_values=global_values,
    )


def build_observation(
    *,
    observation_mode: str,
    ordered_task_names: list[str],
    per_task_values: dict[str, dict[str, float]],
    global_values: dict[str, float],
) -> tuple[float, ...]:
    """Frozen dispatcher for the only observation mode certified by this route."""

    if observation_mode != "v11_full_10d":
        raise ValueError("UNSUPPORTED_OBSERVATION_MODE")
    return build_v11_full_10d_observation(
        ordered_task_names=ordered_task_names,
        per_task_values=per_task_values,
        global_values=global_values,
    )
