"""Deterministic scalar-sample to q-AMC demand mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .models import QAmcTaskProfile


@dataclass(frozen=True, slots=True)
class QAmcDemandSnapshot:
    full_quality_actual_cost: int
    full_quality_application_component: int
    observed_interference_component: int
    target_isolated_wcet: int
    target_application_component: int
    quality_specific_actual_cost: int
    mapping_version: str


def map_full_cost_to_quality_cost(
    *,
    full_quality_actual_cost: int,
    profile: QAmcTaskProfile,
    runtime_level: int,
) -> QAmcDemandSnapshot:
    if full_quality_actual_cost <= 0:
        raise ValueError("QAMC_NONPOSITIVE_FULL_COST")
    w_max = profile.full_quality_isolated_wcet
    w_q = profile.level(runtime_level).isolated_wcet
    application_full = min(full_quality_actual_cost, w_max)
    observed_interference = max(0, full_quality_actual_cost - w_max)
    application_q = max(1, math.ceil(application_full * w_q / w_max))
    quality_cost = application_q + observed_interference
    if runtime_level == profile.initial_runtime_level and quality_cost != full_quality_actual_cost:
        raise AssertionError("QAMC_FULL_QUALITY_IDENTITY_FAILED")
    if not 1 <= quality_cost <= full_quality_actual_cost:
        raise AssertionError("QAMC_QUALITY_COST_MONOTONICITY_FAILED")
    return QAmcDemandSnapshot(
        full_quality_actual_cost=full_quality_actual_cost,
        full_quality_application_component=application_full,
        observed_interference_component=observed_interference,
        target_isolated_wcet=w_q,
        target_application_component=application_q,
        quality_specific_actual_cost=quality_cost,
        mapping_version="wcet_capped_component_split_v1",
    )


__all__ = ["QAmcDemandSnapshot", "map_full_cost_to_quality_cost"]
