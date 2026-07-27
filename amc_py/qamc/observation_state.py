"""q-AMC state exposed to RL observations without advancing runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from amc_py.models import Criticality, Task
from amc_py.qamc.demand import map_full_cost_to_quality_cost
from amc_py.qamc.models import QAmcProfileBundle


@dataclass(frozen=True, slots=True)
class QAmcTaskObservationState:
    runtime_level: int | None
    target_quality_normalized: float
    target_demand_ratio: float


def build_qamc_task_observation_state(
    *,
    ordered_tasks: Sequence[Task],
    profile_bundle: QAmcProfileBundle,
    current_level_by_task: Mapping[str, int],
) -> dict[str, QAmcTaskObservationState]:
    result: dict[str, QAmcTaskObservationState] = {}

    for task in ordered_tasks:
        if task.criticality is Criticality.HI:
            result[task.name] = QAmcTaskObservationState(
                runtime_level=None,
                target_quality_normalized=1.0,
                target_demand_ratio=1.0,
            )
            continue

        try:
            profile = profile_bundle.profiles[task.name]
            runtime_level = int(current_level_by_task[task.name])
            level = profile.level(runtime_level)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"QAMC_OBSERVATION_CONTEXT_INVALID:{task.name}"
            ) from exc

        demand = map_full_cost_to_quality_cost(
            full_quality_actual_cost=profile.design_c_lo,
            profile=profile,
            runtime_level=runtime_level,
        )
        demand_ratio = (
            float(demand.quality_specific_actual_cost)
            / float(profile.design_c_lo)
        )
        quality = float(level.normalized_quality)

        if not math.isfinite(quality) or not 0.0 < quality <= 1.0:
            raise ValueError(f"QAMC_QUALITY_OUT_OF_RANGE:{task.name}")
        if not math.isfinite(demand_ratio) or not 0.0 < demand_ratio <= 1.0:
            raise ValueError(f"QAMC_DEMAND_RATIO_OUT_OF_RANGE:{task.name}")

        result[task.name] = QAmcTaskObservationState(
            runtime_level=runtime_level,
            target_quality_normalized=quality,
            target_demand_ratio=demand_ratio,
        )

    return result
