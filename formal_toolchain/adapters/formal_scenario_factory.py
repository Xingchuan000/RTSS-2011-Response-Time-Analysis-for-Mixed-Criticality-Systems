from __future__ import annotations

from formal_toolchain.adapters.batch_frozen_scenario import BatchFrozenExecutionScenario


def build_formal_scenario(*, base_scenario, ordered_tasks):
    if not hasattr(base_scenario, "actual_cost_for"):
        raise TypeError("base_scenario 必须实现 actual_cost_for(task, release_index)")
    return BatchFrozenExecutionScenario(
        ordered_tasks=tuple(ordered_tasks),
        delegate=base_scenario,
    )
