"""q-AMC native quality state and budget-overlay support.

The package deliberately keeps quality state separate from the legacy
``Job.is_degraded`` flag and from :class:`amc_py.budget_runtime.BudgetState`.
"""

from .demand import QAmcDemandSnapshot, map_full_cost_to_quality_cost
from .models import QAmcProfileBundle, QAmcQualityLevel, QAmcTaskProfile
from .profile_spec import QAmcProfileSpec
from .profiles import (
    build_qamc_profile_bundle,
    load_profile_bundle,
    partition_design_budget,
    write_profile_bundle,
)
from .runtime_controller import (
    QAmcOverrunDecision,
    QAmcQualitySnapshot,
    QAmcRuntimeController,
)
from .metrics_support import QAmcMetrics, compute_qamc_metrics, qamc_metrics_to_row

__all__ = [
    "QAmcDemandSnapshot",
    "QAmcOverrunDecision",
    "QAmcProfileBundle",
    "QAmcProfileSpec",
    "QAmcQualityLevel",
    "QAmcQualitySnapshot",
    "QAmcRuntimeController",
    "QAmcTaskProfile",
    "QAmcMetrics",
    "build_qamc_profile_bundle",
    "load_profile_bundle",
    "map_full_cost_to_quality_cost",
    "partition_design_budget",
    "write_profile_bundle",
    "compute_qamc_metrics",
    "qamc_metrics_to_row",
]
