from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ActionRiskRecord:
    action_id: int
    action_kind: str
    target_task: str | None
    target_criticality: str | None
    target_priority: int | None
    budget_direction: str | None
    interferes_with_hi_tasks: tuple[str, ...] = ()
    risk_class: str = "BENIGN_OR_UNKNOWN"


@dataclass(frozen=True)
class LeafAuditRecord:
    seed: int
    tree_variant: str
    tree_hash: str
    leaf_id: int
    guard: dict[str, Any]
    action_ranking: tuple[int, ...]
    training_samples: int
    hout_hit_count: int
    scenario_coverage: tuple[str, ...]
    raw_top1_invalid_count: int
    fallback_count: int
    all_invalid_count: int
    noop_count: int
    selected_rank_histogram: dict[str, int]
    reject_reason_histogram: dict[str, int]
    action_risks: tuple[ActionRiskRecord, ...]
    selected_region_status: str
    symbolic_witness_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
