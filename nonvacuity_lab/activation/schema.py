from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..schema import ActivationStatus


@dataclass(frozen=True)
class ActivationResult:
    mutation_id: str
    status: ActivationStatus
    evidence_modes: tuple[str, ...] = ()
    leaf_id: int | None = None
    action_id: int | None = None
    hout_hit_count: int = 0
    baseline_reject_count: int = 0
    selected_after_mutation_count: int = 0
    all_invalid_count: int = 0
    guard_satisfiable: bool | None = None
    illegal_action_witness: str | None = None
    post_invariant_violation: bool | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mutation_activation_v1",
            "mutation_id": self.mutation_id,
            "status": self.status.value,
            "evidence_modes": list(self.evidence_modes),
            "leaf_id": self.leaf_id,
            "action_id": self.action_id,
            "hout_hit_count": self.hout_hit_count,
            "baseline_reject_count": self.baseline_reject_count,
            "selected_after_mutation_count": self.selected_after_mutation_count,
            "all_invalid_count": self.all_invalid_count,
            "guard_satisfiable": self.guard_satisfiable,
            "illegal_action_witness": self.illegal_action_witness,
            "post_invariant_violation": self.post_invariant_violation,
            "details": dict(self.details),
        }
