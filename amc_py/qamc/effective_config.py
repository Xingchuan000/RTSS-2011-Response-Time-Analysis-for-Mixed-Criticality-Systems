"""Canonical effective configuration contract for q-AMC reference runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class QAmcReferenceEffectiveConfig:
    schema_version: str
    action_space: str
    q_network_type: str
    action_feature_mode: str
    include_explicit_noop: bool
    budget_increase_ratio: float
    budget_decrease_ratio: float
    budget_rounding_mode: str
    min_budget_delta: int
    budget_floor_ratio: float
    check_safety: bool
    step_guard_semantics: str
    observation_mode: str
    reward_mode: str
    reward_config_path: str
    reward_config_sha256: str
    agent_period: int
    save_best_by: str
    selector_contract_version: str
    enable_deploy_cap_mask: bool
    deploy_cap_mask_ratio: float
    deploy_cap_mask_criticality: str
    forbid_decreasing_hi_budgets: bool
    action_dim: int
    observation_dim: int

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_jsonable())


__all__ = ["QAmcReferenceEffectiveConfig", "canonical_sha256"]
