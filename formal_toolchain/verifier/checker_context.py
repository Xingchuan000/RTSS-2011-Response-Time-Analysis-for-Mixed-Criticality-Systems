from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class FreshVerifierState:
    inputs: Any
    certified_envelope: Mapping[str, Any]
    fresh_reference_taskset: Any
    fresh_rta_production: Mapping[str, Any]
    fresh_rta_replay: Mapping[str, Any]
    concrete_preclosed_engine: Any
    concrete_runtime_snapshot: Any
    reference_preclosed_state: Any
    reference_runtime_snapshot: Any
    phase_k_objects: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class CheckerContext:
    obligation_id: str
    candidate_certificate: Mapping[str, Any]
    candidate_evidence: Mapping[str, Any] | None
    verified_predecessors: Mapping[str, Mapping[str, Any]]
    expected_context_hash: str
    raw_inputs: Any
    fresh_state: FreshVerifierState | None
