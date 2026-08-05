from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class FreshVerifierState:
    inputs: Any
    certified_envelope: Mapping[str, Any] | None
    fresh_reference_taskset: Any
    fresh_rta_production: Mapping[str, Any]
    fresh_rta_replay: Mapping[str, Any]
    concrete_preclosed_engine: Any
    concrete_runtime_snapshot: Any
    reference_preclosed_state: Any
    reference_runtime_snapshot: Any
    phase_k_objects: Mapping[str, Mapping[str, Any]]
    route_strategy: Any = None
    prepared_route: Any = None
    full_reference_taskset: Any = None
    analysis_taskset: Any = None
    terminal_route_context: Mapping[str, Any] | None = None
    route_construction_certificates: Mapping[str, Mapping[str, Any]] = None
    selected_rta_obligation_id: str | None = None
    selected_route_id: str | None = None


@dataclass(frozen=True, slots=True)
class CheckerContext:
    obligation_id: str
    candidate_certificate: Mapping[str, Any]
    candidate_evidence: Mapping[str, Any] | None
    verified_predecessors: Mapping[str, Mapping[str, Any]]
    expected_context_hash: str
    raw_inputs: Any
    fresh_state: FreshVerifierState | None
