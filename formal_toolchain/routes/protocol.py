from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class PreparedRouteAnalysis:
    route_id: str
    full_reference_taskset: Any
    analysis_taskset: Any
    analysis_taskset_kind: str
    route_implementation_version: str
    construction_witnesses: Mapping[str, Any]
    route_metadata: Mapping[str, Any]


RouteTasksetBundle = PreparedRouteAnalysis


class TerminalProofRoute(Protocol):
    route_id: str

    def registry_fragment_path(self) -> Path: ...

    def prepare_analysis(self, *, full_reference_taskset: Any,
                         reference_context_hash: str) -> PreparedRouteAnalysis: ...

    def build_construction_certificates(self, *, prepared: PreparedRouteAnalysis,
                                        terminal_context_hash: str) -> Mapping[str, Mapping[str, Any]]: ...

    def checker_catalog(self) -> Mapping[str, Any]: ...
