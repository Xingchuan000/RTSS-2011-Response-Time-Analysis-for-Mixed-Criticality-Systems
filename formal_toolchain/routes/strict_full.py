from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .protocol import PreparedRouteAnalysis
from .protected_prefix_checkers import check_selected_safety


class StrictFullRoute:
    route_id = "strict_full"
    route_implementation_version = "strict-full-route-v2"

    def registry_fragment_path(self) -> Path:
        return Path(__file__).parents[1] / "specs/routes/strict_full_registry.json"

    def prepare_analysis(self, *, full_reference_taskset: Any,
                         reference_context_hash: str) -> PreparedRouteAnalysis:
        return PreparedRouteAnalysis(
            route_id=self.route_id, full_reference_taskset=full_reference_taskset,
            analysis_taskset=full_reference_taskset,
            analysis_taskset_kind="full_reference",
            route_implementation_version=self.route_implementation_version,
            construction_witnesses={"identity": True, "reference_context_hash": reference_context_hash},
            route_metadata={"analysis_taskset_kind": "full_reference"},
        )

    def build_construction_certificates(self, *, prepared: PreparedRouteAnalysis,
                                        terminal_context_hash: str):
        return {}

    def checker_catalog(self) -> Mapping[str, Any]:
        return {"SELECTED_REFERENCE_HI_SAFETY": check_selected_safety}


ROUTE = StrictFullRoute()
