from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.reference.protected_priority_prefix.construction import build_saturated_protected_prefix
from formal_toolchain.reference.protected_priority_prefix.certificates import (
    build_partition_certificate, build_saturation_certificate,
)
from .protocol import PreparedRouteAnalysis
from .protected_prefix_checkers import (
    check_partition, check_saturation, check_parameter_preservation,
    check_lo_saturation, check_prefix_rta, check_mathematical_conformance,
    check_selected_safety, check_runtime_schema_conformance, check_simulation_domain,
    check_weak_forward_simulation, check_hi_bad_prefix_reflection,
    check_reference_hi_safety_from_protected_prefix, check_prefix_model_conformance,
    check_full_reference_recurring_input_oracle, check_protected_input_stream_projection,
    check_protected_input_demand_receptiveness, check_prefix_reference_prefix_extension,
    check_prefix_canonical_successor_total, check_prefix_same_time_closure_terminates,
    check_prefix_time_divergence, check_prefix_idle_jump_stutter_expansion,
    check_prefix_complete_execution_exists,
)


class ProtectedPrefixRoute:
    route_id = "protected_prefix"
    route_implementation_version = "saturated-protected-prefix-v1"

    def registry_fragment_path(self) -> Path:
        return Path(__file__).parents[1] / "specs/routes/protected_prefix_registry.json"

    def prepare_analysis(self, *, full_reference_taskset: Any,
                         reference_context_hash: str) -> PreparedRouteAnalysis:
        result = build_saturated_protected_prefix(
            full_reference_taskset, source_context_hash=reference_context_hash)
        return PreparedRouteAnalysis(
            route_id=self.route_id, full_reference_taskset=full_reference_taskset,
            analysis_taskset=result.prefix_taskset,
            analysis_taskset_kind="saturated_protected_prefix",
            route_implementation_version=self.route_implementation_version,
            construction_witnesses={"build_result": result},
            route_metadata={"analysis_taskset_kind": "saturated_protected_prefix",
                            "cutoff_task_name": result.cutoff_task_name,
                            "cutoff_priority_index": result.cutoff_priority_index,
                            "protected_count": len(result.protected_task_names),
                            "tail_count": len(result.tail_task_names)},
        )

    def build_construction_certificates(self, *, prepared: PreparedRouteAnalysis,
                                        terminal_context_hash: str):
        """Build immutable construction witnesses only.

        Checker functions belong to :meth:`checker_catalog`; mixing callables into
        this mapping makes the fresh verifier treat functions as certificates and
        leaves the corresponding route obligations permanently UNRESOLVED.
        """

        result = prepared.construction_witnesses["build_result"]
        return {
            "PROTECTED_PRIORITY_PREFIX_PARTITION": build_partition_certificate(
                result, context_hash=terminal_context_hash),
            "SATURATED_PROTECTED_PREFIX_REFERENCE": build_saturation_certificate(
                result, context_hash=terminal_context_hash),
        }

    def checker_catalog(self) -> Mapping[str, Any]:
        return {
            "PROTECTED_PRIORITY_PREFIX_PARTITION": check_partition,
            "SATURATED_PROTECTED_PREFIX_REFERENCE": check_saturation,
            "PROTECTED_PREFIX_PARAMETER_PRESERVATION": check_parameter_preservation,
            "PROTECTED_PREFIX_LO_SATURATION": check_lo_saturation,
            "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC": check_prefix_rta,
            "PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE": check_prefix_model_conformance,
            "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE": check_mathematical_conformance,
            "SELECTED_REFERENCE_HI_SAFETY": check_selected_safety,
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE": check_runtime_schema_conformance,
            "FULL_TO_PREFIX_SIMULATION_DOMAIN": check_simulation_domain,
            "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED": check_weak_forward_simulation,
            "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION": check_hi_bad_prefix_reflection,
            "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX": check_reference_hi_safety_from_protected_prefix,
            "FULL_REFERENCE_RECURRING_INPUT_ORACLE": check_full_reference_recurring_input_oracle,
            "PROTECTED_INPUT_STREAM_PROJECTION": check_protected_input_stream_projection,
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS": check_protected_input_demand_receptiveness,
            "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION": check_prefix_reference_prefix_extension,
            "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL": check_prefix_canonical_successor_total,
            "PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES": check_prefix_same_time_closure_terminates,
            "PROTECTED_PREFIX_TIME_DIVERGENCE": check_prefix_time_divergence,
            "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION": check_prefix_idle_jump_stutter_expansion,
            "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS": check_prefix_complete_execution_exists,
        }


ROUTE = ProtectedPrefixRoute()
