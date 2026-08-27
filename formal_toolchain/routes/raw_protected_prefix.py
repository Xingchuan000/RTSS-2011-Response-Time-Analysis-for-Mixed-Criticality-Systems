"""V8 raw protected-priority-prefix terminal proof route."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.reference.protected_priority_prefix.construction import build_raw_protected_prefix
from formal_toolchain.reference.protected_priority_prefix.certificates import (
    build_raw_inheritance_certificate,
    build_raw_partition_certificate,
)
from .protocol import PreparedRouteAnalysis
from .raw_prefix_checkers import (
    check_abnormal_hi_only_switch,
    check_admissible_set_domination,
    check_arrival_projection,
    check_bad_prefix_reflection,
    check_complete_execution,
    check_completion_correspondence,
    check_deadline_correspondence,
    check_dispatch_correspondence,
    check_execution_existence_conformance,
    check_fixed_supply,
    check_full_idle_implies_raw_idle,
    check_input_independence,
    check_input_receptiveness,
    check_idle_only_recovery,
    check_instance_binding,
    check_job_key_binding,
    check_macrostep,
    check_mathematical_conformance,
    check_mode_order,
    check_n4_route_boundary_alignment,
    check_mode_transparent_lifecycle,
    check_no_saturation,
    check_parameter_inheritance,
    check_partition,
    check_priority_closure,
    check_raw_rta,
    check_raw_taskset_schedulable,
    check_recovery_order,
    check_reference_hi_safety,
    check_release_demand_receptiveness,
    check_release_fixed,
    check_runtime_schema,
    check_selected_safety,
    check_service_correspondence,
    check_switch_order,
    check_tail_pure_lo,
    check_verifier_soundness,
    check_weak_simulation,
)


class RawProtectedPrefixRoute:
    """Unsaturated V8 prefix with an explicit full-to-raw simulation proof."""

    route_id = "raw_protected_prefix"
    route_implementation_version = "raw-protected-prefix-v8"

    def registry_fragment_path(self) -> Path:
        return Path(__file__).parents[1] / "specs/routes/raw_protected_prefix_registry.json"

    def prepare_analysis(self, *, full_reference_taskset: Any,
                         reference_context_hash: str) -> PreparedRouteAnalysis:
        result = build_raw_protected_prefix(
            full_reference_taskset, source_context_hash=reference_context_hash)
        return PreparedRouteAnalysis(
            route_id=self.route_id,
            full_reference_taskset=full_reference_taskset,
            analysis_taskset=result.prefix_taskset,
            analysis_taskset_kind="raw_protected_prefix",
            route_implementation_version=self.route_implementation_version,
            construction_witnesses={"build_result": result},
            route_metadata={
                "analysis_taskset_kind": "raw_protected_prefix",
                "cutoff_task_name": result.cutoff_task_name,
                "cutoff_priority_index": result.cutoff_priority_index,
                "protected_count": len(result.protected_task_names),
                "tail_count": len(result.tail_task_names),
                "saturation_applied": False,
            },
        )

    def build_construction_certificates(self, *, prepared: PreparedRouteAnalysis,
                                        terminal_context_hash: str):
        result = prepared.construction_witnesses["build_result"]
        return {
            "RAW_PROTECTED_PRIORITY_PREFIX_PARTITION": build_raw_partition_certificate(
                result, context_hash=terminal_context_hash),
            "RAW_PREFIX_PARAMETER_INHERITANCE": build_raw_inheritance_certificate(
                result, context_hash=terminal_context_hash),
        }

    def checker_catalog(self) -> Mapping[str, Any]:
        return {
            "RAW_PROTECTED_PRIORITY_PREFIX_PARTITION": check_partition,
            "RAW_PREFIX_PARAMETER_INHERITANCE": check_parameter_inheritance,
            "RAW_PREFIX_NO_SATURATION_BINDING": check_no_saturation,
            "RAW_PREFIX_TAIL_PURE_LO": check_tail_pure_lo,
            "RAW_PREFIX_PRIORITY_CLOSURE": check_priority_closure,
            "RAW_PREFIX_JOB_KEY_BINDING": check_job_key_binding,
            "RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE": check_runtime_schema,
            "RAW_PREFIX_RELEASE_FIXED_POSITIVE_DEMAND": check_release_fixed,
            "RAW_PREFIX_PROTECTED_INPUT_RECEPTIVENESS": check_input_receptiveness,
            "RAW_PREFIX_PROTECTED_INPUT_INDEPENDENCE": check_input_independence,
            "RAW_PREFIX_MODE_TRANSPARENT_LIFECYCLE": check_mode_transparent_lifecycle,
            "RAW_PREFIX_IDLE_ONLY_RECOVERY": check_idle_only_recovery,
            "RAW_PREFIX_ABNORMAL_HI_ONLY_SWITCH": check_abnormal_hi_only_switch,
            "RAW_PREFIX_FIXED_PROCESSOR_SUPPLY": check_fixed_supply,
            "RAW_PREFIX_FULL_IDLE_IMPLIES_RAW_IDLE": check_full_idle_implies_raw_idle,
            "RAW_PREFIX_RECOVERY_ORDER_PRESERVATION": check_recovery_order,
            "RAW_PREFIX_SWITCH_ORDER_PRESERVATION": check_switch_order,
            "RAW_PREFIX_MODE_ORDER_INVARIANT": check_mode_order,
            "RAW_PREFIX_ADMISSIBLE_SET_DOMINATION": check_admissible_set_domination,
            "RAW_PREFIX_RELEASE_DEMAND_RECEPTIVENESS": check_release_demand_receptiveness,
            "RAW_PREFIX_PROTECTED_ARRIVAL_BATCH_PROJECTION": check_arrival_projection,
            "RAW_PREFIX_SERVICE_CORRESPONDENCE": check_service_correspondence,
            "RAW_PREFIX_COMPLETION_CORRESPONDENCE": check_completion_correspondence,
            "RAW_PREFIX_DEADLINE_BATCH_CORRESPONDENCE": check_deadline_correspondence,
            "RAW_PREFIX_TOTAL_FINAL_DISPATCH_CORRESPONDENCE": check_dispatch_correspondence,
            "RAW_PREFIX_CLOSE_TO_CLOSE_MACROSTEP": check_macrostep,
            "RAW_PREFIX_EXECUTION_EXISTENCE_CONFORMANCE": check_execution_existence_conformance,
            "RAW_PREFIX_COMPLETE_EXECUTION_EXISTENCE": check_complete_execution,
            "RAW_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED": check_weak_simulation,
            "N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT": check_n4_route_boundary_alignment,
            "RAW_PREFIX_HI_BAD_PREFIX_REFLECTION": check_bad_prefix_reflection,
            "RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC": check_raw_rta,
            "RAW_PREFIX_VERIFIER_SOUNDNESS": check_verifier_soundness,
            "RAW_PREFIX_INSTANCE_EVIDENCE_BINDING": check_instance_binding,
            "RAW_PREFIX_MATHEMATICAL_CONFORMANCE": check_mathematical_conformance,
            "RAW_PREFIX_TASKSET_SCHEDULABLE": check_raw_taskset_schedulable,
            "REFERENCE_HI_SAFETY_FROM_RAW_PREFIX": check_reference_hi_safety,
            "SELECTED_REFERENCE_HI_SAFETY": check_selected_safety,
        }


ROUTE = RawProtectedPrefixRoute()
