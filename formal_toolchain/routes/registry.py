"""Resolve one route-specific mathematical closure without activating the other."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.registry import load_registry, validate_registry
from .config import ProofRoute


@dataclass(frozen=True, slots=True)
class ResolvedRegistry:
    route_id: str
    common_entries: tuple[dict[str, Any], ...]
    route_entries: tuple[dict[str, Any], ...]
    entries: tuple[dict[str, Any], ...]
    common_fingerprint: str
    route_fingerprint: str
    resolved_fingerprint: str


def _entry_map() -> dict[str, dict[str, Any]]:
    path = Path(__file__).parents[1] / "specs/obligation_registry.json"
    return {str(item["id"]): item for item in load_registry(path)}


def _template(source: dict[str, Any], obligation_id: str, *, depends_on: list[str],
              context_layer: str = "terminal_route_context", producer_id: str | None = None) -> dict[str, Any]:
    item = copy.deepcopy(source)
    item["id"] = obligation_id
    item["depends_on"] = depends_on
    item["context_layer"] = context_layer
    item["producer"] = {"kind": "compiler_candidate", "id": producer_id or obligation_id.lower() + "_v1"}
    item["artifact"] = f"artifacts/{obligation_id.lower()}.json"
    item["summary_path"] = f"obligations.{obligation_id}"
    return item


def resolve_registry(route: ProofRoute | str) -> ResolvedRegistry:
    route_id = route.value if isinstance(route, ProofRoute) else str(route)
    if route_id not in {"strict_full", "protected_prefix"}:
        raise ValueError("PROOF_ROUTE_INVALID")
    source = _entry_map()
    root = copy.deepcopy(source["FINAL_CLAIM_COMPOSITION"])
    finite = copy.deepcopy(source["FINITE_BAD_PREFIX_CONTRADICTION"])
    # The common tail has one root and only refers to the abstract selected node.
    finite["depends_on"] = ["SELECTED_REFERENCE_HI_SAFETY", "HI_BAD_CLOSED_PREFIX_REFLECTION"]
    root["depends_on"] = ["FINITE_BAD_PREFIX_CONTRADICTION"]
    terminal_source = source["REFERENCE_HI_SUBSET_SAFETY"]
    selected = _template(terminal_source, "SELECTED_REFERENCE_HI_SAFETY",
                         depends_on=["REFERENCE_HI_SUBSET_SAFETY"] if route_id == "strict_full"
                         else ["REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX"])
    common_ids = {
        item["id"] for item in source.values()
        if item["id"] not in {
            "ALL_TASK_REFERENCE_RTA_ARITHMETIC", "REFERENCE_TASKSET_SCHEDULABLE",
            "REFERENCE_HI_SUBSET_SAFETY",
            "FINITE_BAD_PREFIX_CONTRADICTION", "FINAL_CLAIM_COMPOSITION",
        }
    }
    common = [copy.deepcopy(source[item]) for item in sorted(common_ids)]
    common.extend([finite, root])
    if route_id == "strict_full":
        route = [copy.deepcopy(source[item]) for item in (
            "ALL_TASK_REFERENCE_RTA_ARITHMETIC", "REFERENCE_TASKSET_SCHEDULABLE",
            "REFERENCE_HI_SUBSET_SAFETY")]
        route.append(selected)
    else:
        prefix_ids = [
            "PROTECTED_PRIORITY_PREFIX_PARTITION", "SATURATED_PROTECTED_PREFIX_REFERENCE",
            "PROTECTED_PREFIX_PARAMETER_PRESERVATION", "PROTECTED_PREFIX_LO_SATURATION",
            "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC", "PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE",
            "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE",
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "FULL_REFERENCE_RECURRING_INPUT_ORACLE", "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
            "PPP_L1_TAIL_SERVICE_EXCLUSION", "PPP_L2_FINAL_DISPATCH_CORRESPONDENCE",
            "PPP_L3_SERVICE_CORRESPONDENCE", "PPP_L4_COMPLETION_REMOVAL_CORRESPONDENCE",
            "PPP_L5_DEADLINE_BATCH_FOLD", "PPP_L6_ARRIVAL_BATCH_FOLD",
            "PPP_L7_CANONICAL_PHASE_JOIN", "PROTECTED_MACRO_STEP_PRESERVATION",
            "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
            "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL",
            "PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES",
            "PROTECTED_PREFIX_TIME_DIVERGENCE",
            "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
            "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
            "PROTECTED_PREFIX_INITIAL_RELATION",
            "FULL_TO_PREFIX_SIMULATION_DOMAIN",
            "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED", "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
            "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
        ]
        rta_deps = list(source["ALL_TASK_REFERENCE_RTA_ARITHMETIC"]["depends_on"])
        route = [_template(terminal_source, item, depends_on=[],
                           producer_id=item.lower() + "_v1") for item in prefix_ids]
        by = {item["id"]: item for item in route}
        by["SATURATED_PROTECTED_PREFIX_REFERENCE"]["depends_on"] = ["PROTECTED_PRIORITY_PREFIX_PARTITION"]
        by["PROTECTED_PREFIX_PARAMETER_PRESERVATION"]["depends_on"] = ["SATURATED_PROTECTED_PREFIX_REFERENCE"]
        by["PROTECTED_PREFIX_LO_SATURATION"]["depends_on"] = ["SATURATED_PROTECTED_PREFIX_REFERENCE"]
        # The reused all-task arithmetic must be bound to the transformed prefix,
        # not merely to the full reference taskset inherited from the common DAG.
        by["PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC"]["depends_on"] = [
            "SATURATED_PROTECTED_PREFIX_REFERENCE", *rta_deps,
        ]
        # This node discharges prefix-specific reference model conformance.
        # It proves that the saturated prefix satisfies all model assumptions
        # required by the C-AMC-sem all-task theorem; it is NOT a copy of
        # the full-reference conformance.
        by["PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE"]["depends_on"] = [
            "SATURATED_PROTECTED_PREFIX_REFERENCE",
            "PROTECTED_PREFIX_PARAMETER_PRESERVATION",
            "PROTECTED_PREFIX_LO_SATURATION",
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
            "ZERO_RELATIVE_START",
        ]
        # Input oracle chain: full oracle -> projection -> demand receptiveness
        by["FULL_REFERENCE_RECURRING_INPUT_ORACLE"]["depends_on"] = [
            "REFERENCE_MODEL_CONFORMANCE",
        ]
        by["PROTECTED_INPUT_STREAM_PROJECTION"]["depends_on"] = [
            "FULL_REFERENCE_RECURRING_INPUT_ORACLE",
            "SATURATED_PROTECTED_PREFIX_REFERENCE",
        ]
        by["PROTECTED_INPUT_DEMAND_RECEPTIVENESS"]["depends_on"] = [
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_PREFIX_LO_SATURATION",
        ]
        # L1-L8 are explicit mathematical obligations.  Their dependencies
        # are kept in the route DAG; no theorem is hidden in derived metadata.
        by["PPP_L1_TAIL_SERVICE_EXCLUSION"]["depends_on"] = [
            "PROTECTED_PRIORITY_PREFIX_PARTITION",
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        ]
        by["PPP_L2_FINAL_DISPATCH_CORRESPONDENCE"]["depends_on"] = [
            "PPP_L1_TAIL_SERVICE_EXCLUSION",
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        ]
        by["PPP_L3_SERVICE_CORRESPONDENCE"]["depends_on"] = [
            "PPP_L1_TAIL_SERVICE_EXCLUSION",
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        ]
        by["PPP_L4_COMPLETION_REMOVAL_CORRESPONDENCE"]["depends_on"] = [
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        ]
        by["PPP_L5_DEADLINE_BATCH_FOLD"]["depends_on"] = [
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_INPUT_STREAM_PROJECTION",
        ]
        by["PPP_L6_ARRIVAL_BATCH_FOLD"]["depends_on"] = [
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        ]
        by["PPP_L7_CANONICAL_PHASE_JOIN"]["depends_on"] = [
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        ]
        by["PROTECTED_MACRO_STEP_PRESERVATION"]["depends_on"] = [
            "PPP_L1_TAIL_SERVICE_EXCLUSION", "PPP_L2_FINAL_DISPATCH_CORRESPONDENCE",
            "PPP_L3_SERVICE_CORRESPONDENCE", "PPP_L4_COMPLETION_REMOVAL_CORRESPONDENCE",
            "PPP_L5_DEADLINE_BATCH_FOLD", "PPP_L6_ARRIVAL_BATCH_FOLD",
            "PPP_L7_CANONICAL_PHASE_JOIN",
        ]
        # Prefix-specific prefix-extension (fresh generated, not full-reference copy)
        by["PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION"]["depends_on"] = [
            "SATURATED_PROTECTED_PREFIX_REFERENCE",
        ]
        # Complete prefix execution existence chain.  The order is intentionally
        # non-circular: local closure termination precedes successor totality.
        by["PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES"]["depends_on"] = [
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
        ]
        by["PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL"]["depends_on"] = [
            "PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES",
            "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        ]
        by["PROTECTED_PREFIX_TIME_DIVERGENCE"]["depends_on"] = [
            "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL",
        ]
        # Idle-jump stuttering is a local transition/frame theorem.  It must be
        # independent of the complete-execution witness to avoid a witness cycle.
        by["PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION"]["depends_on"] = [
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
        ]
        by["PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS"]["depends_on"] = [
            "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL",
            "PROTECTED_PREFIX_TIME_DIVERGENCE",
            "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        ]
        by["PROTECTED_PREFIX_INITIAL_RELATION"]["depends_on"] = [
            "PROTECTED_PRIORITY_PREFIX_PARTITION",
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        ]
        by["PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE"]["depends_on"] = [
            "PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE",
            "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
            "THEORY_LIBRARY_VERSION",
        ]
        by["PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"]["depends_on"] = [
            "SATURATED_PROTECTED_PREFIX_REFERENCE",
        ]
        by["FULL_TO_PREFIX_SIMULATION_DOMAIN"]["depends_on"] = [
            "REFERENCE_MODEL_CONFORMANCE",
            "PROTECTED_PRIORITY_PREFIX_PARTITION",
            "PROTECTED_PREFIX_LO_SATURATION",
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
            "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
        ]
        by["PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED"]["depends_on"] = [
            "FULL_TO_PREFIX_SIMULATION_DOMAIN",
            "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
            "PROTECTED_PREFIX_INITIAL_RELATION",
            "PROTECTED_MACRO_STEP_PRESERVATION",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        ]
        by["PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION"]["depends_on"] = [
            "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED",
            "PPP_L5_DEADLINE_BATCH_FOLD",
        ]
        by["REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX"]["depends_on"] = ["PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION", "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE"]
        # Concrete-to-full-reference N6 remains the shared bridge predecessor of
        # FINITE_BAD_PREFIX_CONTRADICTION.  The prefix reflection is used only to
        # derive full-reference safety and must never replace N6.
        route = [by[item] for item in prefix_ids] + [selected]
    # Keep only the resolved claim closure.  The source registry contains
    # archived/alternative obligations; carrying them into a route would make
    # the non-selected route an accidental active predecessor.
    all_entries = common + route
    by_id = {str(item["id"]): item for item in all_entries}
    reachable: set[str] = set()
    def visit(oid: str) -> None:
        if oid in reachable:
            return
        if oid not in by_id:
            raise ValueError(f"{oid} 依赖未知 obligation")
        reachable.add(oid)
        for dep in by_id[oid].get("depends_on", []):
            visit(str(dep))
    visit("FINAL_CLAIM_COMPOSITION")
    route_ids = {str(item["id"]) for item in route}
    common = [item for item in all_entries if str(item["id"]) in reachable and str(item["id"]) not in route_ids]
    legacy_ids = {
        "PROTECTED_HI_RTA_ARITHMETIC", "PER_HI_TASK_INDUCTIVE_WCRT",
        "PROTECTED_HI_SAFETY_COROLLARY",
    }
    common = [item for item in common if str(item["id"]) not in legacy_ids]
    # Structural gates are claim-wide and are not mathematical predecessors;
    # retain them in every resolved registry so bundle verification remains
    # route-independent.
    retained = {str(item["id"]) for item in common}
    structural_ids = {
        "ARTIFACT_MANIFEST", "COMPONENT_CONTEXT_INTEGRITY", "DIRECT_PREDECESSOR_HASHES",
        "STATUS_EVIDENCE", "OUTER_BUNDLE_ROOT", "INDEPENDENT_BUNDLE_VERIFICATION",
        "CLAIM_AGGREGATION_RESULT",
    }
    for item in source.values():
        if (item["id"] not in legacy_ids and
                (item.get("proof_role") in {"structural_gate", "authorization_gate"}
                 or item["id"] in structural_ids) and item["id"] not in retained):
            common.append(copy.deepcopy(item))
            retained.add(str(item["id"]))
    route = [item for item in route if str(item["id"]) in reachable]
    entries = tuple(common + route)
    # validate_registry is intentionally applied to the resolved graph, not the
    # user's candidate bundle.  This catches duplicate IDs and dangling edges.
    validate_registry(list(entries))
    # The common fingerprint is deliberately computed from the same published
    # common source set for both routes; route pruning must not contaminate the
    # shared bootstrap/full-reference identity.
    common_source_excluded = {
        "ALL_TASK_REFERENCE_RTA_ARITHMETIC", "REFERENCE_TASKSET_SCHEDULABLE",
        "REFERENCE_HI_SUBSET_SAFETY",
        "FINITE_BAD_PREFIX_CONTRADICTION", "FINAL_CLAIM_COMPOSITION",
        "PROTECTED_PRIORITY_PREFIX_PARTITION", "SATURATED_PROTECTED_PREFIX_REFERENCE",
        "PROTECTED_PREFIX_PARAMETER_PRESERVATION", "PROTECTED_PREFIX_LO_SATURATION",
        "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC", "PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE",
        "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE",
        "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        "FULL_REFERENCE_RECURRING_INPUT_ORACLE", "PROTECTED_INPUT_STREAM_PROJECTION",
        "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
        "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL",
        "PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES",
        "PROTECTED_PREFIX_TIME_DIVERGENCE",
        "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
        "FULL_TO_PREFIX_SIMULATION_DOMAIN",
            "PROTECTED_MACRO_STEP_PRESERVATION", "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED", "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
        "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX", "SELECTED_REFERENCE_HI_SAFETY",
    }
    common_fp = sha256_object([source[item] for item in sorted(source) if item not in common_source_excluded])
    route_fp = sha256_object(route)
    return ResolvedRegistry(route_id, tuple(common), tuple(route), entries,
                            common_fp, route_fp, sha256_object(entries))
