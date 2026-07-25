from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.rta_production import all_task_protected_prefix_rta
from formal_toolchain.reference.rta_replay import replay_all_task_rta
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.routes.resolver import resolve_route


def _envelope(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "obligation_status": result["status"],
        "witness": result.get("witness", {}),
        "artifact_hash": sha256_object(result),
    }


def test_protected_prefix_parameterized_chain_reaches_selected_reference_hi_safety() -> None:
    """Exercise the complete seed-local PPP DAG with a real all-task RTA receipt.

    This is intentionally a small schedulable reference taskset.  The test does
    not mock the protected-prefix RTA or the route-local dynamic theorems.  It
    verifies that all PPP obligations can be constructed in dependency order,
    which is the prerequisite for entering real-seed debugging.
    """

    full_reference = ReferenceTaskset(
        (
            ReferenceTask("lo_protected", 20, 20, 4, 2, "LO", 0, 3, 3, 2, 0),
            ReferenceTask("hi_cutoff", 25, 25, 2, 5, "HI", 1, 2, 5, None, 0),
            ReferenceTask("lo_tail", 40, 40, 3, 1, "LO", 2, 2, 2, 1, 0),
        ),
        "a" * 64,
    )
    route = resolve_route("protected_prefix")
    prepared = route.prepare_analysis(
        full_reference_taskset=full_reference,
        reference_context_hash="a" * 64,
    )
    construction = route.build_construction_certificates(
        prepared=prepared,
        terminal_context_hash="b" * 64,
    )
    production_rta = all_task_protected_prefix_rta(
        prepared.analysis_taskset,
        certificate_context_hash="d" * 64,
    )
    replay = replay_all_task_rta(
        prepared.analysis_taskset,
        production_rta,
        expected_obligation_id="PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        expected_route_id="protected_prefix",
    )
    fresh_state = SimpleNamespace(
        prepared_route=prepared,
        route_construction_certificates=construction,
        full_reference_taskset=full_reference,
        analysis_taskset=prepared.analysis_taskset,
        selected_route_id="protected_prefix",
        selected_rta_obligation_id="PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        fresh_rta_replay=replay,
    )
    context = SimpleNamespace(fresh_state=fresh_state)
    catalog = route.checker_catalog()
    verified: dict[str, dict[str, Any]] = {}

    def run(obligation_id: str, dependencies: tuple[str, ...] = ()) -> None:
        result = catalog[obligation_id](
            context=context,
            verified_predecessors={dep: verified[dep] for dep in dependencies},
        )
        assert result["status"] == "PASS", (obligation_id, result)
        verified[obligation_id] = _envelope(result)

    for obligation_id in (
        "PROTECTED_PRIORITY_PREFIX_PARTITION",
        "SATURATED_PROTECTED_PREFIX_REFERENCE",
    ):
        run(obligation_id)

    run("PROTECTED_PREFIX_PARAMETER_PRESERVATION", ("SATURATED_PROTECTED_PREFIX_REFERENCE",))
    run("PROTECTED_PREFIX_LO_SATURATION", ("SATURATED_PROTECTED_PREFIX_REFERENCE",))
    run("PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE", ("SATURATED_PROTECTED_PREFIX_REFERENCE",))

    model_conditions = (
        "FINITE_INDEPENDENT_PERIODIC_SUBLANGUAGE",
        "RELEASE_FIXED_DEMAND_DOMINATION",
        "REFERENCE_PREFIX_EXTENSIBILITY",
        "STANDARD_EMPTY_LO_INITIALIZATION",
        "REFERENCE_TRANSITION_SYSTEM_IDENTITY",
    )
    verified["REFERENCE_MODEL_CONFORMANCE"] = {
        "obligation_status": "PASS",
        "artifact_hash": "c" * 64,
        "witness": {
            "condition_results": [
                {"condition_id": condition, "passed": True}
                for condition in model_conditions
            ]
        },
    }
    verified["ZERO_RELATIVE_START"] = {
        "obligation_status": "PASS",
        "artifact_hash": "e" * 64,
        "witness": {"status": "PASS"},
    }
    verified["THEORY_LIBRARY_VERSION"] = {
        "obligation_status": "PASS",
        "artifact_hash": "f" * 64,
        "witness": {"status": "PASS"},
    }

    run("FULL_REFERENCE_RECURRING_INPUT_ORACLE", ("REFERENCE_MODEL_CONFORMANCE",))
    run(
        "PROTECTED_INPUT_STREAM_PROJECTION",
        ("FULL_REFERENCE_RECURRING_INPUT_ORACLE", "SATURATED_PROTECTED_PREFIX_REFERENCE"),
    )
    run(
        "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        ("PROTECTED_INPUT_STREAM_PROJECTION", "PROTECTED_PREFIX_LO_SATURATION"),
    )
    run("PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION", ("SATURATED_PROTECTED_PREFIX_REFERENCE",))
    run(
        "PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES",
        ("PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE", "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION"),
    )
    run(
        "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL",
        (
            "PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES",
            "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        ),
    )
    run("PROTECTED_PREFIX_TIME_DIVERGENCE", ("PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL",))
    run(
        "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        ("PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE", "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION"),
    )
    run(
        "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
        (
            "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL",
            "PROTECTED_PREFIX_TIME_DIVERGENCE",
            "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        ),
    )
    run(
        "PROTECTED_PREFIX_INITIAL_RELATION",
        ("PROTECTED_PRIORITY_PREFIX_PARTITION", "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"),
    )

    run(
        "PPP_L1_TAIL_SERVICE_EXCLUSION",
        ("PROTECTED_PRIORITY_PREFIX_PARTITION", "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"),
    )
    run(
        "PPP_L2_FINAL_DISPATCH_CORRESPONDENCE",
        ("PPP_L1_TAIL_SERVICE_EXCLUSION", "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"),
    )
    run(
        "PPP_L3_SERVICE_CORRESPONDENCE",
        (
            "PPP_L1_TAIL_SERVICE_EXCLUSION",
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        ),
    )
    run("PPP_L4_COMPLETION_REMOVAL_CORRESPONDENCE", ("PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",))
    run(
        "PPP_L5_DEADLINE_BATCH_FOLD",
        ("PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE", "PROTECTED_INPUT_STREAM_PROJECTION"),
    )
    run(
        "PPP_L6_ARRIVAL_BATCH_FOLD",
        (
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        ),
    )
    run(
        "PPP_L7_CANONICAL_PHASE_JOIN",
        ("PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE", "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION"),
    )
    run(
        "PROTECTED_MACRO_STEP_PRESERVATION",
        (
            "PPP_L1_TAIL_SERVICE_EXCLUSION",
            "PPP_L2_FINAL_DISPATCH_CORRESPONDENCE",
            "PPP_L3_SERVICE_CORRESPONDENCE",
            "PPP_L4_COMPLETION_REMOVAL_CORRESPONDENCE",
            "PPP_L5_DEADLINE_BATCH_FOLD",
            "PPP_L6_ARRIVAL_BATCH_FOLD",
            "PPP_L7_CANONICAL_PHASE_JOIN",
        ),
    )
    run(
        "FULL_TO_PREFIX_SIMULATION_DOMAIN",
        (
            "REFERENCE_MODEL_CONFORMANCE",
            "PROTECTED_PRIORITY_PREFIX_PARTITION",
            "PROTECTED_PREFIX_LO_SATURATION",
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
            "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
        ),
    )
    run(
        "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED",
        (
            "FULL_TO_PREFIX_SIMULATION_DOMAIN",
            "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
            "PROTECTED_PREFIX_INITIAL_RELATION",
            "PROTECTED_MACRO_STEP_PRESERVATION",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        ),
    )
    run(
        "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
        ("PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED", "PPP_L5_DEADLINE_BATCH_FOLD"),
    )
    run("PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC")
    run(
        "PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE",
        (
            "SATURATED_PROTECTED_PREFIX_REFERENCE",
            "PROTECTED_PREFIX_PARAMETER_PRESERVATION",
            "PROTECTED_PREFIX_LO_SATURATION",
            "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
            "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
            "ZERO_RELATIVE_START",
        ),
    )
    run(
        "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE",
        (
            "PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE",
            "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
            "THEORY_LIBRARY_VERSION",
        ),
    )
    run(
        "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
        ("PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION", "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE"),
    )
    run("SELECTED_REFERENCE_HI_SAFETY", ("REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",))
