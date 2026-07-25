from __future__ import annotations

import pytest

from formal_toolchain.routes.registry import resolve_registry
from formal_toolchain.verifier.checker_catalog import checker_for


def _by_id(route: str):
    return {row["id"]: row for row in resolve_registry(route).entries}


def test_both_routes_keep_concrete_to_full_reference_bad_prefix_chain():
    required = {
        "BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION",
        "CLOSED_PREFIX_REFINEMENT",
        "REFERENCE_PREFIX_EXTENSION",
        "HI_BAD_CLOSED_PREFIX_REFLECTION",
        "FINITE_BAD_PREFIX_CONTRADICTION",
    }
    for route in ("strict_full", "protected_prefix"):
        assert required <= set(_by_id(route))


def test_protected_prefix_rta_and_theorem_nodes_bind_the_transformed_taskset():
    by_id = _by_id("protected_prefix")
    assert "SATURATED_PROTECTED_PREFIX_REFERENCE" in by_id[
        "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC"
    ]["depends_on"]
    deps = set(by_id["PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE"]["depends_on"])
    assert "PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE" in deps
    assert "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC" in deps
    assert "THEORY_LIBRARY_VERSION" in deps
    assert len(deps) == 3


def test_route_checker_collision_is_rejected_before_dispatch():
    class BadRoute:
        @staticmethod
        def checker_catalog():
            return {"REFERENCE_TASKSET": lambda **_: {"status": "PASS"}}

    with pytest.raises(ValueError, match="ROUTE_CHECKER_ID_COLLISION"):
        checker_for("REFERENCE_TASKSET", route_strategy=BadRoute())


def test_unresolved_route_checker_requires_exact_predecessor_set():
    from formal_toolchain.routes.checkers import unresolved_derived_checker

    checker = unresolved_derived_checker("X", expected_predecessors=("A",))
    result = checker(predecessors={"A": {"status": "PASS"}, "B": {"status": "PASS"}})
    assert result["status"] == "UNRESOLVED"
    assert result["code"] == "PREDECESSOR_SET_MISMATCH"
    assert result["witness"]["extra"] == ["B"]


def _sample_full_taskset():
    from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset

    return ReferenceTaskset((
        ReferenceTask("lo_protected", 20, 20, 4, 2, "LO", 0, 3, 3, 2, 0),
        ReferenceTask("hi_cutoff", 25, 25, 2, 5, "HI", 1, 2, 5, None, 0),
        ReferenceTask("lo_tail", 40, 40, 3, 1, "LO", 2, 2, 2, 1, 0),
    ), "a" * 64)


def test_protected_route_registers_executable_base_checkers_and_only_certificates():
    from types import SimpleNamespace

    from formal_toolchain.routes.resolver import resolve_route

    route = resolve_route("protected_prefix")
    prepared = route.prepare_analysis(
        full_reference_taskset=_sample_full_taskset(),
        reference_context_hash="a" * 64,
    )
    certificates = route.build_construction_certificates(
        prepared=prepared, terminal_context_hash="b" * 64)
    assert set(certificates) == {
        "PROTECTED_PRIORITY_PREFIX_PARTITION",
        "SATURATED_PROTECTED_PREFIX_REFERENCE",
    }
    assert all(isinstance(value, dict) for value in certificates.values())
    catalog = route.checker_catalog()
    required = {
        "PROTECTED_PRIORITY_PREFIX_PARTITION",
        "SATURATED_PROTECTED_PREFIX_REFERENCE",
        "PROTECTED_PREFIX_PARAMETER_PRESERVATION",
        "PROTECTED_PREFIX_LO_SATURATION",
        "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE",
        "SELECTED_REFERENCE_HI_SAFETY",
    }
    assert required <= set(catalog)

    state = SimpleNamespace(
        prepared_route=prepared,
        route_construction_certificates=certificates,
        full_reference_taskset=prepared.full_reference_taskset,
        analysis_taskset=prepared.analysis_taskset,
    )
    context = SimpleNamespace(fresh_state=state)
    assert catalog["PROTECTED_PRIORITY_PREFIX_PARTITION"](context=context)["status"] == "PASS"
    assert catalog["SATURATED_PROTECTED_PREFIX_REFERENCE"](context=context)["status"] == "PASS"
    assert catalog["PROTECTED_PREFIX_PARAMETER_PRESERVATION"](context=context)["status"] == "PASS"
    assert catalog["PROTECTED_PREFIX_LO_SATURATION"](context=context)["status"] == "PASS"


def test_protected_rta_replay_compares_candidate_to_full_reference(monkeypatch):
    from types import SimpleNamespace

    from formal_toolchain.routes.resolver import resolve_route
    from formal_toolchain.verifier.recompute import _rta_replay

    full = _sample_full_taskset()
    prepared = resolve_route("protected_prefix").prepare_analysis(
        full_reference_taskset=full, reference_context_hash="a" * 64)
    state = SimpleNamespace(
        analysis_taskset=prepared.analysis_taskset,
        full_reference_taskset=full,
        selected_route_id="protected_prefix",
        selected_rta_obligation_id="PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        terminal_route_context={"hash": "b" * 64},
    )

    monkeypatch.setattr(
        "formal_toolchain.reference.rta_production.all_task_protected_prefix_rta",
        lambda taskset, certificate_context_hash=None: {
            "obligation_id": "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
            "route_id": "protected_prefix",
        })
    monkeypatch.setattr(
        "formal_toolchain.reference.rta_replay.replay_all_task_rta",
        lambda taskset, production, expected_obligation_id=None, expected_route_id=None: {
            "status": "PASS", "witness": {"all_tasks_covered": True}})
    candidate = {
        "REFERENCE_TASKSET": {
            "witness": {"evidence": {"taskset": full.to_dict()}}
        }
    }
    result = _rta_replay(
        inputs=SimpleNamespace(), certified_envelope={}, candidate=candidate,
        fresh_reference=full, fresh_state=state)
    assert result["status"] == "PASS"
    assert result["analysis_taskset"]["fingerprint"] == prepared.analysis_taskset.to_dict()["fingerprint"]


def test_protected_route_candidate_rta_uses_saturated_analysis_taskset():
    """A failing low-priority LO tail must not erase prefix Case1/Case2 domains."""
    from formal_toolchain.reference.rta_certificate import build_rta_composite
    from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
    from formal_toolchain.routes.resolver import resolve_route

    full = ReferenceTaskset((
        ReferenceTask("lo_protected", 20, 20, 2, 1, "LO", 0, 1, 1, 1, 0),
        ReferenceTask("hi_cutoff", 25, 25, 3, 5, "HI", 1, 3, 5, None, 0),
        # Intentionally unschedulable full-reference LO tail.
        ReferenceTask("lo_tail", 25, 25, 30, 1, "LO", 2, 1, 1, 1, 0),
    ), "a" * 64)
    prepared = resolve_route("protected_prefix").prepare_analysis(
        full_reference_taskset=full,
        reference_context_hash="a" * 64,
    )

    strict = build_rta_composite(full, route_id="strict_full")
    protected = build_rta_composite(
        prepared.analysis_taskset, route_id="protected_prefix")

    assert strict["status"] == "FAIL"
    assert protected["status"] == "PASS"
    production = protected["production"]
    assert production["route_id"] == "protected_prefix"
    assert production["obligation_id"] == "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC"
    assert production["task_order"] == ["lo_protected", "hi_cutoff"]
    assert production["complete_integer_candidate_domains"] is True
    for row in production["tasks"]:
        assert [entry["start"] for entry in row["case1"]] == list(range(row["r_lo"]))
