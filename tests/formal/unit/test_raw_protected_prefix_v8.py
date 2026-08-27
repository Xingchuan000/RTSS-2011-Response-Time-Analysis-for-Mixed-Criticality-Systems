from __future__ import annotations

from formal_toolchain.core.contexts import context_layer_for_obligation
from formal_toolchain.reference.rta_certificate import build_rta_composite
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.routes.config import ProofRoute, route_config
from formal_toolchain.routes.registry import resolve_registry
from formal_toolchain.routes.resolver import resolve_route


def _sample() -> ReferenceTaskset:
    return ReferenceTaskset((
        ReferenceTask("lo_protected", 20, 20, 4, 2, "LO", 0, 3, 3, 2, 0),
        ReferenceTask("hi_cutoff", 25, 25, 2, 5, "HI", 1, 2, 5, None, 0),
        ReferenceTask("lo_tail", 40, 40, 30, 1, "LO", 2, 2, 2, 1, 0),
    ), "a" * 64)


def test_raw_route_is_third_distinct_route_and_keeps_common_fingerprint():
    assert route_config("raw_protected_prefix").route is ProofRoute.RAW_PROTECTED_PREFIX
    fps = {r: resolve_registry(r) for r in ("strict_full", "raw_protected_prefix", "protected_prefix")}
    assert len({x.common_fingerprint for x in fps.values()}) == 1
    assert len({x.route_fingerprint for x in fps.values()}) == 3


def test_raw_prefix_deletes_only_lo_tail_and_does_not_saturate_wcet():
    full = _sample()
    prepared = resolve_route("raw_protected_prefix").prepare_analysis(
        full_reference_taskset=full, reference_context_hash="a" * 64)
    raw = prepared.analysis_taskset
    assert [t.name for t in raw.tasks] == ["lo_protected", "hi_cutoff"]
    assert raw.tasks[0].c_lo == 4
    assert raw.tasks[0].c_hi == 2
    assert raw.tasks[0].c_lo != raw.tasks[0].c_hi
    result = prepared.construction_witnesses["build_result"]
    assert result.inheritance_witness["all_parameters_inherited"] is True
    assert result.inheritance_witness["no_saturation_applied"] is True
    assert result.partition_witness["tail_all_lo"] is True
    assert prepared.analysis_taskset_kind == "raw_protected_prefix"


def test_raw_route_registers_v8_derived_chain_with_non_circular_domain():
    by = {e["id"]: e for e in resolve_registry("raw_protected_prefix").entries}
    assert set(by["RAW_PREFIX_MODE_ORDER_INVARIANT"]["depends_on"]) == {
        "RAW_PREFIX_RECOVERY_ORDER_PRESERVATION",
        "RAW_PREFIX_SWITCH_ORDER_PRESERVATION",
        "RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
    }
    weak = set(by["RAW_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED"]["depends_on"])
    assert "REFERENCE_MODEL_CONFORMANCE" in weak
    assert "RAW_PREFIX_EXECUTION_EXISTENCE_CONFORMANCE" in weak
    bad = set(by["RAW_PREFIX_HI_BAD_PREFIX_REFLECTION"]["depends_on"])
    assert "N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT" in bad
    assert set(by["RAW_PREFIX_TASKSET_SCHEDULABLE"]["depends_on"]) == {
        "RAW_PREFIX_MATHEMATICAL_CONFORMANCE", "RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        "RAW_PREFIX_VERIFIER_SOUNDNESS", "RAW_PREFIX_INSTANCE_EVIDENCE_BINDING",
    }
    assert set(by["REFERENCE_HI_SAFETY_FROM_RAW_PREFIX"]["depends_on"]) == {
        "RAW_PREFIX_HI_BAD_PREFIX_REFLECTION", "RAW_PREFIX_TASKSET_SCHEDULABLE",
    }
    assert by["SELECTED_REFERENCE_HI_SAFETY"]["depends_on"] == ["REFERENCE_HI_SAFETY_FROM_RAW_PREFIX"]


def test_raw_rta_runs_on_unsaturated_analysis_taskset_and_is_independently_labeled():
    prepared = resolve_route("raw_protected_prefix").prepare_analysis(
        full_reference_taskset=_sample(), reference_context_hash="a" * 64)
    composite = build_rta_composite(prepared.analysis_taskset, route_id="raw_protected_prefix")
    production = composite["production"]
    assert production["route_id"] == "raw_protected_prefix"
    assert production["obligation_id"] == "RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC"
    assert production["task_order"] == ["lo_protected", "hi_cutoff"]
    assert production["taskset"] == prepared.analysis_taskset.to_dict()


def test_raw_obligations_are_terminal_route_context_bound():
    for oid in (
        "RAW_PREFIX_MODE_ORDER_INVARIANT",
        "RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        "RAW_PREFIX_TASKSET_SCHEDULABLE",
        "N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT",
        "REFERENCE_HI_SAFETY_FROM_RAW_PREFIX",
    ):
        assert context_layer_for_obligation(oid) == "terminal_route_context"


def test_v8_auto_orchestrator_keeps_route_bundles_isolated_and_uses_order(tmp_path, monkeypatch):
    import importlib
    mod = importlib.import_module("formal_toolchain.workflow.prove_seed_v8")

    calls = []

    def fake_prove_seed(**kwargs):
        route = kwargs["proof_route"]
        calls.append(route)
        out = kwargs["out"]
        (out / "verified").mkdir(parents=True, exist_ok=True)
        (out / "verified" / "proof_summary.json").write_text(
            '{"obligation_statuses": {}}\n', encoding="utf-8")
        if route == "raw_protected_prefix":
            return 0, {"result_status": "DEPLOYED_TREE_PROVED", "failure_route": None, "failure_code": None}
        return 20, {"result_status": "UNRESOLVED", "failure_route": "UNRESOLVED", "failure_code": "ROUTE_INCONCLUSIVE"}

    monkeypatch.setattr(mod, "prove_seed", fake_prove_seed)
    monkeypatch.setattr(mod, "_route_local_inconclusive", lambda route_id, route_dir: (True, []))
    code, result = mod.prove_seed_v8(
        seed_dir=tmp_path, tree_variant="best_overall", code_root=tmp_path,
        out=tmp_path / "v8", overwrite=False,
    )
    assert code == 0
    assert calls == ["strict_full", "raw_protected_prefix"]
    assert result["selected_terminal_route"] == "raw_protected_prefix"
    assert result["terminal_certificate_kind"] == "PROVED_BY_RAW_PREFIX_ROUTE"
    assert (tmp_path / "v8" / "strict_full").is_dir()
    assert (tmp_path / "v8" / "raw_protected_prefix").is_dir()
    assert not (tmp_path / "v8" / "protected_prefix").exists()


def test_raw_route_registry_has_a_fresh_checker_for_every_route_obligation():
    route = resolve_route("raw_protected_prefix")
    route_ids = set(resolve_registry("raw_protected_prefix").route_entries[i]["id"]
                    for i in range(len(resolve_registry("raw_protected_prefix").route_entries)))
    assert route_ids == set(route.checker_catalog())


def test_v8_instance_binding_theorem_set_matches_declared_raw_derived_manifest():
    import json
    from pathlib import Path
    from formal_toolchain.reference.protected_priority_prefix.raw_mode_order import RAW_DERIVED_THEOREM_IDS

    manifest = json.loads((Path(__file__).resolve().parents[3] / "formal_toolchain" / "theory" / "theory_manifest.json").read_text(encoding="utf-8"))
    assert set(RAW_DERIVED_THEOREM_IDS) == set(manifest["route_derived_obligations"]["raw_protected_prefix"])


def test_v8_route_boundary_alignment_is_consumed_by_both_prefix_routes():
    raw = {e["id"]: e for e in resolve_registry("raw_protected_prefix").entries}
    sat = {e["id"]: e for e in resolve_registry("protected_prefix").entries}
    assert "N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT" in raw["RAW_PREFIX_HI_BAD_PREFIX_REFLECTION"]["depends_on"]
    assert "N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT" in sat["PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION"]["depends_on"]
    assert set(sat["N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT"]["depends_on"]) == {
        "CONTROLLER_BOUNDARY", "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
    }
