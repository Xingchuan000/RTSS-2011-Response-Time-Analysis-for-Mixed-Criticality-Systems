from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from amc_py.dqn.experiment import build_small_nominal_experiment_config, resolve_experiment_bundle
from amc_py.evaluation.c_amc_sem_tree_scenario import run_c_amc_sem_tree_scenario
from amc_py.viper.metrics import _build_leaf_audit_fields
from nonvacuity_lab.activation.default_runtime_binding import build_default_runtime_binding
from nonvacuity_lab.activation.witness_replay import replay_symbolic_witness
from nonvacuity_lab.mutators.action_config import ActionStepMutation
from nonvacuity_lab.mutators.base import MutationContext
from nonvacuity_lab.mutators.catalog.guard_mutations import build_guard_catalog
from nonvacuity_lab.mutators.catalog.model_mutations import build_current_source_model_catalog
from nonvacuity_lab.mutators.catalog.rounding_mutations import build_rounding_catalog
from nonvacuity_lab.mutators.catalog.selection_mutations import build_selection_catalog
from nonvacuity_lab.mutators.coherent_source_patch import CoherentSourcePatchMutation
from nonvacuity_lab.mutators.factory import build_mutator


SOURCE_ROOT = Path.cwd().resolve()


def _copy_patch_targets(tmp_path: Path, patches: tuple[dict, ...] | list[dict]) -> Path:
    overlay = tmp_path / "overlay"
    for patch in patches:
        relative = Path(str(patch["target_file"]))
        target = overlay / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ROOT / relative, target)
    return overlay


def _apply_catalog(tmp_path: Path, semantic_change_id: str, patches) -> None:
    patches = tuple(dict(item) for item in patches)
    overlay = _copy_patch_targets(tmp_path, patches)
    context = MutationContext(
        mutation_id=semantic_change_id,
        source_root=SOURCE_ROOT,
        mutated_seed=None,
        source_overlay=overlay,
        parameters={"semantic_change_id": semantic_change_id, "patches": list(patches)},
    )
    mutator = CoherentSourcePatchMutation()
    assert mutator.preflight(context).status == "PASS"
    result = mutator.apply(context)
    assert mutator.verify_single_change(result).status == "PASS"
    assert result.before_hash != result.after_hash


def test_c1_factory_dispatches_to_action_step_mutator():
    assert isinstance(build_mutator("ACTION_SEMANTICS"), ActionStepMutation)
    assert isinstance(build_mutator("ACTION_RATIO_2_TO_5"), ActionStepMutation)


@pytest.mark.parametrize(
    ("catalog_key", "semantic_change_id"),
    [
        ("B1", "raw_top1_selection"),
        ("B2", "top1_valid_else_noop"),
        ("B3", "all_invalid_force_top1"),
    ],
)
def test_selection_catalog_applies_to_current_source(tmp_path: Path, catalog_key: str, semantic_change_id: str):
    _apply_catalog(tmp_path, semantic_change_id, build_selection_catalog(SOURCE_ROOT)[catalog_key])


@pytest.mark.parametrize("guard_id", ["decrease_hi_forbidden", "budget_floor_violation", "deploy_cap", "safety_checker"])
def test_guard_catalog_applies_to_current_source(tmp_path: Path, guard_id: str):
    _apply_catalog(tmp_path, "guard_removal", build_guard_catalog(SOURCE_ROOT)[guard_id])


def test_rounding_catalog_applies_to_current_source(tmp_path: Path):
    _apply_catalog(tmp_path, "rounding_to_nearest", build_rounding_catalog(SOURCE_ROOT))


@pytest.mark.parametrize("mutation_id", ["E1", "E2", "E3", "E4", "E5", "E6"])
def test_model_catalog_applies_to_current_source(tmp_path: Path, mutation_id: str):
    entry = build_current_source_model_catalog(SOURCE_ROOT)[mutation_id]
    _apply_catalog(tmp_path, entry.semantic_change_id, entry.patches)


def _write_activation_artifacts(tmp_path: Path):
    tree = {
        "schema_version": "integer_tree_v1",
        "root_node_id": 0,
        "nodes": [{"node_id": 0, "feature_index": 0, "threshold_int": 10, "left_child": 1, "right_child": 2}],
        "leaves": [
            {"node_id": 1, "action_ranking": [0, 1]},
            {"node_id": 2, "action_ranking": [1, 0]},
        ],
    }
    actions = {"actions": [
        {"action_id": 0, "increase_task": "HI", "increase_ratio": 0.02, "decrease_tasks": [], "is_noop": False},
        {"action_id": 1, "increase_task": None, "decrease_tasks": [], "is_noop": True},
    ]}
    taskset = {"ordered_tasks": [
        {"name": "HI", "period": 10, "deadline": 10, "criticality": "HI", "code_c_lo": 10, "code_c_hi": 10,
         "initial_runtime_budget": 10, "budget_floor": 10, "action_hard_upper": 10},
    ]}
    paths = {}
    for name, payload in (("tree", tree), ("actions", actions), ("taskset", taskset)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    return paths


def test_default_runtime_binding_replays_mask_and_bypass(tmp_path: Path):
    paths = _write_activation_artifacts(tmp_path)
    target = {"tree_path": str(paths["tree"]), "leaf_id": 1, "action_id": 0}
    binding = {
        "action_definitions_path": str(paths["actions"]),
        "taskset_path": str(paths["taskset"]),
        "overlay_semantics": "raw_top1",
        "overlay_unchecked_apply": True,
    }
    runtime = build_default_runtime_binding(
        clean_source_root=SOURCE_ROOT, overlay_source_root=SOURCE_ROOT,
        resolved_target=target, binding=binding,
    )
    replay = replay_symbolic_witness(
        {"budgets": {"HI": 10}, "features": {"0": 0}},
        binding=runtime.state_binding,
        clean_runtime=runtime.clean_runtime,
        overlay_runtime=runtime.overlay_runtime,
        expected={"formula_kind": "B_RAW_TOP1_BREAKS_INVARIANT", "leaf_id": 1, "action_id": 0},
    )
    assert replay["status"] == "MATCHED"
    assert replay["mutated_observation"]["budgets_after"]["HI"] == 11


def test_leaf_audit_records_raw_top1_specific_reject_reason():
    policy = SimpleNamespace(
        action_definition=lambda action_id: {"action_id": action_id},
    )
    fields = _build_leaf_audit_fields(
        step_index=0,
        state_vector=(0.0,),
        feature_names=("x",),
        tree_policy=policy,
        tree_info={
            "tree_leaf_id": 1,
            "tree_raw_top1_action_id": 2,
            "tree_raw_top1_invalid": True,
            "tree_action_ranking": [2, 1],
            "tree_action_proba": [0.0, 0.2, 0.8],
        },
        selected_action_id=1,
        valid_action_mask=(False, True, False),
        mask_details=(
            {"action_id": 0, "valid": False, "reject_reason": "other"},
            {"action_id": 1, "valid": True, "reject_reason": None},
            {"action_id": 2, "valid": False, "reject_reason": "deploy_cap"},
        ),
        teacher_diag=None,
        leaf_audit_state_mode="none",
        leaf_audit_top_k_actions=2,
    )
    assert fields["rejected_action_id"] == 2
    assert fields["tree_raw_top1_reject_reason"] == "deploy_cap"


def test_hout_scenario_uses_exact_taskset_and_exports_reject_reason(tmp_path: Path, monkeypatch):
    config = build_small_nominal_experiment_config()
    tasks = resolve_experiment_bundle(config, 7).ordered_tasks
    taskset = tmp_path / "taskset.json"
    taskset.write_text(json.dumps({"ordered_tasks": [
        {"name": t.name, "period": t.period, "deadline": t.deadline, "code_c_lo": t.c_lo, "code_c_hi": t.c_hi,
         "criticality": t.criticality.value}
        for t in tasks
    ]}), encoding="utf-8")
    seed_dir = tmp_path / "s7"
    variant = seed_dir / "best_overall"
    variant.mkdir(parents=True)
    tree = variant / "integer_tree.json"
    tree.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("amc_py.evaluation.c_amc_sem_tree_scenario._factory_from_config", lambda raw: (lambda: config))
    monkeypatch.setattr("amc_py.evaluation.c_amc_sem_tree_scenario.load_tree_policy_artifact", lambda *a, **k: object())
    monkeypatch.setattr(
        "amc_py.evaluation.c_amc_sem_tree_scenario.evaluate_tree_policy_once",
        lambda **kwargs: (
            {"hi_deadline_misses": 0, "lo_deadline_misses": 0, "lo_cancellations": 0},
            SimpleNamespace(),
            [{
                "tree_leaf_id": 1, "tree_raw_top1_action_id": 2, "tree_raw_top1_invalid": True,
                "tree_raw_top1_reject_reason": "budget_floor_violation", "tree_selected_action_id": 1,
                "tree_selected_rank": 1, "tree_no_valid_action": False,
            }],
        ),
    )
    summary, events = run_c_amc_sem_tree_scenario(
        seed_dir=seed_dir, tree_path=tree, scenario_seed=7,
        runtime_config={
            "experiment_name": "small_nominal", "horizon": 100, "taskset_seed": 7,
            "scenario_seed_drives_bundle": True, "taskset_path": str(taskset),
        },
    )
    assert summary["runtime_taskset_fingerprint"]
    assert events[0]["rejected_action_id"] == 2
    assert events[0]["reject_reason"] == "budget_floor_violation"


def test_hout_scenario_rejects_mismatched_taskset(tmp_path: Path, monkeypatch):
    config = build_small_nominal_experiment_config()
    taskset = tmp_path / "taskset.json"
    taskset.write_text(json.dumps({"ordered_tasks": [
        {"name": "WRONG", "period": 10, "deadline": 10, "code_c_lo": 1, "code_c_hi": 1, "criticality": "HI"}
    ]}), encoding="utf-8")
    seed_dir = tmp_path / "s7"
    variant = seed_dir / "best_overall"
    variant.mkdir(parents=True)
    tree = variant / "integer_tree.json"
    tree.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("amc_py.evaluation.c_amc_sem_tree_scenario._factory_from_config", lambda raw: (lambda: config))
    monkeypatch.setattr("amc_py.evaluation.c_amc_sem_tree_scenario.load_tree_policy_artifact", lambda *a, **k: object())
    with pytest.raises(ValueError, match="HOUT_TASKSET_MISMATCH"):
        run_c_amc_sem_tree_scenario(
            seed_dir=seed_dir, tree_path=tree, scenario_seed=7,
            runtime_config={"experiment_name": "small_nominal", "horizon": 100, "taskset_path": str(taskset)},
        )


def test_b2_b3_target_binding_uses_original_raw_top1():
    from nonvacuity_lab.config_resolver import _bind_observed_raw_target

    mutation = {"resolved_target": {}}
    rows = [{
        "leaf_id": 5, "action_ranking": [1, 9], "raw_top1_invalid_count": 7,
        "all_invalid_count": 2, "hout_hit_count": 10, "training_samples": 3,
        "action_risks": [{"action_id": 9, "risk_class": "HI_BUDGET_DECREASE", "observed_reject_count": 7}],
    }]
    _bind_observed_raw_target(mutation, rows, evidence_field="raw_top1_invalid_count")
    assert mutation["resolved_target"]["action_id"] == 1
    assert mutation["resolved_target"]["original_ranking"] == [1, 9]


def test_c2_resolver_selects_activated_raw_top1(tmp_path: Path):
    from nonvacuity_lab.config_resolver import _bind_c2_rounding_target

    seed = tmp_path / "s185"
    variant = seed / "best_overall"
    formal = seed / "formal_inputs"
    variant.mkdir(parents=True)
    formal.mkdir()
    tree = variant / "integer_tree.json"
    tree.write_text(json.dumps({
        "schema_version": "integer_tree_v1", "root_node_id": 1,
        "nodes": [], "leaves": [{"node_id": 1, "action_ranking": [0]}],
        "feature_names": ["x"],
    }), encoding="utf-8")
    (variant / "action_definitions.json").write_text(json.dumps({"actions": [{
        "action_id": 0, "increase_task": "LO", "increase_ratio": 0.02,
        "decrease_tasks": [], "minimum_increment": 1, "is_noop": False,
    }]}), encoding="utf-8")
    (variant / "feature_names.json").write_text(json.dumps(["x"]), encoding="utf-8")
    (formal / "code_taskset_canonical.json").write_text(json.dumps({"ordered_tasks": [{
        "name": "LO", "criticality": "LO", "period": 100, "deadline": 100,
        "code_c_lo": 60, "code_c_hi": 100, "initial_runtime_budget": 60,
        "budget_floor": 60, "action_hard_upper": 100,
    }]}), encoding="utf-8")
    mutation = {"resolved_target": {"seed_dir": str(seed), "tree_path": str(tree)}}
    _bind_c2_rounding_target(mutation, [{
        "leaf_id": 1, "action_ranking": [0], "hout_hit_count": 9, "training_samples": 1,
    }])
    target = mutation["resolved_target"]
    assert target["action_id"] == 0
    assert target["rounding_witness_ceil_floor"] != target["rounding_witness_nearest"]


def test_v2_hout_activation_is_bound_to_resolved_target(tmp_path: Path):
    from nonvacuity_lab.v2_runner import _v2_mutation_to_v1

    base = {
        "mutation_id": "B1_demo", "mutation_class": "MASK_BYPASS", "enabled": False,
        "seed": 185, "tree_variant": "best_overall", "pair_with": "A1_demo",
        "resolved_target": {"seed_dir": str(tmp_path), "tree_path": str(tmp_path / "tree.json"), "leaf_id": 4, "action_id": 7},
        "mutator": {"kind": "coherent_source_patch", "parameters": {}},
        "activation": {"mode": "hout"},
        "expected": {"allowed_result_statuses": ["POLICY_CONTRACT_VIOLATION"]},
    }
    translated = _v2_mutation_to_v1(base, config={"hout_profiles": {}}, base_dir=tmp_path)
    activation = translated["activation"]
    assert activation["required_leaf_id"] == 4
    assert activation["required_action_id"] == 7
    assert activation["require_baseline_reject"] is True
    assert activation["require_selected_after_mutation"] is True


def test_hout_activation_requires_mutated_reject_and_budget_difference():
    from nonvacuity_lab.activation.hout_activation import evaluate_hout_activation

    common = {"scenario_seed": 1, "controller_decision_index": 0, "leaf_id": 2, "raw_top1_action_id": 3,
              "all_invalid": False, "implicit_noop": False}
    result = evaluate_hout_activation(
        mutation_id="A1",
        base_events=[{**common, "raw_top1_invalid": False, "budget_after": {"T": 10}}],
        mutated_events=[{**common, "raw_top1_invalid": True, "budget_after": {"T": 11}}],
        rule={"required_leaf_id": 2, "required_action_id": 3, "require_mutated_reject": True,
              "require_any_budget_difference": True},
    )
    assert result.status.value == "ACTIVATED"
    assert result.details["mutated_reject_count"] == 1
    assert result.details["budget_difference_count"] == 1


def test_hout_activation_detects_retroactive_release_budget_difference():
    from nonvacuity_lab.activation.hout_activation import evaluate_hout_activation

    base = [{"scenario_seed": 1, "controller_decision_index": 0, "all_invalid": False, "implicit_noop": False,
             "active_release_budgets_after_update": {"T#0": 10}}]
    mutated = [{"scenario_seed": 1, "controller_decision_index": 0, "all_invalid": False, "implicit_noop": False,
                "active_release_budgets_after_update": {"T#0": 11}}]
    result = evaluate_hout_activation(
        mutation_id="C3", base_events=base, mutated_events=mutated,
        rule={"require_active_release_budget_difference": True},
    )
    assert result.status.value == "ACTIVATED"
    assert result.details["active_release_budget_difference_count"] == 1


def test_b4_resolver_binds_observed_raw_top1_only():
    from nonvacuity_lab.resolver.guard_ablation import resolve_guard_ablation

    catalog = {
        "guards": {
            "deploy_cap": {
                "guard_id": "DEPLOY_CAP_GUARD",
                "reject_reason_prefixes": ["deploy_cap"],
                "symbolic_constraint": {"constant": True},
            }
        }
    }
    rows = [
        {
            "leaf_id": 3,
            "hout_hit_count": 20,
            "action_ranking": [1, 9],
            "reject_reason_histogram": {"deploy_cap": 12},
            # Higher aggregate count for a lower-ranked action must not be used.
            "rejected_action_histogram": {"1": 4, "9": 11},
        },
        {
            "leaf_id": 4,
            "hout_hit_count": 8,
            "action_ranking": [7, 6],
            "reject_reason_histogram": {"deploy_cap": 5},
            "rejected_action_histogram": {"7": 5},
        },
    ]
    resolved = resolve_guard_ablation(rows, catalog)
    assert resolved["leaf_id"] == 4
    assert resolved["action_id"] == 7
    assert resolved["raw_top1_action_id"] == 7
    assert resolved["hit_count"] == 5


def test_leaf_coverage_counts_rank_one_as_fallback(tmp_path: Path):
    from nonvacuity_lab.audit.leaf_coverage import aggregate_hout_events

    events_path = tmp_path / "s185" / "best_overall" / "events.json"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "seed": 185,
                        "tree_variant": "best_overall",
                        "scenario_id": 1,
                        "leaf_id": 2,
                        "selected_rank": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = aggregate_hout_events([events_path])
    assert rows[(185, "best_overall", 2)]["fallback_count"] == 1


def test_leaf_coverage_reads_jsonl_hout_output(tmp_path: Path):
    from nonvacuity_lab.audit.leaf_coverage import aggregate_hout_events

    events_path = tmp_path / "s1264" / "best_balanced" / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        json.dumps({
            "seed": 1264, "tree_variant": "best_balanced", "scenario_id": 10,
            "leaf_id": 8, "selected_rank": 2, "raw_top1_invalid": True,
            "raw_top1_action_id": 4, "rejected_action_id": 4,
            "reject_reason": "deploy_cap",
        }) + "\n",
        encoding="utf-8",
    )
    rows = aggregate_hout_events([events_path])
    row = rows[(1264, "best_balanced", 8)]
    assert row["hout_hit_count"] == 1
    assert row["fallback_count"] == 1
    assert row["rejected_action_histogram"]["4"] == 1


def test_d1_coordinate_requires_authoritative_binding(tmp_path: Path):
    import pytest
    from nonvacuity_lab.runners.campaign import _resolve_d1_envelope_coordinate

    seed_dir = tmp_path / "s77"
    formal = seed_dir / "formal_inputs"
    formal.mkdir(parents=True)
    target = formal / "certified_reference_envelope.json"
    target.write_text(
        json.dumps({"tasks": {"LO_B": {"upper": 30}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="D1_AUTHORITATIVE_ENVELOPE_BINDING_MISSING"):
        _resolve_d1_envelope_coordinate(
            selected={"limiting_lo_task": "LO_B"},
            parameters={},
            seed_dir=seed_dir,
        )

    target_file, pointer = _resolve_d1_envelope_coordinate(
        selected={
            "limiting_lo_task": "LO_B",
            "envelope_target_file": "formal_inputs/certified_reference_envelope.json",
            "envelope_json_pointer": "/tasks/LO_B/upper",
        },
        parameters={},
        seed_dir=seed_dir,
    )
    assert target_file == "formal_inputs/certified_reference_envelope.json"
    assert pointer == "/tasks/LO_B/upper"


def test_research_campaign_configurator_enables_and_disables_rows(tmp_path: Path):
    from nonvacuity_lab.config_io import write_resolved_campaign
    from scripts.configure_ppp_nonvacuity_campaign import configure_campaign

    path = tmp_path / "resolved.json"
    write_resolved_campaign(path, {
        "schema_version": "nonvacuity_campaign_v2",
        "config_kind": "RESOLVED",
        "campaign_id": "demo",
        "enabled": False,
        "resolver_receipt": {"path": "receipt.json", "sha256": "demo"},
        "mutations": [
            {"mutation_id": "P0_demo", "enabled": False},
            {"mutation_id": "A1_demo", "enabled": False},
            {"mutation_id": "B1_demo", "enabled": False},
        ],
    })
    enabled = configure_campaign(path, groups=("A", "B"))
    assert enabled["campaign_enabled"] is True
    assert enabled["enabled_mutations"] == ["A1_demo", "B1_demo"]
    disabled = configure_campaign(path, disable=True)
    assert disabled["campaign_enabled"] is False
    assert disabled["enabled_mutations"] == []


def test_b2_activation_requires_a_lower_ranked_legal_action(tmp_path: Path):
    paths = _write_activation_artifacts(tmp_path)
    target = {
        "tree_path": str(paths["tree"]),
        "leaf_id": 1,
        "action_id": 0,
        "original_ranking": [0, 1],
    }
    binding = {
        "action_definitions_path": str(paths["actions"]),
        "taskset_path": str(paths["taskset"]),
        "overlay_semantics": "top1_valid_else_noop",
    }
    runtime = build_default_runtime_binding(
        clean_source_root=SOURCE_ROOT,
        overlay_source_root=SOURCE_ROOT,
        resolved_target=target,
        binding=binding,
    )

    replay = replay_symbolic_witness(
        {"budgets": {"HI": 10}, "features": {"0": 0}},
        binding=runtime.state_binding,
        clean_runtime=runtime.clean_runtime,
        overlay_runtime=runtime.overlay_runtime,
        expected={
            "formula_kind": "B2_NO_FIRST_VALID_DIFFERENCE",
            "leaf_id": 1,
            "action_id": 0,
        },
    )

    assert replay["status"] == "MATCHED"
    assert replay["base_observation"]["selected_action_id"] == 1
    assert replay["mutated_observation"]["selected_action_id"] is None


def test_b2_activation_rejects_an_all_invalid_state(tmp_path: Path):
    paths = _write_activation_artifacts(tmp_path)
    actions = json.loads(paths["actions"].read_text(encoding="utf-8"))
    actions["actions"][1] = {
        "action_id": 1,
        "increase_task": "HI",
        "increase_ratio": 0.02,
        "decrease_tasks": [],
        "is_noop": False,
    }
    paths["actions"].write_text(json.dumps(actions), encoding="utf-8")
    target = {
        "tree_path": str(paths["tree"]),
        "leaf_id": 1,
        "action_id": 0,
        "original_ranking": [0, 1],
    }
    binding = {
        "action_definitions_path": str(paths["actions"]),
        "taskset_path": str(paths["taskset"]),
        "overlay_semantics": "top1_valid_else_noop",
    }
    runtime = build_default_runtime_binding(
        clean_source_root=SOURCE_ROOT,
        overlay_source_root=SOURCE_ROOT,
        resolved_target=target,
        binding=binding,
    )

    replay = replay_symbolic_witness(
        {"budgets": {"HI": 10}, "features": {"0": 0}},
        binding=runtime.state_binding,
        clean_runtime=runtime.clean_runtime,
        overlay_runtime=runtime.overlay_runtime,
        expected={
            "formula_kind": "B2_NO_FIRST_VALID_DIFFERENCE",
            "leaf_id": 1,
            "action_id": 0,
        },
    )

    assert replay["status"] == "ACTIVATION_MODEL_RUNTIME_MISMATCH"
    assert replay["checks"]["clean_selects_lower_valid"] is False


def test_b2_resolver_uses_difference_formula():
    from nonvacuity_lab.config_resolver import _bind_symbolic_activation

    mutation = {
        "resolved_target": {
            "tree_path": "unused.json",
            "leaf_id": 1,
            "action_id": 0,
        },
        "activation": {"mode": "symbolic_auto"},
    }
    # Artifact discovery is independently tested; isolate the formula mapping.
    from unittest.mock import patch

    with patch(
        "nonvacuity_lab.config_resolver._discover_symbolic_binding",
        return_value={},
    ):
        _bind_symbolic_activation(mutation, "B2")

    assert mutation["activation"]["formula_kind"] == "B2_NO_FIRST_VALID_DIFFERENCE"
