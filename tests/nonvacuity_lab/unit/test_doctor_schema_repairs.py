from __future__ import annotations

import json
from pathlib import Path

from nonvacuity_lab.doctor.check_registry import load_obligation_ids
from nonvacuity_lab.doctor.checks import check_patch_binding
from nonvacuity_lab.doctor.runner import _coherent_text_patches
from nonvacuity_lab.mutators.python_binding import bind_symbol
from nonvacuity_lab.preflight import audit_mutation
from nonvacuity_lab.schema import MutationManifest


def test_doctor_patch_binding_counts_inside_declared_symbol(tmp_path: Path):
    target = tmp_path / "amc_py" / "rl" / "env.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "def selected(value):\n"
        "    if value:\n"
        "        return 1\n"
        "    return 0\n\n"
        "def unrelated(value):\n"
        "    if value:\n"
        "        return 2\n"
        "    return 0\n",
        encoding="utf-8",
    )
    bound = bind_symbol(target.read_text(encoding="utf-8"), "selected")
    result = check_patch_binding(
        tmp_path,
        {
            "target_file": "amc_py/rl/env.py",
            "target_symbol": "selected",
            "before_ast_hash": bound.ast_hash,
            "before_snippet": "    if value:\n",
            "after_snippet": "    if bool(value):\n",
            "occurrence": 1,
        },
    )
    assert result.status.value == "PASS"


def test_doctor_only_treats_coherent_source_patches_as_text_patches():
    c3 = {
        "mutator": {
            "kind": "retroactive_release_budget",
            "parameters": {"patches": [{"target_file": "amc_py/event_runtime.py"}]},
        }
    }
    assert _coherent_text_patches(c3) == ()

    coherent = {
        "mutator": {
            "kind": "coherent_source_patch",
            "parameters": {"patches": [{"before_snippet": "x", "after_snippet": "y"}]},
        }
    }
    assert len(_coherent_text_patches(coherent)) == 1


def test_obligation_loader_accepts_current_entries_schema(tmp_path: Path):
    registry = tmp_path / "formal_toolchain" / "specs" / "obligation_registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"schema_version": "v", "entries": [{"id": "A"}, {"id": "B"}]}),
        encoding="utf-8",
    )
    routes = tmp_path / "formal_toolchain" / "specs" / "routes"
    routes.mkdir()
    (routes / "protected_prefix_registry.json").write_text(
        json.dumps({"obligation_ids": ["PROTECTED_PREFIX_RTA"]}),
        encoding="utf-8",
    )
    assert load_obligation_ids(tmp_path) == {"A", "B", "PROTECTED_PREFIX_RTA"}


def test_source_binding_tamper_preflight_checks_clean_source_root(tmp_path: Path):
    source = tmp_path / "source"
    target = source / "amc_py" / "rl" / "actions.py"
    target.parent.mkdir(parents=True)
    target.write_text('    if mode == "ceil_floor":\n', encoding="utf-8")
    reuse = tmp_path / "proof_run"
    reuse.mkdir()
    seed_dir = tmp_path / "s185"
    seed_dir.mkdir()
    manifest = MutationManifest.from_mapping(
        {
            "schema_version": "nonvacuity_mutation_v1",
            "enabled": True,
            "mutation_id": "F7_source_binding_tamper",
            "mutation_class": "SOURCE_BINDING_TAMPER",
            "seed_dir": str(seed_dir),
            "reuse_source_bundle": str(reuse),
            "tree_variant": "best_overall",
            "mutator": {
                "kind": "bundle_tamper",
                "parameters": {
                    "tamper_kind": "source_file",
                    "target_file": "amc_py/rl/actions.py",
                    "before_snippet": '    if mode == "ceil_floor":\n',
                    "after_snippet": '    if mode in {"ceil_floor"}:\n',
                },
            },
            "activation": {"mode": "none"},
        },
        base_dir=tmp_path,
    )
    result = audit_mutation(manifest, source_root=source)
    assert result["status"] == "PASS"
