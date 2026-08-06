from __future__ import annotations

import ast
from pathlib import Path

from nonvacuity_lab.mutators.base import MutationContext
from nonvacuity_lab.mutators.coherent_source_patch import CoherentSourcePatchMutation
from nonvacuity_lab.mutators.python_binding import bind_symbol


def _write_overlay(tmp_path: Path) -> Path:
    overlay = tmp_path / "overlay"
    deployed = overlay / "amc_py" / "runtime.py"
    mirror = overlay / "formal_toolchain" / "semantics" / "runtime.py"
    deployed.parent.mkdir(parents=True)
    mirror.parent.mkdir(parents=True)
    source = (
        "def transition(value: int) -> int:\n"
        "    if value > 0:\n"
        "        return value\n"
        "    return 0\n"
    )
    deployed.write_text(source, encoding="utf-8")
    mirror.write_text(source, encoding="utf-8")
    return overlay


def test_coherent_source_patch_binds_mirrored_current_symbols(tmp_path: Path):
    overlay = _write_overlay(tmp_path)
    patches = []
    for role, relative in (
        ("DEPLOYED_IMPLEMENTATION", "amc_py/runtime.py"),
        ("FORMAL_SEMANTIC_MIRROR", "formal_toolchain/semantics/runtime.py"),
    ):
        source = (overlay / relative).read_text(encoding="utf-8")
        bound = bind_symbol(source, "transition")
        patches.append(
            {
                "role": role,
                "target_file": relative,
                "target_symbol": "transition",
                "before_ast_hash": bound.ast_hash,
                "before_snippet": "if value > 0:",
                "after_snippet": "if value >= 0:",
                "occurrence": 1,
            }
        )
    mutator = CoherentSourcePatchMutation()
    context = MutationContext(
        mutation_id="E",
        source_root=tmp_path,
        mutated_seed=None,
        source_overlay=overlay,
        parameters={"semantic_change_id": "BOUNDARY_CHANGE", "patches": patches},
    )
    assert mutator.preflight(context).status == "PASS"
    result = mutator.apply(context)
    assert mutator.verify_single_change(result).status == "PASS"
    assert set(result.details["coherent_roles"]) == {
        "DEPLOYED_IMPLEMENTATION",
        "FORMAL_SEMANTIC_MIRROR",
    }
    for relative in ("amc_py/runtime.py", "formal_toolchain/semantics/runtime.py"):
        ast.parse((overlay / relative).read_text(encoding="utf-8"))
        assert "if value >= 0:" in (overlay / relative).read_text(encoding="utf-8")


def test_coherent_source_patch_rejects_verifier_target(tmp_path: Path):
    overlay = _write_overlay(tmp_path)
    verifier = overlay / "formal_toolchain" / "verifier" / "checker.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("def check():\n    return True\n", encoding="utf-8")
    bound = bind_symbol(verifier.read_text(encoding="utf-8"), "check")
    context = MutationContext(
        mutation_id="BAD",
        source_root=tmp_path,
        mutated_seed=None,
        source_overlay=overlay,
        parameters={
            "semantic_change_id": "BAD",
            "patches": [
                {
                    "role": "DEPLOYED_IMPLEMENTATION",
                    "target_file": "formal_toolchain/verifier/checker.py",
                    "target_symbol": "check",
                    "before_ast_hash": bound.ast_hash,
                    "before_snippet": "return True",
                    "after_snippet": "return False",
                }
            ],
        },
    )
    result = CoherentSourcePatchMutation().preflight(context)
    assert result.status == "FAIL"
    assert "FORBIDDEN_PATCH_TARGET" in str(result.details.get("reason"))


def test_b2_patch_does_not_mislabel_top1_invalid_noop_as_all_invalid():
    from nonvacuity_lab.mutators.catalog.selection_mutations import _policy_patch

    patch = _policy_patch(Path('.'), semantics="top1_valid_else_noop")
    assert '"tree_top1_invalid_noop": True' in patch.after_snippet
    assert '"tree_no_valid_action": False' in patch.after_snippet
