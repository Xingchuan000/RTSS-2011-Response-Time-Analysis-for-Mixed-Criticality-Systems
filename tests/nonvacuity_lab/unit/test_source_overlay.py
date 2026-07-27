from __future__ import annotations

from pathlib import Path

from nonvacuity_lab.mutators.base import MutationContext
from nonvacuity_lab.mutators.runtime_source import (
    MultiPythonSymbolMutation,
    PythonSymbolMutation,
)


def test_python_patch_is_limited_to_declared_symbol(tmp_path: Path):
    overlay = tmp_path / "overlay"
    target = overlay / "amc_py" / "runtime.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "class Runtime:\n"
        "    def cleanup(self, done):\n"
        "        if done:\n"
        "            return 'keep'\n"
        "        return 'wait'\n\n"
        "def unrelated():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    mutator = PythonSymbolMutation()
    result = mutator.apply(
        MutationContext(
            mutation_id="E1",
            source_root=tmp_path,
            mutated_seed=None,
            source_overlay=overlay,
            parameters={
                "target_file": "amc_py/runtime.py",
                "target_symbol": "Runtime.cleanup",
                "before_snippet": "return 'keep'",
                "after_snippet": "return 'remove'",
            },
        )
    )
    assert mutator.verify_single_change(result).status == "PASS"
    assert result.changed_symbols == ("Runtime.cleanup",)
    assert "return 'remove'" in target.read_text(encoding="utf-8")


def test_mirrored_source_patches_count_as_one_conceptual_change(tmp_path: Path):
    overlay = tmp_path / "overlay"
    for package in ("amc_py", "formal_toolchain"):
        target = overlay / package / "runtime.py"
        target.parent.mkdir(parents=True)
        target.write_text(
            "def transition(enabled=False):\n"
            "    if enabled:\n"
            "        return 'mutated'\n"
            "    return 'base'\n",
            encoding="utf-8",
        )
    mutator = MultiPythonSymbolMutation()
    result = mutator.apply(
        MutationContext(
            mutation_id="E",
            source_root=tmp_path,
            mutated_seed=None,
            source_overlay=overlay,
            parameters={
                "semantic_group": "mirrored_transition",
                "patches": [
                    {
                        "target_file": f"{package}/runtime.py",
                        "target_symbol": "transition",
                        "before_snippet": "if enabled:",
                        "after_snippet": "if True:",
                    }
                    for package in ("amc_py", "formal_toolchain")
                ],
            },
        )
    )
    assert mutator.verify_single_change(result).status == "PASS"
    assert result.semantic_change_count == 1
    assert len(result.changed_files) == 2
