from __future__ import annotations

from pathlib import Path

from nonvacuity_lab.mutators.base import MutationContext
from nonvacuity_lab.mutators.bundle_tamper import BundleTamperMutation


def test_bundle_tamper_changes_only_copied_artifact(tmp_path: Path):
    workspace = tmp_path / "integrity"
    workspace.mkdir()
    target = workspace / "witness.json"
    target.write_text('{"value": 1}\n', encoding="utf-8")
    mutator = BundleTamperMutation()
    result = mutator.apply(
        MutationContext(
            mutation_id="F5",
            source_root=tmp_path,
            mutated_seed=None,
            source_overlay=None,
            parameters={
                "workspace_root": str(workspace),
                "tamper_kind": "json_pointer",
                "target_file": "witness.json",
                "json_pointer": "/value",
                "value": 2,
            },
        )
    )
    assert result.status == "PASS"
    assert result.changed_files == ("witness.json",)
