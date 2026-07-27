from __future__ import annotations

from pathlib import Path

import pytest

from nonvacuity_lab.workspace import (
    ExperimentWorkspace,
    verify_original_inputs_unchanged,
)


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "repo"
    (source / "amc_py").mkdir(parents=True)
    (source / "formal_toolchain").mkdir()
    (source / "amc_py" / "__init__.py").write_text("", encoding="utf-8")
    (source / "formal_toolchain" / "__init__.py").write_text("", encoding="utf-8")
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "input.json").write_text("{}\n", encoding="utf-8")
    (seed / ".formal_proof_protected_prefix").mkdir()
    (seed / ".formal_proof_protected_prefix" / "proof_result.json").write_text(
        "{}\n", encoding="utf-8"
    )
    return source, seed


def test_workspace_copies_inputs_and_never_copies_normal_proof_output(tmp_path: Path):
    source, seed = _repo(tmp_path)
    workspace, before = ExperimentWorkspace.create(
        output_root=tmp_path / "outputs",
        campaign_id="c",
        mutation_id="m",
        seed_dir=seed,
        source_root=source,
    )
    assert (workspace.mutated_seed / "input.json").is_file()
    assert not (
        workspace.mutated_seed / ".formal_proof_protected_prefix"
    ).exists()
    assert verify_original_inputs_unchanged(
        hashes_before=before, source_root=source, seed_dir=seed
    )["status"] == "PASS"


def test_output_inside_seed_is_rejected(tmp_path: Path):
    source, seed = _repo(tmp_path)
    with pytest.raises(ValueError, match="受保护输入"):
        ExperimentWorkspace.create(
            output_root=seed,
            campaign_id="c",
            mutation_id="m",
            seed_dir=seed,
            source_root=source,
        )


def test_seed_symlink_is_rejected(tmp_path: Path):
    source, seed = _repo(tmp_path)
    (seed / "escape").symlink_to(source / "amc_py", target_is_directory=True)
    with pytest.raises(ValueError, match="软链接"):
        ExperimentWorkspace.create(
            output_root=tmp_path / "outputs",
            campaign_id="c",
            mutation_id="m",
            seed_dir=seed,
            source_root=source,
        )
