"""Read-only input snapshots and isolated experiment workspaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Any

from .canonical import tree_hash


@dataclass(frozen=True)
class ExperimentWorkspace:
    root: Path
    base_snapshot: Path
    mutated_seed: Path
    source_overlay: Path
    semantic_output: Path
    integrity_output: Path
    hout_output: Path
    activation_output: Path
    diff_output: Path
    comparison_output: Path
    report_output: Path

    @classmethod
    def create(
        cls,
        *,
        output_root: Path,
        campaign_id: str,
        mutation_id: str,
        seed_dir: Path | None,
        source_root: Path,
    ) -> tuple["ExperimentWorkspace", dict[str, str]]:
        root = (Path(output_root) / campaign_id / mutation_id).resolve()
        _validate_isolated(root, seed_dir=seed_dir, source_root=source_root)
        if root.exists():
            raise FileExistsError(f"实验工作区已存在，拒绝覆盖: {root}")
        paths = cls(
            root=root,
            base_snapshot=root / "base",
            mutated_seed=root / "semantic_recompile" / "mutated_seed",
            source_overlay=root / "semantic_recompile" / "source_overlay",
            semantic_output=root / "semantic_recompile" / "proof",
            integrity_output=root / "integrity_reuse" / "verify",
            hout_output=root / "hout",
            activation_output=root / "activation",
            diff_output=root / "semantic_recompile" / "mutation_diff.json",
            comparison_output=root / "comparison",
            report_output=root / "report",
        )
        paths.root.mkdir(parents=True)
        for path in (
            paths.base_snapshot,
            paths.activation_output,
            paths.comparison_output,
            paths.report_output,
        ):
            path.mkdir(parents=True)
        hashes = {"source_before": tree_hash(source_root)}
        if seed_dir is not None:
            hashes["seed_before"] = tree_hash(seed_dir)
            _copytree_readonly_input(seed_dir, paths.base_snapshot / "seed")
            shutil.copytree(paths.base_snapshot / "seed", paths.mutated_seed, symlinks=False)
        return paths, hashes

    def create_source_overlay(
        self,
        source_root: Path,
        *,
        destination: Path | None = None,
    ) -> Path:
        overlay = (destination or self.source_overlay).resolve()
        if self.root.resolve() not in overlay.parents:
            raise ValueError("source overlay 必须位于实验工作区内")
        if overlay.exists():
            raise FileExistsError(f"source overlay 已存在: {overlay}")
        overlay.mkdir(parents=True)
        for package in ("amc_py", "formal_toolchain"):
            source = Path(source_root) / package
            if not source.is_dir() or source.is_symlink():
                raise ValueError(f"源码包缺失或为软链接: {source}")
            nested_symlink = next((path for path in source.rglob("*") if path.is_symlink()), None)
            if nested_symlink is not None:
                raise ValueError(f"源码 overlay 输入不得包含软链接: {nested_symlink}")
            shutil.copytree(
                source,
                overlay / package,
                symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        return overlay


def verify_original_inputs_unchanged(
    *,
    hashes_before: dict[str, str],
    source_root: Path,
    seed_dir: Path | None,
) -> dict[str, Any]:
    after = {"source_after": tree_hash(source_root)}
    unchanged = after["source_after"] == hashes_before["source_before"]
    if seed_dir is not None:
        after["seed_after"] = tree_hash(seed_dir)
        unchanged = unchanged and after["seed_after"] == hashes_before.get("seed_before")
    return {"status": "PASS" if unchanged else "FAIL", **hashes_before, **after}


def write_input_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copytree_readonly_input(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"seed 输入必须为普通目录: {source}")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"seed 输入不得包含软链接: {path}")
    shutil.copytree(
        source,
        destination,
        symlinks=False,
        copy_function=shutil.copy2,
        ignore=shutil.ignore_patterns(".formal_proof_*", "__pycache__", "*.pyc"),
    )


def _validate_isolated(root: Path, *, seed_dir: Path | None, source_root: Path) -> None:
    if ".." in root.parts:
        raise ValueError("工作区路径不得包含 ..")
    protected = [Path(source_root).resolve() / "formal_toolchain", Path(source_root).resolve() / "amc_py"]
    if seed_dir is not None:
        protected.append(Path(seed_dir).resolve())
    for item in protected:
        if root == item or item in root.parents:
            raise ValueError(f"实验输出不得位于受保护输入内: {item}")
