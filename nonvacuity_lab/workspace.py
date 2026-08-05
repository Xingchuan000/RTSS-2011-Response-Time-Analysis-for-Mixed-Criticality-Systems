"""Read-only input snapshots and isolated experiment workspaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Any

from .canonical import file_hash, tree_hash


@dataclass(frozen=True)
class ExperimentWorkspace:
    root: Path
    base_snapshot: Path
    mutated_seed: Path
    source_overlay: Path
    source_snapshot: Path
    semantic_output: Path
    integrity_output: Path
    hout_output: Path
    activation_output: Path
    diff_output: Path
    coherence_output: Path
    command_output: Path
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
        overwrite_existing: bool = False,
    ) -> tuple["ExperimentWorkspace", dict[str, str]]:
        root = (Path(output_root) / campaign_id / mutation_id).resolve()
        _validate_isolated(root, seed_dir=seed_dir, source_root=source_root)
        if root.exists():
            if not overwrite_existing:
                raise FileExistsError(f"实验工作区已存在，拒绝覆盖: {root}")
            shutil.rmtree(root)
        paths = cls(
            root=root,
            base_snapshot=root / "base",
            mutated_seed=root / "semantic_recompile" / "mutated_seed",
            source_overlay=root / "semantic_recompile" / "source_overlay",
            source_snapshot=root / "semantic_recompile" / "source_snapshot.json",
            semantic_output=root / "semantic_recompile" / "proof",
            integrity_output=root / "integrity_reuse" / "verify",
            hout_output=root / "hout",
            activation_output=root / "activation",
            diff_output=root / "semantic_recompile" / "mutation_diff.json",
            coherence_output=root / "semantic_recompile" / "coherence_receipt.json",
            command_output=root / "commands",
            comparison_output=root / "comparison",
            report_output=root / "report",
        )
        paths.root.mkdir(parents=True)
        for path in (
            paths.base_snapshot,
            paths.activation_output,
            paths.comparison_output,
            paths.report_output,
            paths.command_output,
        ):
            path.mkdir(parents=True)
        marker = {
            "schema_version": "nonvacuity_workspace_marker_v1",
            "artifact_class": "NONVACUITY_EXPERIMENT_ONLY",
            "deployment_certificate_eligible": False,
            "campaign_id": campaign_id,
            "mutation_id": mutation_id,
        }
        (root / "DO_NOT_USE_AS_DEPLOYMENT_CERTIFICATE.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
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
        snapshot: dict[str, str] = {}
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
            for path in (overlay / package).rglob("*.py"):
                snapshot[path.relative_to(overlay).as_posix()] = file_hash(path)
        for relative in ("requirements.txt", "pyproject.toml"):
            source = Path(source_root) / relative
            if source.is_file():
                shutil.copy2(source, overlay / relative)
        replay_script = Path(source_root) / "scripts" / "run_nonvacuity_hout.py"
        if replay_script.is_file():
            (overlay / "scripts").mkdir(parents=True, exist_ok=True)
            shutil.copy2(replay_script, overlay / "scripts" / replay_script.name)
        config_source = Path(source_root) / "configs"
        if config_source.is_dir():
            shutil.copytree(
                config_source, overlay / "configs",
                ignore=shutil.ignore_patterns("nonvacuity", "__pycache__", "*.pyc"),
            )
        write_input_snapshot(self.source_snapshot, {
            "schema_version": "nonvacuity_source_snapshot_v1",
            "files": sorted(snapshot),
        })
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


def write_command_receipt(path: Path, *, argv: list[str], cwd: Path,
                          env: dict[str, str], returncode: int) -> None:
    allowed = {key: env.get(key) for key in (
        "PYTHONPATH", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
    ) if key in env}
    write_input_snapshot(path, {
        "schema_version": "nonvacuity_command_receipt_v1",
        "argv": argv, "cwd": str(cwd), "environment": allowed,
        "returncode": returncode,
    })


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
