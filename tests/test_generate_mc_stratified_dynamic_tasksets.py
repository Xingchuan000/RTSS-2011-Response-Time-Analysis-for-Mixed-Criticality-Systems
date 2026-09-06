"""CLI tests for the standalone MC-Stratified-Dynamic candidate generator."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys


def _run_generator(repo: Path, manifest: Path, rejections: Path, *extra: str) -> None:
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(repo)
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mc_stratified_dynamic_tasksets.py",
            "--candidate-seed-start",
            "10",
            "--num-candidates",
            "4",
            "--output-manifest",
            str(manifest),
            "--output-rejections",
            str(rejections),
            *extra,
        ],
        cwd=str(repo),
        env=env,
        check=True,
    )


def test_standalone_generator_writes_manifest_schema(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest.csv"
    rejections = tmp_path / "rejections.csv"
    _run_generator(repo, manifest, rejections)

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []

    assert len(rows) == 4
    assert all(row["schema_version"] == "mc_stratified_dynamic_manifest_v1" for row in rows)
    required = {
        "candidate_seed",
        "period_family",
        "num_tasks",
        "num_hi",
        "num_lo",
        "total_util_target",
        "total_util_actual",
        "criticality_factor_mean",
        "criticality_factor_min",
        "criticality_factor_max",
        "initial_budget_util_total",
        "initial_budget_util_hi",
        "initial_budget_util_lo",
        "admission_method",
        "admission_priority_policy",
        "c_amc_sem_xf",
        "admission_schedulable",
        "admission_min_slack",
        "amc_rtb_schedulable",
        "amc_rtb_min_slack",
        "attempts",
        "generator_config_hash",
    }
    assert required.issubset(fields)
    assert len({row["generator_config_hash"] for row in rows}) == 1


def test_standalone_generator_is_seed_reconstructible(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest_a = tmp_path / "manifest_a.csv"
    reject_a = tmp_path / "reject_a.csv"
    manifest_b = tmp_path / "manifest_b.csv"
    reject_b = tmp_path / "reject_b.csv"
    _run_generator(repo, manifest_a, reject_a, "--candidate-seed-start", "21")
    _run_generator(repo, manifest_b, reject_b, "--candidate-seed-start", "21")

    assert manifest_a.read_text(encoding="utf-8") == manifest_b.read_text(encoding="utf-8")
    assert reject_a.read_text(encoding="utf-8") == reject_b.read_text(encoding="utf-8")


def test_standalone_generator_can_use_schedulability_gate(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest.csv"
    rejections = tmp_path / "rejections.csv"
    _run_generator(
        repo,
        manifest,
        rejections,
        "--num-tasks",
        "6",
        "--hi-ratio",
        "0.5",
        "--total-util-min",
        "0.10",
        "--total-util-max",
        "0.20",
        "--max-task-util",
        "0.10",
        "--require-schedulable",
        "--max-attempts",
        "20",
    )

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["admission_schedulable"] == "True" for row in rows)
    assert all(row["admission_method"] == "amc_rtb" for row in rows)


def test_standalone_generator_can_use_c_amc_sem_opa_gate(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest_csem.csv"
    rejections = tmp_path / "rejections_csem.csv"
    _run_generator(
        repo,
        manifest,
        rejections,
        "--num-tasks",
        "6",
        "--hi-ratio",
        "0.5",
        "--total-util-min",
        "0.10",
        "--total-util-max",
        "0.20",
        "--max-task-util",
        "0.10",
        "--require-schedulable",
        "--sched-method",
        "c_amc_sem",
        "--priority-policy",
        "opa",
        "--c-amc-sem-xf",
        "0.5",
    )

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["admission_method"] == "c_amc_sem" for row in rows)
    assert all(row["admission_priority_policy"] == "opa" for row in rows)
    assert all(row["admission_schedulable"] == "True" for row in rows)
