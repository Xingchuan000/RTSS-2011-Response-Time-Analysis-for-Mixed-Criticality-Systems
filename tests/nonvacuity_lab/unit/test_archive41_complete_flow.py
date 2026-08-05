from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from formal_toolchain.binding.action_binding import bind_action_runtime
from nonvacuity_lab.doctor.runner import _check_ordinary_source_bindings
from nonvacuity_lab.mutators.catalog.selection_mutations import build_selection_catalog
from nonvacuity_lab.runners.paired_hout import _format_command
from nonvacuity_lab.workspace import ExperimentWorkspace


ROOT = Path(__file__).resolve().parents[3]


def _copy_core(tmp_path: Path) -> Path:
    for package in ("amc_py", "formal_toolchain"):
        shutil.copytree(
            ROOT / package,
            tmp_path / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return tmp_path


def test_ordinary_source_bindings_are_a_doctor_gate():
    result = _check_ordinary_source_bindings(ROOT)
    assert result.status.value == "PASS"
    assert result.details["selection_semantics"] == "ranked_first_valid"


def test_all_selection_overlays_remain_source_bindable(tmp_path: Path):
    catalog = build_selection_catalog(ROOT)
    for mutation_id, patches in catalog.items():
        overlay = _copy_core(tmp_path / mutation_id)
        for patch in patches:
            path = overlay / patch["target_file"]
            source = path.read_text(encoding="utf-8")
            occurrence = int(patch.get("occurrence", 1))
            assert source.count(patch["before_snippet"]) == occurrence
            path.write_text(
                source.replace(
                    patch["before_snippet"], patch["after_snippet"], occurrence
                ),
                encoding="utf-8",
            )
        binding = bind_action_runtime(overlay)
        assert binding["status"] == "PASS", (mutation_id, binding)
        expected = {
            "B1": "raw_top1",
            "B2": "top1_valid_else_noop",
            "B3": "all_invalid_force_top1",
        }[mutation_id]
        assert binding["order_evidence"]["selection_semantics"] == expected


def test_cli_module_is_directly_executable():
    completed = subprocess.run(
        [sys.executable, "-m", "nonvacuity_lab.cli", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "run" in completed.stdout


def test_hout_python_token_uses_current_interpreter():
    command = _format_command(
        ["python", "scripts/run_nonvacuity_hout.py", "--output-dir", "{output_dir}"],
        {"output_dir": "/tmp/hout"},
    )
    assert command[0] == sys.executable


def test_workspace_overwrite_is_explicit(tmp_path: Path):
    seed = tmp_path / "seed"
    seed.mkdir()
    source = tmp_path / "source"
    (source / "amc_py").mkdir(parents=True)
    (source / "formal_toolchain").mkdir()
    output = tmp_path / "output"
    first, _ = ExperimentWorkspace.create(
        output_root=output,
        campaign_id="campaign",
        mutation_id="M1",
        seed_dir=seed,
        source_root=source,
    )
    marker = first.root / "stale.txt"
    marker.write_text("stale", encoding="utf-8")
    second, _ = ExperimentWorkspace.create(
        output_root=output,
        campaign_id="campaign",
        mutation_id="M1",
        seed_dir=seed,
        source_root=source,
        overwrite_existing=True,
    )
    assert second.root == first.root
    assert not marker.exists()


def test_research_helper_scripts_are_directly_executable():
    for relative in (
        "scripts/configure_ppp_nonvacuity_campaign.py",
        "scripts/prepare_ppp_nonvacuity_hout.py",
    ):
        completed = subprocess.run(
            [sys.executable, relative, "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, (relative, completed.stderr)
        assert "usage:" in completed.stdout
