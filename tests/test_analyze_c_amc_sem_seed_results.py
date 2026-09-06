"""Smoke test for the frozen-formal C-AMC-sem seed audit CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analyze_c_amc_sem_seed_results import main


def _write_seed_fixture(root: Path, seed: int) -> None:
    formal_inputs = (
        root
        / f"s{seed}_best_overall_v9_1_e2e"
        / "request"
        / "inputs"
        / "formal_inputs"
    )
    formal_inputs.mkdir(parents=True)
    tasks = [
        {
            "priority_index": 0,
            "name": "tau1",
            "criticality": "LO",
            "period": 10,
            "deadline": 10,
            "code_c_lo": 2,
            "code_c_hi": 2,
        },
        {
            "priority_index": 1,
            "name": "tau2",
            "criticality": "HI",
            "period": 40,
            "deadline": 40,
            "code_c_lo": 10,
            "code_c_hi": 12,
        },
        {
            "priority_index": 2,
            "name": "tau3",
            "criticality": "HI",
            "period": 200,
            "deadline": 200,
            "code_c_lo": 40,
            "code_c_hi": 60,
        },
    ]
    (formal_inputs / "code_taskset_canonical.json").write_text(
        json.dumps(
            {
                "schema_version": "code_taskset_canonical_v1",
                "ordered_tasks": tasks,
                "priority_order": [row["name"] for row in tasks],
            }
        ),
        encoding="utf-8",
    )
    (formal_inputs / "effective_runtime_config.json").write_text(
        json.dumps(
            {
                "schema_version": "effective_runtime_config_v1",
                "fields": {
                    "c_amc_sem_lo_degradation_ratio": {"value": "0.5"},
                    "c_amc_sem_primary_on_switch_time": {"value": True},
                },
            }
        ),
        encoding="utf-8",
    )


def test_seed_audit_writes_csv_json_and_markdown(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    _write_seed_fixture(formal, 123)
    output = tmp_path / "audit"

    assert main(["--formal-root", str(formal), "--seeds", "123", "--output-dir", str(output)]) == 0

    data = json.loads((output / "c_amc_sem_seed_audit.json").read_text(encoding="utf-8"))
    report = data["seeds"][0]
    assert report["seed"] == 123
    assert report["c_amc_sem_frozen_order"]["schedulable"] is True
    assert report["c_amc_sem_opa"]["schedulable"] is True

    with (output / "c_amc_sem_seed_audit.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["seed"] == "123"
    assert "C-AMC-sem baseline" in (output / "c_amc_sem_seed_audit.md").read_text(encoding="utf-8")
