from __future__ import annotations

import csv
import subprocess


def test_probe_outputs_required_fields(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_seed"])
        w.writeheader()
        w.writerow({"candidate_seed": 0})
        w.writerow({"candidate_seed": 1})

    summary = tmp_path / "summary.csv"
    detail = tmp_path / "detail.csv"

    cmd = [
        "python",
        "scripts/probe_stable_improvement_tasksets.py",
        "--taskset-manifest",
        str(manifest),
        "--seeds",
        "0:2",
        "--end-time",
        "200000",
        "--output-summary",
        str(summary),
        "--output-detail",
        str(detail),
        "--mc-fairgen-num-tasks",
        "6",
    ]
    subprocess.run(cmd, check=True)

    with summary.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for key in [
        "probe_stable_best_found",
        "probe_stable_best_relative_lo_cancellation_reduction",
        "probe_tradeoff_risk",
    ]:
        assert key in rows[0]
