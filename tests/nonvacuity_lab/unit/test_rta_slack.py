from __future__ import annotations

import json
from pathlib import Path

from nonvacuity_lab.analysis.rta_slack import scan_rta_slack, select_minimum_slack


def test_rta_scanner_dynamically_selects_minimum_slack(tmp_path: Path):
    first = tmp_path / "s185" / "best_overall"
    second = tmp_path / "s1264" / "best_balanced"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "rta.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "hi_a",
                        "criticality": "HI",
                        "deadline": 20,
                        "R_LO": 10,
                        "R_HI": 17,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (second / "rta.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "hi_b",
                        "criticality": "HI",
                        "deadline": 20,
                        "R_LO": 19,
                        "R_HI": 18,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = scan_rta_slack([first, second])
    selected = select_minimum_slack(rows)
    assert selected["seed"] == 1264
    assert selected["task_id"] == "hi_b"
    assert selected["slack"] == 1
