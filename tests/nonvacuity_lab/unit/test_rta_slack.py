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


def test_rta_scanner_supports_current_all_task_rta_v3_rows(tmp_path: Path):
    root = tmp_path / "formalv1_csem_t10_s1550_1599" / "r0_s185" / "best_overall_protected_prefix"
    root.mkdir(parents=True)
    (root / "all_task_rta.json").write_text(
        json.dumps(
            {
                "schema_version": "all_task_rta_v3",
                "tasks": [
                    {
                        "task": {"name": "hi_a", "criticality": "HI", "deadline": 20},
                        "r_lo": 12,
                        "r_hi": 18,
                        "case1": [
                            {
                                "response_for_deadline": 18,
                                "trace": [{"il_terms": {"lo_1": 5, "lo_2": 2}}],
                            }
                        ],
                        "case2": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = scan_rta_slack([root.parent.parent.parent])
    assert len(rows) == 1
    assert rows[0]["seed"] == 185
    assert rows[0]["variant"] == "best_overall"
    assert rows[0]["slack"] == 2
    assert rows[0]["limiting_lo_task"] == "lo_1"
