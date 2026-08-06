#!/usr/bin/env python3
"""Summarize PPP nonvacuity campaign results without modifying experiment data."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MODES = [
    "S185Core",
    "PositiveControls",
    "S397Pair",
    "B234",
    "C123",
    "IntegrityAll",
    "ModelEAll",
    "D1",
]

INVALID_STATUSES = {
    "BASELINE_REGRESSION",
    "SETUP_INVALID",
    "MUTATION_NOT_ACTIVATED",
    "UNEXPECTED_PASS",
    "UNEXPECTED_FAIL",
    "INTEGRITY_REJECTION_MISSING",
    "PROOF_BUNDLE_INVALID",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_mode(path: Path, data: dict[str, Any], modes: list[str]) -> str | None:
    text = " ".join(
        [
            str(path).lower(),
            str(data.get("mode", "")).lower(),
            str(data.get("campaign_id", "")).lower(),
        ]
    )
    aliases = {
        "S185Core": ["s185core", "s185_core"],
        "PositiveControls": ["positivecontrols", "positive_controls"],
        "S397Pair": ["s397pair", "s397_pair"],
        "B234": ["b234"],
        "C123": ["c123"],
        "IntegrityAll": ["integrityall", "integrity_all"],
        "ModelEAll": ["modeleall", "model_e_all", "modele_all"],
        "D1": ["_d1", "v2_d1", "mode=d1", "\\d1\\", "/d1/"],
    }
    for mode in modes:
        if any(token in text for token in aliases.get(mode, [mode.lower()])):
            return mode
    return None


def result_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("mutation_results", "results", "entries"):
        value = data.get(key)
        if isinstance(value, list) and all(isinstance(x, dict) for x in value):
            return value
    summary = data.get("result_summary")
    if isinstance(summary, dict):
        for key in ("mutation_results", "results", "entries"):
            value = summary.get(key)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return value
    return []


def recursive_first(obj: Any, keys: Iterable[str]) -> Any:
    wanted = set(keys)
    if isinstance(obj, dict):
        for key in wanted:
            value = obj.get(key)
            if value not in (None, "", [], {}):
                return value
        for value in obj.values():
            found = recursive_first(value, wanted)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = recursive_first(value, wanted)
            if found not in (None, "", [], {}):
                return found
    return None


def normalize_counts(data: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = data.get("counts")
    if not isinstance(counts, dict):
        counts = (data.get("result_summary") or {}).get("counts")
    if isinstance(counts, dict):
        return {str(k): int(v) for k, v in counts.items() if isinstance(v, (int, float))}
    out: dict[str, int] = {}
    for item in entries:
        status = str(item.get("status") or item.get("classification") or "UNKNOWN")
        out[status] = out.get(status, 0) + 1
    return out


def campaign_is_valid(status: str, counts: dict[str, int]) -> bool:
    if status not in {"COMPLETED", "PASS", "SUCCESS"}:
        return False
    return not any(counts.get(key, 0) > 0 for key in INVALID_STATUSES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    args = parser.parse_args()

    results_root = args.data_root / "results"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latest: dict[str, tuple[Path, dict[str, Any]]] = {}
    parse_errors: list[tuple[str, str]] = []

    for path in results_root.rglob("campaign_result.json"):
        try:
            data = load_json(path)
        except Exception as exc:
            parse_errors.append((str(path), repr(exc)))
            continue
        mode = detect_mode(path, data, args.modes)
        if mode is None:
            continue
        old = latest.get(mode)
        if old is None or path.stat().st_mtime > old[0].stat().st_mtime:
            latest[mode] = (path, data)

    mode_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []

    for mode in args.modes:
        if mode not in latest:
            mode_rows.append(
                {
                    "mode": mode,
                    "campaign_status": "NOT_FOUND",
                    "valid": False,
                    "campaign_id": "",
                    "campaign_result": "",
                    "counts": "{}",
                    "last_modified": "",
                }
            )
            continue

        path, data = latest[mode]
        entries = result_entries(data)
        counts = normalize_counts(data, entries)
        status = str(data.get("status") or (data.get("result_summary") or {}).get("status") or "UNKNOWN")
        valid = campaign_is_valid(status, counts)

        mode_rows.append(
            {
                "mode": mode,
                "campaign_status": status,
                "valid": valid,
                "campaign_id": data.get("campaign_id", ""),
                "campaign_result": str(path),
                "counts": json.dumps(counts, ensure_ascii=False, sort_keys=True),
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )

        for item in entries:
            mutation_rows.append(
                {
                    "mode": mode,
                    "mutation_id": item.get("mutation_id") or item.get("id") or "",
                    "classification": item.get("status") or item.get("classification") or "UNKNOWN",
                    "activation": recursive_first(item, ("activation_status", "activation", "hout_activation_status")),
                    "proof_result_status": recursive_first(item, ("result_status",)),
                    "failure_route": recursive_first(item, ("failure_route",)),
                    "failure_code": recursive_first(item, ("failure_code",)),
                    "violated_obligation_id": recursive_first(item, ("violated_obligation_id", "obligation_id")),
                    "workdir": recursive_first(item, ("workdir", "workspace", "result_dir")),
                }
            )

    mode_csv = args.output_dir / "ppp_mode_summary.csv"
    mutation_csv = args.output_dir / "ppp_mutation_summary.csv"
    report_md = args.output_dir / "ppp_final_report.md"
    summary_json = args.output_dir / "ppp_final_summary.json"

    with mode_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mode_rows)

    mutation_fields = [
        "mode",
        "mutation_id",
        "classification",
        "activation",
        "proof_result_status",
        "failure_route",
        "failure_code",
        "violated_obligation_id",
        "workdir",
    ]
    with mutation_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mutation_fields)
        writer.writeheader()
        writer.writerows(mutation_rows)

    valid_modes = [r["mode"] for r in mode_rows if r["valid"]]
    invalid_modes = [r["mode"] for r in mode_rows if not r["valid"]]

    payload = {
        "schema_version": "ppp_final_summary_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "valid_modes": valid_modes,
        "invalid_or_missing_modes": invalid_modes,
        "mode_rows": mode_rows,
        "mutation_rows": mutation_rows,
        "parse_errors": parse_errors,
        "paper_minimum_policy": "aggregate_independent_modes; do not use incomplete PaperMinimum rerun",
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# PPP 非空洞性实验最终结果汇总",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 数据根目录：`{args.data_root}`",
        f"- 有效 mode：{len(valid_modes)}",
        f"- 无效或缺失 mode：{len(invalid_modes)}",
        "- PaperMinimum：不作为结果来源；从独立 mode 聚合。",
        "",
        "## Mode 汇总",
        "",
        "| Mode | Campaign 状态 | 有效 | Counts |",
        "|---|---|---:|---|",
    ]
    for row in mode_rows:
        lines.append(
            f"| {row['mode']} | {row['campaign_status']} | "
            f"{'是' if row['valid'] else '否'} | `{row['counts']}` |"
        )

    lines += [
        "",
        "## Mutation 汇总",
        "",
        "| Mode | Mutation | 分类 | 激活 | Proof 状态 | Failure route/code | Obligation |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in mutation_rows:
        route_code = " / ".join(
            str(x) for x in (row["failure_route"], row["failure_code"]) if x not in (None, "")
        )
        lines.append(
            f"| {row['mode']} | {row['mutation_id']} | {row['classification']} | "
            f"{row['activation'] or ''} | {row['proof_result_status'] or ''} | "
            f"{route_code} | {row['violated_obligation_id'] or ''} |"
        )

    if parse_errors:
        lines += ["", "## JSON 解析错误", ""]
        for path, error in parse_errors:
            lines.append(f"- `{path}`：`{error}`")

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Mode summary    : {mode_csv}")
    print(f"Mutation summary: {mutation_csv}")
    print(f"Markdown report : {report_md}")
    print(f"JSON summary    : {summary_json}")
    print(f"Valid modes     : {len(valid_modes)}")
    print(f"Invalid/missing : {len(invalid_modes)}")
    return 0 if not invalid_modes else 2


if __name__ == "__main__":
    raise SystemExit(main())
