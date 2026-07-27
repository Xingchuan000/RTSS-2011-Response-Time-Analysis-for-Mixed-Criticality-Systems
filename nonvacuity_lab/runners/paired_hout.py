"""Deterministic paired HOUT command runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..analysis.comparison import compare_metrics
from ..canonical import canonical_json_hash
from ..subprocess_runner import run_command


DETERMINISM_KEYS = (
    "taskset",
    "scenario_seeds",
    "demand_trace",
    "horizon",
    "controller_release_times",
    "worker_count",
    "runtime_config",
    "random_seed",
)


def run_paired_hout(
    *,
    config: Mapping[str, Any],
    workspace_root: Path,
    command_context: Mapping[str, str],
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    determinism = {key: config.get(key) for key in DETERMINISM_KEYS}
    if any(value is None for value in determinism.values()):
        missing = [key for key, value in determinism.items() if value is None]
        return {"setup_valid": False, "reason": f"HOUT determinism fields missing: {missing}"}
    base_dir = workspace_root / "base"
    mutated_dir = workspace_root / "mutated"
    base_dir.mkdir(parents=True, exist_ok=True)
    mutated_dir.mkdir(parents=True, exist_ok=True)
    base_command = _format_command(config.get("base_command"), command_context | {"out": str(base_dir)})
    mutated_command = _format_command(
        config.get("mutated_command"), command_context | {"out": str(mutated_dir)}
    )
    base_receipt = run_command(
        base_command,
        cwd=Path(command_context.get("base_source_root", command_context["source_root"])),
        log_dir=base_dir / "logs",
        timeout_seconds=timeout_seconds,
    )
    mutated_source = Path(
        command_context.get("mutated_source_root", command_context["source_root"])
    ).resolve()
    base_source = Path(command_context["source_root"]).resolve()
    mutated_env = {}
    if mutated_source != base_source:
        mutated_env["PYTHONPATH"] = os.pathsep.join(
            [str(mutated_source), str(base_source)]
        )
    mutated_receipt = run_command(
        mutated_command,
        cwd=mutated_source,
        log_dir=mutated_dir / "logs",
        env=mutated_env,
        timeout_seconds=timeout_seconds,
    )
    base_summary = _read_json(base_dir / str(config.get("summary_file", "summary.json")))
    mutated_summary = _read_json(mutated_dir / str(config.get("summary_file", "summary.json")))
    determinism_checks = _runtime_determinism_checks(base_summary, mutated_summary)
    events_name = str(config.get("events_file", "events.jsonl"))
    result = {
        "schema_version": "paired_hout_run_v1",
        "setup_valid": (
            base_receipt["returncode"] == 0
            and mutated_receipt["returncode"] == 0
            and all(item["match"] for item in determinism_checks.values())
        ),
        "determinism_hash": canonical_json_hash(determinism),
        "determinism": determinism,
        "base_receipt": base_receipt,
        "mutated_receipt": mutated_receipt,
        "runtime_determinism_checks": determinism_checks,
        "comparison": compare_metrics(
            base_summary,
            mutated_summary,
            metrics=tuple(str(item) for item in config.get("metrics", ())),
        ),
        "base_events": str(base_dir / events_name),
        "mutated_events": str(mutated_dir / events_name),
    }
    (workspace_root / "paired_hout_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _format_command(raw: Any, context: Mapping[str, str]) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("HOUT command 必须为非空字符串数组")
    return [str(item).format_map(context) for item in raw]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def _runtime_determinism_checks(
    base: Mapping[str, Any],
    mutated: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    aliases = {
        "demand_trace_fingerprint": ("demand_trace_fingerprint", "demand_fingerprint"),
        "taskset_hash": ("taskset_hash", "taskset_fingerprint"),
        "scenario_list": ("scenario_list", "scenario_ids", "scenario_seeds"),
        "horizon": ("horizon",),
    }
    checks: dict[str, dict[str, Any]] = {}
    for label, keys in aliases.items():
        left = next((base[key] for key in keys if key in base), None)
        right = next((mutated[key] for key in keys if key in mutated), None)
        checks[label] = {
            "base": left,
            "mutated": right,
            "present": left is not None and right is not None,
            "match": left is not None and right is not None and left == right,
        }
    return checks
