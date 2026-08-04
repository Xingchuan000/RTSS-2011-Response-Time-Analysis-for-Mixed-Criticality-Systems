"""Deterministic paired HOUT command runner."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from ..analysis.comparison import compare_metrics
from ..canonical import canonical_json_hash
from ..subprocess_runner import run_command
from ..hout.normalizer import load_events as load_normalized_events
from ..hout.aggregate import compare_paired_hout


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
    config = _materialize_profile_inputs(
        _normalize_profile_config(config),
        workspace_root=workspace_root,
    )
    determinism = {key: config.get(key) for key in DETERMINISM_KEYS}
    required_determinism = set(DETERMINISM_KEYS) - {"demand_trace"}
    if any(determinism[key] is None for key in required_determinism):
        missing = [key for key in required_determinism if determinism[key] is None]
        return {"setup_valid": False, "reason": f"HOUT determinism fields missing: {missing}"}
    base_dir = workspace_root / "base"
    mutated_dir = workspace_root / "mutated"
    base_seed = Path(command_context["base_seed"]).resolve()
    mutated_seed = Path(command_context["mutated_seed"]).resolve()
    variant = str(command_context["tree_variant"])
    common_context = {
        **dict(command_context),
        "scenario_file": str(config["scenario_file"]),
        "runtime_config": str(config["runtime_config"]),
        "taskset": str(config["taskset"]),
    }
    base_context = {
        **common_context,
        "seed_dir": str(base_seed),
        "tree_path": str(base_seed / variant / "integer_tree.json"),
        "out": str(base_dir),
        "output_dir": str(base_dir),
    }
    mutated_context = {
        **common_context,
        "seed_dir": str(mutated_seed),
        "tree_path": str(mutated_seed / variant / "integer_tree.json"),
        "out": str(mutated_dir),
        "output_dir": str(mutated_dir),
    }
    base_command = _format_command(config.get("base_command"), base_context)
    mutated_command = _format_command(config.get("mutated_command"), mutated_context)
    base_receipt = run_command(
        base_command,
        cwd=Path(command_context.get("base_source_root", command_context["source_root"])),
        log_dir=base_dir / "logs",
        env={"PYTHONPATH": str(Path(command_context.get("base_source_root", command_context["source_root"])).resolve())},
        timeout_seconds=timeout_seconds,
    )
    mutated_source = Path(
        command_context.get("mutated_source_root", command_context["source_root"])
    ).resolve()
    base_source = Path(command_context["source_root"]).resolve()
    mutated_env = {"PYTHONPATH": str(mutated_source)}
    # Never mix clean and overlay imports: ordinary base and mutated runs
    # must each have exactly one source root in a fresh process.
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
    try:
        base_events = load_normalized_events(base_dir / events_name)
        mutated_events = load_normalized_events(mutated_dir / events_name)
        required_scenarios = set(config.get("required_scenarios", ()))
        seen_scenarios = {event.scenario_seed for event in base_events} & {event.scenario_seed for event in mutated_events}
        if not required_scenarios <= seen_scenarios:
            raise ValueError(f"required scenario missing from paired outputs: {sorted(required_scenarios - seen_scenarios)}")
        result["normalized_comparison"] = compare_paired_hout(base_events, mutated_events)
        result["hout_schema_status"] = "PASS"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result["hout_schema_status"] = "HOUT_SCHEMA_FAILED"
        result["hout_schema_error"] = str(exc)
        result["setup_valid"] = False
    (workspace_root / "paired_hout_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result



def _materialize_profile_inputs(
    config: Mapping[str, Any],
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    """Freeze the paired HOUT inputs inside the experiment workspace.

    The two fresh processes receive the same scenario list and runtime config
    files.  Only the source root, seed directory and tree path differ.
    """
    raw = dict(config)
    inputs = workspace_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)

    scenario_value = raw.get("scenario_seeds")
    if isinstance(scenario_value, (list, tuple)):
        scenario_file = inputs / "scenarios.json"
        scenario_file.write_text(
            json.dumps([int(item) for item in scenario_value], indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        source = Path(str(raw.get("scenario_file", scenario_value or ""))).resolve()
        if not source.is_file():
            raise ValueError(f"HOUT scenario file missing: {source}")
        scenario_file = inputs / "scenarios.json"
        shutil.copy2(source, scenario_file)

    runtime_source = Path(str(raw.get("runtime_config_path", raw.get("runtime_config", "")))).resolve()
    if not runtime_source.is_file():
        raise ValueError(f"HOUT runtime config missing: {runtime_source}")
    runtime_file = inputs / "runtime_config.json"
    shutil.copy2(runtime_source, runtime_file)

    taskset = Path(str(raw.get("taskset_path", raw.get("taskset", "")))).resolve()
    if not taskset.is_file():
        raise ValueError(f"HOUT taskset missing: {taskset}")

    raw["scenario_file"] = str(scenario_file)
    raw["runtime_config"] = str(runtime_file)
    raw["taskset"] = str(taskset)
    return raw

def _format_command(raw: Any, context: Mapping[str, str]) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("HOUT command 必须为非空字符串数组")
    values = dict(context)
    if "out" in values:
        values.setdefault("output_dir", values["out"])
    if "output_dir" in values:
        values.setdefault("out", values["output_dir"])
    return [str(item).format_map(values) for item in raw]


def _normalize_profile_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Accept both the v1 manifest HOUT shape and the Phase-5 profile shape."""
    raw = dict(config)
    aliases = {
        "taskset": "taskset_path",
        "demand_trace": "demand_trace_path",
        "runtime_config": "runtime_config_path",
        "summary_file": "summary_relative_path",
        "events_file": "events_relative_path",
    }
    for old, new in aliases.items():
        if raw.get(old) is None and raw.get(new) is not None:
            raw[old] = raw[new]
    if raw.get("scenario_seeds") is None and raw.get("scenario_file") is not None:
        raw["scenario_seeds"] = raw["scenario_file"]
    if raw.get("metrics") is None and raw.get("required_metrics") is not None:
        raw["metrics"] = raw["required_metrics"]
    return raw


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
