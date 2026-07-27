"""D1 dynamic minimum-slack selection over ordinary proof artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEADLINE_KEYS = ("deadline", "D", "deadline_ticks")
R_LO_KEYS = ("R_LO", "r_lo", "response_time_lo")
R_HI_KEYS = ("R_HI", "r_hi", "response_time_hi")
TASK_KEYS = ("task_id", "task", "name")


def scan_rta_slack(bundle_roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in bundle_roots:
        root = Path(root).resolve()
        for path in sorted(root.rglob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for candidate in _objects(raw):
                adapted = _adapt_rta_record(candidate)
                if adapted is None:
                    continue
                adapted.update(
                    {
                        "bundle_root": str(root),
                        "artifact": str(path),
                        "seed": _infer_seed(root, candidate),
                        "variant": _infer_variant(root, candidate),
                    }
                )
                rows.append(adapted)
    unique = {
        (
            row["artifact"],
            row["task_id"],
            row["deadline"],
            row["r_lo"],
            row["r_hi"],
        ): row
        for row in rows
    }
    return sorted(unique.values(), key=lambda item: (item["slack"], item["seed"], item["task_id"]))


def select_minimum_slack(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("未发现可解析的 HI RTA record")
    return dict(min(materialized, key=lambda item: float(item["slack"])))


def find_delta_star(results: Iterable[Mapping[str, Any]]) -> int | None:
    ordered = sorted(results, key=lambda item: int(item["delta"]))
    for row in ordered:
        if str(row.get("result_status")) != "DEPLOYED_TREE_PROVED":
            return int(row["delta"])
    return None


def _adapt_rta_record(value: Mapping[str, Any]) -> dict[str, Any] | None:
    deadline = _first_number(value, DEADLINE_KEYS)
    r_lo = _first_number(value, R_LO_KEYS)
    r_hi = _first_number(value, R_HI_KEYS)
    if deadline is None or r_lo is None or r_hi is None:
        return None
    task_id = _first(value, TASK_KEYS)
    if task_id is None:
        return None
    criticality = str(value.get("criticality", value.get("level", "HI"))).upper()
    if criticality not in {"HI", "HIGH", "1", "TRUE"}:
        return None
    return {
        "task_id": str(task_id),
        "deadline": deadline,
        "r_lo": r_lo,
        "r_hi": r_hi,
        "slack": deadline - max(r_lo, r_hi),
        "limiting_case": "LO" if r_lo >= r_hi else "HI",
        "witness": value.get("witness", value.get("limiting_witness")),
        "lo_interference_components": value.get(
            "lo_interference_components", value.get("interference", [])
        ),
        "limiting_lo_task": value.get("limiting_lo_task"),
        "envelope_target_file": value.get("envelope_target_file"),
        "envelope_json_pointer": value.get("envelope_json_pointer"),
    }


def _objects(value: Any):
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _objects(item)


def _first_number(value: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    raw = _first(value, keys)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _first(value: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    return next((value[key] for key in keys if key in value), None)


def _infer_seed(root: Path, value: Mapping[str, Any]) -> int:
    raw = value.get("taskset_seed", value.get("seed"))
    if raw is not None:
        return int(raw)
    for part in root.parts:
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:])
    return -1


def _infer_variant(root: Path, value: Mapping[str, Any]) -> str:
    raw = value.get("tree_variant", value.get("variant"))
    if raw:
        return str(raw)
    return next(
        (
            part
            for part in root.parts
            if part in {"best_overall", "best_balanced", "best_performance", "compact", "balanced"}
        ),
        "unknown",
    )
