"""Resolve the authoritative D1 minimum-slack envelope coordinate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..mutators.action_config import get_pointer


def resolve_envelope_target(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed_root: Path,
) -> dict[str, Any]:
    """Select only RTA rows that explicitly bind an envelope source coordinate.

    The resolver deliberately does not infer a coordinate from task names or
    budgets.  Such inference could turn D1 into a different experiment.  The
    authoritative RTA/audit artifact must provide both ``envelope_target_file``
    and ``envelope_json_pointer``.
    """

    candidates: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        try:
            slack = float(row["slack"])
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not math.isfinite(slack)
            or seed < 0
            or not row.get("limiting_lo_task")
            or not row.get("envelope_target_file")
            or not row.get("envelope_json_pointer")
            or str(row.get("variant", "unknown")) == "unknown"
        ):
            continue
        candidates.append((slack, row))
    if not candidates:
        raise ValueError("D1_ENVELOPE_TARGET_UNRESOLVED")

    _, selected = min(candidates, key=lambda item: item[0])
    seed = int(selected["seed"])
    seed_dir = _find_seed_dir(Path(seed_root), seed)
    relative = Path(str(selected["envelope_target_file"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("D1_UNSAFE_ENVELOPE_TARGET_FILE")
    target_path = (seed_dir / relative).resolve()
    if seed_dir.resolve() not in target_path.parents or not target_path.is_file():
        raise ValueError(f"D1_ENVELOPE_TARGET_FILE_MISSING:{target_path}")
    raw = json.loads(target_path.read_text(encoding="utf-8"))
    pointer = str(selected["envelope_json_pointer"])
    baseline = get_pointer(raw, pointer)
    if isinstance(baseline, bool) or not isinstance(baseline, int):
        raise ValueError("D1_ENVELOPE_COORDINATE_NOT_INTEGER")
    return {
        "seed": seed,
        "tree_variant": _normalize_variant(str(selected["variant"])),
        "seed_dir": str(seed_dir.resolve()),
        "baseline_slack": float(selected["slack"]),
        "limiting_hi_task": str(selected["task_id"]),
        "limiting_lo_task": str(selected["limiting_lo_task"]),
        "target_file": relative.as_posix(),
        "json_pointer": pointer,
        "baseline_value": int(baseline),
        "source_rta_artifact": str(selected.get("artifact", "")),
    }


def _find_seed_dir(root: Path, seed: int) -> Path:
    direct = root / f"s{seed}"
    if direct.is_dir():
        return direct
    matches = [path for path in root.rglob(f"s{seed}") if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(f"D1_SEED_DIR_UNRESOLVED:s{seed}:{len(matches)}")
    return matches[0]


def _normalize_variant(value: str) -> str:
    return {"compact": "best_overall", "balanced": "best_balanced"}.get(value, value)
