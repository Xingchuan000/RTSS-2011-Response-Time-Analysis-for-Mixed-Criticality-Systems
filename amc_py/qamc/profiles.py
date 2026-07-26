"""Materialization and persistence of per-taskset q-AMC profiles."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from amc_py.models import Criticality, Task

from .models import QAmcProfileBundle, QAmcQualityLevel, QAmcTaskProfile
from .profile_spec import QAmcProfileSpec


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def partition_design_budget(
    design_c_lo: int,
    *,
    isolated_to_interference_ratio: float,
) -> tuple[int, int]:
    """Return integer ``W_max, I_design`` minimizing ``W/I`` ratio error."""

    if design_c_lo <= 0:
        raise ValueError("QAMC_C_LO_MUST_BE_POSITIVE")
    if isolated_to_interference_ratio < 0.0:
        raise ValueError("QAMC_RATIO_MUST_BE_NONNEGATIVE")
    if design_c_lo == 1:
        return 1, 0
    candidates: list[tuple[float, int, int]] = []
    for isolated in range(1, design_c_lo):
        interference = design_c_lo - isolated
        error = abs(isolated / interference - isolated_to_interference_ratio)
        candidates.append((error, isolated, interference))
    _, isolated, interference = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return isolated, interference


def _profile_for_task(task: Task, spec: QAmcProfileSpec) -> QAmcTaskProfile:
    w_max, interference = partition_design_budget(
        task.c_lo,
        isolated_to_interference_ratio=spec.isolated_to_interference_ratio,
    )
    if task.criticality is Criticality.HI:
        return None  # type: ignore[return-value]
    candidates: dict[int, tuple[int, float, float]] = {}
    for raw_rank, quality, ratio in zip(
        spec.raw_quality_ranks, spec.normalized_quality, spec.isolated_work_ratios, strict=True
    ):
        w_q = max(1, math.ceil(w_max * ratio))
        # A duplicate integer level retains the higher raw rank/quality.
        candidates[w_q] = (raw_rank, quality, ratio)
    levels = tuple(
        QAmcQualityLevel(
            runtime_level=index,
            raw_rank=raw_rank,
            normalized_quality=quality,
            isolated_work_ratio=ratio,
            isolated_wcet=w_q,
        )
        for index, (w_q, (raw_rank, quality, ratio)) in enumerate(sorted(candidates.items()))
    )
    return QAmcTaskProfile(
        task_name=task.name,
        design_c_lo=task.c_lo,
        full_quality_isolated_wcet=w_max,
        design_lo_interference_budget=interference,
        levels=levels,
        initial_runtime_level=len(levels) - 1,
        threshold_runtime_level=0,
        degradable=len(levels) > 1,
    )


def qamc_profile_bundle_fingerprint(bundle_payload: dict[str, Any]) -> str:
    payload = dict(bundle_payload)
    payload.pop("fingerprint", None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def build_qamc_profile_bundle(
    ordered_tasks: Sequence[Task],
    *,
    taskset_fingerprint: str,
    spec: QAmcProfileSpec,
) -> QAmcProfileBundle:
    profiles = {
        task.name: profile
        for task in ordered_tasks
        if (profile := _profile_for_task(task, spec)) is not None
    }
    payload = {
        "schema_version": "qamc_profile_bundle_v2",
        "semantic_version": spec.semantic_version,
        "taskset_fingerprint": taskset_fingerprint,
        "spec_fingerprint": spec.fingerprint,
        "ratio_semantics": "isolated_work_to_interference",
        "integer_partition_rule": spec.integer_partition_rule,
        "demand_mapping_version": spec.demand_mapping_version,
        "profiles": {name: profile.to_jsonable() for name, profile in sorted(profiles.items())},
    }
    fingerprint = qamc_profile_bundle_fingerprint(payload)
    return QAmcProfileBundle(
        schema_version=payload["schema_version"],
        semantic_version=payload["semantic_version"],
        taskset_fingerprint=payload["taskset_fingerprint"],
        spec_fingerprint=payload["spec_fingerprint"],
        profiles=profiles,
        fingerprint=fingerprint,
        ratio_semantics=payload["ratio_semantics"],
        integer_partition_rule=payload["integer_partition_rule"],
        demand_mapping_version=payload["demand_mapping_version"],
    )


def profile_bundle_from_jsonable(payload: dict[str, Any]) -> QAmcProfileBundle:
    profiles: dict[str, QAmcTaskProfile] = {}
    for name, raw in payload["profiles"].items():
        levels = tuple(QAmcQualityLevel(**level) for level in raw["levels"])
        profile = QAmcTaskProfile(
            task_name=raw["task_name"],
            design_c_lo=int(raw["design_c_lo"]),
            full_quality_isolated_wcet=int(raw["full_quality_isolated_wcet"]),
            design_lo_interference_budget=int(raw["design_lo_interference_budget"]),
            levels=levels,
            initial_runtime_level=int(raw["initial_runtime_level"]),
            threshold_runtime_level=int(raw["threshold_runtime_level"]),
            degradable=bool(raw["degradable"]),
        )
        if name != profile.task_name:
            raise ValueError("QAMC_PROFILE_KEY_NAME_MISMATCH")
        profiles[name] = profile
    bundle = QAmcProfileBundle(
        schema_version=payload["schema_version"],
        semantic_version=payload["semantic_version"],
        taskset_fingerprint=payload["taskset_fingerprint"],
        spec_fingerprint=payload["spec_fingerprint"],
        profiles=profiles,
        fingerprint=payload["fingerprint"],
        ratio_semantics=payload.get("ratio_semantics", "isolated_work_to_interference"),
        integer_partition_rule=payload.get(
            "integer_partition_rule", "minimum_ratio_error_then_lower_w"
        ),
        demand_mapping_version=payload.get(
            "demand_mapping_version", "wcet_capped_component_split_v1"
        ),
    )
    expected = qamc_profile_bundle_fingerprint(
        {key: value for key, value in payload.items() if key != "fingerprint"}
    )
    if expected != bundle.fingerprint:
        raise ValueError("QAMC_PROFILE_FINGERPRINT_MISMATCH")
    return bundle


def write_profile_bundle(bundle: QAmcProfileBundle, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(bundle.to_jsonable(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_profile_bundle(path: str | Path) -> QAmcProfileBundle:
    with Path(path).open(encoding="utf-8") as handle:
        return profile_bundle_from_jsonable(json.load(handle))


__all__ = [
    "build_qamc_profile_bundle",
    "load_profile_bundle",
    "partition_design_budget",
    "profile_bundle_from_jsonable",
    "qamc_profile_bundle_fingerprint",
    "write_profile_bundle",
]
