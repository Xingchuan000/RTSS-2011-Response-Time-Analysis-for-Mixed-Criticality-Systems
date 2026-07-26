"""Global, taskset-independent q-AMC profile specification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class QAmcProfileSpec:
    raw_quality_ranks: tuple[int, ...] = (1, 2, 3, 4)
    normalized_quality: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    isolated_work_ratios: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    isolated_to_interference_ratio: float = 0.5
    integer_partition_rule: str = "minimum_ratio_error_then_lower_w"
    duplicate_level_rule: str = "keep_higher_raw_rank"
    quality_recovery_policy: str = "persistent_no_restore"
    demand_mapping_version: str = "wcet_capped_component_split_v1"
    schema_version: str = "qamc_profile_spec_v2"
    semantic_version: str = "qamc_budget_overlay_v5"

    def __post_init__(self) -> None:
        lengths = {
            len(self.raw_quality_ranks),
            len(self.normalized_quality),
            len(self.isolated_work_ratios),
        }
        if len(lengths) != 1 or not self.raw_quality_ranks:
            raise ValueError("QAMC_PROFILE_SPEC_LENGTH_MISMATCH")
        if tuple(sorted(self.raw_quality_ranks)) != self.raw_quality_ranks or len(set(self.raw_quality_ranks)) != len(self.raw_quality_ranks):
            raise ValueError("QAMC_RAW_RANKS_MUST_BE_STRICTLY_INCREASING")
        if any(not 0.0 < value <= 1.0 for value in self.normalized_quality):
            raise ValueError("QAMC_NORMALIZED_QUALITY_OUT_OF_RANGE")
        if any(not 0.0 < value <= 1.0 for value in self.isolated_work_ratios):
            raise ValueError("QAMC_ISOLATED_WORK_RATIO_OUT_OF_RANGE")
        if tuple(sorted(self.normalized_quality)) != self.normalized_quality:
            raise ValueError("QAMC_NORMALIZED_QUALITY_NOT_MONOTONIC")
        if tuple(sorted(self.isolated_work_ratios)) != self.isolated_work_ratios:
            raise ValueError("QAMC_ISOLATED_WORK_RATIO_NOT_MONOTONIC")
        if self.normalized_quality[-1] != 1.0:
            raise ValueError("QAMC_MAX_NORMALIZED_QUALITY_MUST_BE_ONE")
        if self.isolated_work_ratios[-1] != 1.0:
            raise ValueError("QAMC_MAX_ISOLATED_WORK_RATIO_MUST_BE_ONE")
        if not math.isfinite(self.isolated_to_interference_ratio) or self.isolated_to_interference_ratio < 0.0:
            raise ValueError("QAMC_RATIO_MUST_BE_NONNEGATIVE")
        if self.integer_partition_rule != "minimum_ratio_error_then_lower_w":
            raise ValueError("QAMC_UNSUPPORTED_INTEGER_PARTITION_RULE")
        if self.duplicate_level_rule != "keep_higher_raw_rank":
            raise ValueError("QAMC_UNSUPPORTED_DUPLICATE_LEVEL_RULE")
        if self.quality_recovery_policy != "persistent_no_restore":
            raise ValueError("QAMC_UNSUPPORTED_QUALITY_RECOVERY_POLICY")
        if self.demand_mapping_version != "wcet_capped_component_split_v1":
            raise ValueError("QAMC_UNSUPPORTED_DEMAND_MAPPING_VERSION")

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("raw_quality_ranks", "normalized_quality", "isolated_work_ratios"):
            payload[key] = list(payload[key])
        return payload

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> "QAmcProfileSpec":
        values = dict(payload)
        for key in ("raw_quality_ranks", "normalized_quality", "isolated_work_ratios"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_jsonable(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_profile_spec(path: str | Path) -> QAmcProfileSpec:
    with Path(path).open(encoding="utf-8") as handle:
        return QAmcProfileSpec.from_jsonable(json.load(handle))


def write_profile_spec(spec: QAmcProfileSpec, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(spec.to_jsonable(), handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = ["QAmcProfileSpec", "load_profile_spec", "write_profile_spec"]
