#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Five-strata, structure-only selector for ``mc_stratified_dynamic``."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from scripts.diagnose_mc_stratified_dynamic_structure import (
    DIAGNOSTICS_SCHEMA_VERSION,
    FORBIDDEN_SELECTION_PREFIXES,
    MANIFEST_SCHEMA_VERSION,
    SELECTION_CONFIG,
    SELECTION_FEATURES,
    canonical_hash,
    normalize_period_family,
    selection_config_hash,
)


PRIMARY10_SCHEMA_VERSION = "mc_stratified_dynamic_primary10_v2"
AUDIT_SCHEMA_VERSION = "mc_stratified_dynamic_selection_audit_v2"

STRATA = ("S1", "S2", "S3", "S4", "S5")
PERIOD_FAMILIES = ("semi-harmonic", "log-uniform")

# Values are pre-declared regime prototypes, not hard gates.  Unspecified
# dimensions use the neutral midpoint so all eight declared features
# participate in every distance calculation.
PROTOTYPES: dict[str, dict[str, float]] = {
    "S1": {
        "load_p": 0.50,
        "tightness_p": 0.45,
        "autocorr_p": 0.45,
        "stress_p": 0.50,
        "leader_turnover_p": 0.40,
        "mask_turnover_p": 0.50,
        "competition_p": 0.40,
        "mode_pressure_p": 0.35,
    },
    "S2": {
        "load_p": 0.50,
        "tightness_p": 0.50,
        "autocorr_p": 0.80,
        "stress_p": 0.75,
        "leader_turnover_p": 0.45,
        "mask_turnover_p": 0.50,
        "competition_p": 0.45,
        "mode_pressure_p": 0.35,
    },
    "S3": {
        "load_p": 0.80,
        "tightness_p": 0.80,
        "autocorr_p": 0.50,
        "stress_p": 0.50,
        "leader_turnover_p": 0.50,
        "mask_turnover_p": 0.60,
        "competition_p": 0.80,
        "mode_pressure_p": 0.50,
    },
    "S4": {
        "load_p": 0.60,
        "tightness_p": 0.65,
        "autocorr_p": 0.50,
        "stress_p": 0.50,
        "leader_turnover_p": 0.50,
        "mask_turnover_p": 0.50,
        "competition_p": 0.55,
        "mode_pressure_p": 0.80,
    },
    "S5": {
        "load_p": 0.50,
        "tightness_p": 0.50,
        "autocorr_p": 0.65,
        "stress_p": 0.50,
        "leader_turnover_p": 0.80,
        "mask_turnover_p": 0.80,
        "competition_p": 0.65,
        "mode_pressure_p": 0.50,
    },
}

DISTANCE_WEIGHTS = {feature: 1.0 for feature in SELECTION_FEATURES}
DIVERSITY_PENALTY = 0.05
DIVERSITY_RADIUS = 0.08

STRUCTURAL_VARIATION_FIELDS = (
    "mask_turnover_rate",
    "budget_competition_index",
)
STRUCTURAL_VARIATION_EPS = 1e-12
STRUCTURAL_VARIATION_MIN_UNIQUE = 3
STRUCTURAL_VARIATION_MIN_POSITIVE = 4



class SelectionShortageError(RuntimeError):
    """Raised when fail-closed selection cannot fill all 5 x 2 slots."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__(json.dumps(self.report, ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True, slots=True)
class SelectedCandidate:
    row: dict[str, Any]
    stratum: str
    period_family: str
    prototype_distance: float
    rank_within_stratum_family: int
    feature_vector: dict[str, float]


def assert_selection_feature_guard(feature_names: Sequence[str] = SELECTION_FEATURES) -> None:
    """Ensure the explicit selection whitelist cannot contain performance fields."""

    violations = [
        name
        for name in feature_names
        if any(name.startswith(prefix) for prefix in FORBIDDEN_SELECTION_PREFIXES)
    ]
    if violations:
        raise AssertionError(f"forbidden selection features: {violations}")


def validate_diagnostics_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("diagnostics CSV is empty")
    for index, row in enumerate(rows, start=2):
        if str(row.get("schema_version", "")).strip() != DIAGNOSTICS_SCHEMA_VERSION:
            raise ValueError(
                f"diagnostics row {index} has unsupported schema_version="
                f"{row.get('schema_version')!r}; expected {DIAGNOSTICS_SCHEMA_VERSION!r}"
            )
        if str(row.get("input_schema_version", "")).strip() not in {
            MANIFEST_SCHEMA_VERSION,
            "",
        }:
            raise ValueError("diagnostics input_schema_version is not mc_stratified_dynamic_manifest_v1")
        if str(row.get("candidate_seed", "")).strip() == "":
            raise ValueError(f"diagnostics row {index} is missing candidate_seed")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    validate_diagnostics_rows(rows)
    return rows


def _float(row: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _int(row: Mapping[str, Any], name: str, default: int = 0) -> int:
    try:
        return int(float(row.get(name, default)))
    except (TypeError, ValueError):
        return default


def _percentile(value: float, values: Sequence[float]) -> float:
    """Midrank percentile in [0, 1], deterministic under ties."""

    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return 0.5
    lower = sum(item < value for item in ordered)
    equal = sum(item == value for item in ordered)
    return (lower + 0.5 * equal) / float(len(ordered))


def _raw_feature_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    return {
        "load": [_float(row, "total_util_lo_mode") for row in rows],
        "tightness": [-_float(row, "analysis_normalized_slack") for row in rows],
        "autocorr": [
            statistics.fmean(
                [_float(row, "lo_cost_lag1_autocorr_mean"), _float(row, "hi_cost_lag1_autocorr_mean")]
            )
            for row in rows
        ],
        "stress": [_float(row, "stress_duty_empirical_mean") for row in rows],
        "leader_turnover": [
            statistics.fmean(
                [
                    _float(row, "stress_leader_turnover_rate"),
                    _float(row, "lo_pressure_leader_turnover_rate"),
                ]
            )
            for row in rows
        ],
        "mask_turnover": [_float(row, "mask_turnover_rate") for row in rows],
        "competition": [_float(row, "budget_competition_index") for row in rows],
        "mode_pressure": [
            statistics.fmean(
                [
                    _float(row, "mode_change_rate"),
                    _float(row, "hi_overrun_event_rate"),
                    _float(row, "fraction_time_hi_mode"),
                ]
            )
            for row in rows
        ],
    }


def build_percentile_features(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, float]]:
    """Convert only declared structural metrics to accepted-pool percentiles."""

    assert_selection_feature_guard()
    raw = _raw_feature_values(rows)
    result: dict[int, dict[str, float]] = {}
    for index, row in enumerate(rows):
        result[int(row["candidate_seed"])] = {
            "load_p": _percentile(raw["load"][index], raw["load"]),
            "tightness_p": _percentile(raw["tightness"][index], raw["tightness"]),
            "autocorr_p": _percentile(raw["autocorr"][index], raw["autocorr"]),
            "stress_p": _percentile(raw["stress"][index], raw["stress"]),
            "leader_turnover_p": _percentile(raw["leader_turnover"][index], raw["leader_turnover"]),
            "mask_turnover_p": _percentile(raw["mask_turnover"][index], raw["mask_turnover"]),
            "competition_p": _percentile(raw["competition"][index], raw["competition"]),
            "mode_pressure_p": _percentile(raw["mode_pressure"][index], raw["mode_pressure"]),
        }
    return result


def prototype_distance(feature_vector: Mapping[str, float], stratum: str) -> float:
    if stratum not in PROTOTYPES:
        raise ValueError(f"unknown stratum: {stratum}")
    return math.sqrt(
        sum(
            DISTANCE_WEIGHTS[name] * (float(feature_vector[name]) - PROTOTYPES[stratum][name]) ** 2
            for name in SELECTION_FEATURES
        )
    )


def _vector_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return math.sqrt(sum((left[name] - right[name]) ** 2 for name in SELECTION_FEATURES))


def _is_accepted_gate(row: Mapping[str, Any], *, require_schedulable: bool) -> bool:
    if _float(row, "baseline_deadline_misses_sum") > 0.0:
        return False
    if require_schedulable:
        if str(row.get("admission_method", "")).strip() != "c_amc_sem":
            return False
        if str(row.get("admission_priority_policy", "")).strip() != "opa":
            return False
        raw_schedulable = row.get("analysis_schedulable", row.get("admission_schedulable", ""))
        if str(raw_schedulable).strip().lower() not in {"1", "true", "yes", "y"}:
            return False
    # Symmetric and intentionally loose action-space gate.
    if "valid_lo_increase_count_mean" in row and _float(row, "valid_lo_increase_count_mean") < 1.0:
        return False
    if "valid_lo_decrease_count_mean" in row and _float(row, "valid_lo_decrease_count_mean") < 1.0:
        return False
    if "mask_observation_count" in row and _int(row, "mask_observation_count") <= 0:
        return False
    return True


def accepted_population(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_schedulable: bool = False,
) -> list[dict[str, Any]]:
    """Apply only structural, baseline-quality and symmetric availability gates."""

    candidates = [dict(row) for row in rows if _is_accepted_gate(row, require_schedulable=require_schedulable)]
    # Very loose endpoint exclusion.  This is a population-level sanity gate,
    # never a performance ranking; if a field is absent it is not invented.
    qos_rows = [row for row in candidates if "baseline_lo_quality_qos" in row]
    if len(qos_rows) >= 5:
        qos_values = sorted(_float(row, "baseline_lo_quality_qos") for row in qos_rows)
        low = qos_values[max(0, math.floor(0.05 * (len(qos_values) - 1)))]
        high = qos_values[min(len(qos_values) - 1, math.ceil(0.95 * (len(qos_values) - 1)))]
        candidates = [
            row
            for row in candidates
            if "baseline_lo_quality_qos" not in row
            or low <= _float(row, "baseline_lo_quality_qos") <= high
        ]
    return candidates


def structural_variation_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit whether selector-critical structural metrics actually vary.

    S3 and S5 are only scientifically meaningful when competition and the
    deployment safety-frontier turnover are observable in the accepted D2
    population.  This guard does not rank candidates; it only prevents a
    constant-zero diagnostic from silently becoming percentile 0.5 for every
    taskset.
    """

    metrics: dict[str, Any] = {}
    problems: list[dict[str, Any]] = []
    families = PERIOD_FAMILIES
    for field in STRUCTURAL_VARIATION_FIELDS:
        values = [_float(row, field) for row in rows]
        rounded_unique = sorted({round(value, 12) for value in values})
        positive_rows = [row for row in rows if _float(row, field) > STRUCTURAL_VARIATION_EPS]
        positive_by_family = {
            family: sum(
                normalize_period_family(row.get("period_family")) == family
                and _float(row, field) > STRUCTURAL_VARIATION_EPS
                for row in rows
            )
            for family in families
        }
        stats = {
            "count": len(values),
            "min": min(values, default=0.0),
            "max": max(values, default=0.0),
            "mean": statistics.fmean(values) if values else 0.0,
            "unique_rounded_12_count": len(rounded_unique),
            "positive_count": len(positive_rows),
            "positive_by_period_family": positive_by_family,
        }
        metrics[field] = stats
        field_problems: list[str] = []
        if len(values) >= 10:
            if stats["max"] - stats["min"] <= STRUCTURAL_VARIATION_EPS:
                field_problems.append("near_constant")
            if stats["unique_rounded_12_count"] < STRUCTURAL_VARIATION_MIN_UNIQUE:
                field_problems.append("too_few_unique_values")
            if stats["positive_count"] < STRUCTURAL_VARIATION_MIN_POSITIVE:
                field_problems.append("too_few_positive_candidates")
            for family, count in positive_by_family.items():
                if count < 1:
                    field_problems.append(f"no_positive_candidate_in_{family}")
        if field_problems:
            problems.append({"field": field, "reasons": field_problems, "stats": stats})
    return {
        "schema_version": "mc_stratified_dynamic_structural_variation_report_v1",
        "candidate_count": len(rows),
        "metrics": metrics,
        "problems": problems,
        "ok": not problems,
    }


def assert_structural_feature_variation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    report = structural_variation_report(rows)
    if len(rows) >= 10 and not report["ok"]:
        raise SelectionShortageError(report)
    return report


def _stratum_structural_gate(row: Mapping[str, Any], stratum: str) -> bool:
    """Minimal structure-only identity gates for named strata.

    Prototype distance still performs the ranking.  These gates only ensure
    that S3/S5 actually exhibit the property named by the stratum instead of
    receiving a percentile score derived from zero/near-zero measurements.
    """

    if stratum == "S3":
        return _float(row, "budget_competition_index") > STRUCTURAL_VARIATION_EPS
    if stratum == "S5":
        return _float(row, "mask_turnover_rate") > STRUCTURAL_VARIATION_EPS
    return True


def _diversity_adjustment(vector: Mapping[str, float], selected: Sequence[SelectedCandidate]) -> float:
    if not selected:
        return 0.0
    nearest = min(_vector_distance(vector, item.feature_vector) for item in selected)
    return DIVERSITY_PENALTY * max(0.0, DIVERSITY_RADIUS - nearest)


def _shortage_report(rows: Sequence[Mapping[str, Any]], accepted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available_by_family = {
        family: sum(normalize_period_family(row.get("period_family")) == family for row in accepted)
        for family in PERIOD_FAMILIES
    }
    counts = {
        stratum: {
            family: available_by_family[family]
            for family in PERIOD_FAMILIES
        }
        for stratum in STRATA
    }
    return {
        "schema_version": "mc_stratified_dynamic_shortage_report_v1",
        "required_slots": {stratum: list(PERIOD_FAMILIES) for stratum in STRATA},
        "accepted_candidate_count": len(accepted),
        "input_candidate_count": len(rows),
        "available_by_period_family": available_by_family,
        "shortage": counts,
    }


def select_primary10(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_schedulable: bool = False,
) -> tuple[list[SelectedCandidate], list[dict[str, Any]], dict[str, Any]]:
    """Select exactly one candidate per stratum and period family."""

    validate_diagnostics_rows(rows)
    assert_selection_feature_guard()
    accepted = accepted_population(rows, require_schedulable=require_schedulable)
    if not accepted:
        report = _shortage_report(rows, accepted)
        raise SelectionShortageError(report)
    if len(accepted) >= 10:
        assert_structural_feature_variation(accepted)

    feature_vectors = build_percentile_features(accepted)
    selected: list[SelectedCandidate] = []
    audit: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    for stratum in STRATA:
        for family in PERIOD_FAMILIES:
            family_rows = [
                row
                for row in accepted
                if normalize_period_family(row.get("period_family")) == family
                and int(row["candidate_seed"]) not in used_seeds
                and _stratum_structural_gate(row, stratum)
            ]
            ranked: list[tuple[float, int, dict[str, Any], dict[str, float]]] = []
            for row in family_rows:
                seed = int(row["candidate_seed"])
                vector = feature_vectors[seed]
                distance = prototype_distance(vector, stratum)
                score = distance + _diversity_adjustment(vector, selected)
                ranked.append((score, seed, row, vector))
            ranked.sort(key=lambda item: (item[0], item[1]))
            if not ranked:
                report = _shortage_report(rows, accepted)
                report["missing_slot"] = {"stratum": stratum, "period_family": family}
                raise SelectionShortageError(report)
            score, seed, row, vector = ranked[0]
            distance = prototype_distance(vector, stratum)
            rank_within_family = 1 + sum(
                prototype_distance(feature_vectors[int(other["candidate_seed"])], stratum) < distance
                for other in family_rows
            )
            chosen = SelectedCandidate(
                row=dict(row),
                stratum=stratum,
                period_family=family,
                prototype_distance=distance,
                rank_within_stratum_family=rank_within_family,
                feature_vector=dict(vector),
            )
            selected.append(chosen)
            used_seeds.add(seed)
            audit.append(
                {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "candidate_seed": seed,
                    "stratum": stratum,
                    "period_family": family,
                    "selected": True,
                    "prototype_distance": distance,
                    "selection_score_with_diversity_penalty": score,
                    "selection_rank_within_stratum_family": rank_within_family,
                    "selection_feature_list": json.dumps(SELECTION_FEATURES, ensure_ascii=False),
                    "selection_forbidden_prefixes": json.dumps(FORBIDDEN_SELECTION_PREFIXES, ensure_ascii=False),
                }
            )

    if len(selected) != 10 or len({item.row["candidate_seed"] for item in selected}) != 10:
        report = _shortage_report(rows, accepted)
        report["reason"] = "duplicate_or_incomplete_primary10"
        raise SelectionShortageError(report)

    return selected, audit, {
        "schema_version": PRIMARY10_SCHEMA_VERSION,
        "selection_config_hash": selection_config_hash(),
        "selection_feature_list": list(SELECTION_FEATURES),
        "selection_forbidden_prefixes": list(FORBIDDEN_SELECTION_PREFIXES),
        "accepted_candidate_count": len(accepted),
    }


def primary_rows(selected: Sequence[SelectedCandidate], metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in selected:
        # Performance columns may remain in the diagnostics audit, but they
        # are intentionally not copied into the deployable primary10 file.
        row = {
            key: value
            for key, value in item.row.items()
            if not any(key.startswith(prefix) for prefix in FORBIDDEN_SELECTION_PREFIXES)
        }
        row.update(
            {
                "schema_version": PRIMARY10_SCHEMA_VERSION,
                "candidate_seed": int(item.row["candidate_seed"]),
                "stratum": item.stratum,
                "period_family": item.period_family,
                "prototype_distance": item.prototype_distance,
                "selection_rank_within_stratum_family": item.rank_within_stratum_family,
                "selection_config_hash": metadata["selection_config_hash"],
                "selection_feature_list": json.dumps(SELECTION_FEATURES, ensure_ascii=False),
                "selection_forbidden_prefixes": json.dumps(FORBIDDEN_SELECTION_PREFIXES, ensure_ascii=False),
            }
        )
        row.update(item.feature_vector)
        result.append(row)
    result.sort(key=lambda row: (str(row["stratum"]), str(row["period_family"]), int(row["candidate_seed"])))
    return result


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing to write empty selection artifact")
    preferred = [
        "schema_version",
        "candidate_seed",
        "stratum",
        "period_family",
        "prototype_distance",
        "selection_rank_within_stratum_family",
        *SELECTION_FEATURES,
        "total_util_lo_mode",
        "analysis_normalized_slack",
        "analysis_priority_order",
        "admission_method",
        "admission_priority_policy",
        "c_amc_sem_xf",
        "amc_rtb_normalized_slack",
        "baseline_lo_quality_qos",
        "valid_lo_increase_count_mean",
        "valid_lo_decrease_count_mean",
        "budget_competition_index",
        "selection_config_hash",
        "selection_feature_list",
        "selection_forbidden_prefixes",
    ]
    columns = list(dict.fromkeys(preferred + [key for row in rows for key in row]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-primary10", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--output-shortage-report", type=Path, required=True)
    parser.add_argument("--require-schedulable", action="store_true")
    return parser


def run_cli(args: argparse.Namespace) -> None:
    rows = read_csv(args.diagnostics)
    try:
        selected, audit, metadata = select_primary10(
            rows,
            require_schedulable=args.require_schedulable,
        )
    except SelectionShortageError as exc:
        write_json(args.output_shortage_report, exc.report)
        # No primary10 or cross-stratum fallback is written on shortage.
        raise
    write_rows(args.output_primary10, primary_rows(selected, metadata))
    write_rows(args.output_audit, audit)
    write_json(args.output_shortage_report, {"schema_version": "mc_stratified_dynamic_shortage_report_v1", "shortage": {}})


def main() -> None:
    try:
        run_cli(build_parser().parse_args())
    except SelectionShortageError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
