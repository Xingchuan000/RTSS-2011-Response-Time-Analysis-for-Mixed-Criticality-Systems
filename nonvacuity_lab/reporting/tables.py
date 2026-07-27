from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping


def write_campaign_csv_tables(directory: Path, campaign: Mapping[str, Any]) -> None:
    results = list(campaign.get("mutation_results", ()))
    directory.mkdir(parents=True, exist_ok=True)
    _write_rows(
        directory / "verifier_nonvacuity.csv",
        (
            {
                "mutation_id": item.get("mutation_id"),
                "mutation_class": item.get("mutation_class"),
                "experiment_status": item.get("status"),
                "activation_status": (item.get("activation") or {}).get("status"),
                "proof_status": _proof(item).get("result_status"),
                "failure_route": _proof(item).get("failure_route"),
                "first_obligation": _proof(item).get("violated_obligation_id"),
            }
            for item in results
        ),
    )
    _write_rows(
        directory / "mutation_activation_summary.csv",
        (
            {
                "mutation_id": item.get("mutation_id"),
                "status": (item.get("activation") or {}).get("status"),
                "leaf_id": (item.get("activation") or {}).get("leaf_id"),
                "action_id": (item.get("activation") or {}).get("action_id"),
                "hout_hit_count": (item.get("activation") or {}).get("hout_hit_count"),
                "baseline_reject_count": (item.get("activation") or {}).get(
                    "baseline_reject_count"
                ),
                "selected_after_mutation_count": (item.get("activation") or {}).get(
                    "selected_after_mutation_count"
                ),
                "all_invalid_count": (item.get("activation") or {}).get(
                    "all_invalid_count"
                ),
            }
            for item in results
        ),
    )
    _write_rows(
        directory / "mask_challenge.csv",
        (
            {
                "mutation_id": item.get("mutation_id"),
                "leaf_id": (item.get("activation") or {}).get("leaf_id"),
                "action_id": (item.get("activation") or {}).get("action_id"),
                "baseline_reject_count": (item.get("activation") or {}).get(
                    "baseline_reject_count"
                ),
                "post_invariant_violation": (item.get("activation") or {}).get(
                    "post_invariant_violation"
                ),
                "proof_status": _proof(item).get("result_status"),
            }
            for item in results
            if item.get("mutation_class") in {"DANGEROUS_TOP1", "MASK_BYPASS"}
        ),
    )
    _write_rows(
        directory / "mask_fallback_ablation.csv",
        (
            {
                "mutation_id": item.get("mutation_id"),
                "mutation_class": item.get("mutation_class"),
                "activation_status": (item.get("activation") or {}).get("status"),
                "selected_after_mutation_count": (item.get("activation") or {}).get(
                    "selected_after_mutation_count"
                ),
                "all_invalid_count": (item.get("activation") or {}).get(
                    "all_invalid_count"
                ),
                "experiment_status": item.get("status"),
            }
            for item in results
            if item.get("mutation_class")
            in {
                "MASK_BYPASS",
                "NO_FIRST_VALID",
                "ALL_INVALID_FORCE_TOP1",
                "GUARD_ABLATION",
            }
        ),
    )
    _write_rows(
        directory / "envelope_safety_margin.csv",
        (
            {
                "mutation_id": item.get("mutation_id"),
                "delta": (
                    (item.get("semantic_recompile") or {}).get("mutation_result")
                    or {}
                ).get("details", {}).get("delta"),
                "proof_status": _proof(item).get("result_status"),
                "first_obligation": _proof(item).get("violated_obligation_id"),
                "experiment_status": item.get("status"),
            }
            for item in results
            if item.get("mutation_class") == "ENVELOPE"
        ),
    )
    _write_rows(
        directory / "bundle_binding.csv",
        (
            {
                "mutation_id": item.get("mutation_id"),
                "experiment_status": item.get("status"),
                "recompiled": (item.get("integrity_reuse") or {}).get("recompiled"),
                "result_status": (
                    (item.get("integrity_reuse") or {}).get("proof_result") or {}
                ).get("result_status"),
                "failure_code": (
                    (item.get("integrity_reuse") or {}).get("proof_result") or {}
                ).get("failure_code"),
            }
            for item in results
            if item.get("integrity_reuse")
        ),
    )
    _write_rows(
        directory / "kill_rate_groups.csv",
        (
            {"status": status, "count": count}
            for status, count in dict(campaign.get("summary", {}).get("counts", {})).items()
        ),
    )


def _proof(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return (item.get("semantic_recompile") or {}).get("proof_result") or (
        item.get("integrity_reuse") or {}
    ).get("proof_result") or {}


def _write_rows(path: Path, rows) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
