from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .tables import write_campaign_csv_tables


def write_experiment_report(path: Path, result: Mapping[str, Any]) -> None:
    activation = result.get("activation") or {}
    proof = (result.get("semantic_recompile") or {}).get("proof_result") or {}
    integrity = (result.get("integrity_reuse") or {}).get("proof_result") or {}
    lines = [
        "# PPP Non-vacuity Experiment",
        "",
        "- artifact_class: `NONVACUITY_EXPERIMENT_ONLY`",
        "- deployment_certificate_eligible: `false`",
        f"- mutation_id: `{result.get('mutation_id')}`",
        f"- mutation_class: `{result.get('mutation_class')}`",
        f"- experiment_status: `{result.get('status')}`",
        f"- activation_status: `{activation.get('status', 'NOT_APPLICABLE')}`",
        f"- semantic_result: `{proof.get('result_status', 'NOT_RUN')}`",
        f"- semantic_first_obligation: `{proof.get('violated_obligation_id', 'NONE')}`",
        f"- integrity_result: `{integrity.get('result_status', 'NOT_RUN')}`",
        "",
        "This report is experiment-only and must not be aggregated as a deployed proof certificate.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_campaign_report(path: Path, campaign: Mapping[str, Any]) -> None:
    results = campaign.get("mutation_results", ())
    lines = [
        "# PPP Non-vacuity Campaign",
        "",
        "- artifact_class: `NONVACUITY_EXPERIMENT_ONLY`",
        "- deployment_certificate_eligible: `false`",
        f"- campaign_id: `{campaign.get('campaign_id')}`",
        f"- status: `{campaign.get('status')}`",
        "",
        "| Mutation | Class | Activation | Result | Proof status | First obligation |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        activation = result.get("activation") or {}
        proof = (result.get("semantic_recompile") or {}).get("proof_result") or (
            result.get("integrity_reuse") or {}
        ).get("proof_result") or {}
        lines.append(
            "| {mutation} | {kind} | {activation} | {status} | {proof} | {obligation} |".format(
                mutation=result.get("mutation_id", ""),
                kind=result.get("mutation_class", ""),
                activation=activation.get("status", "N/A"),
                status=result.get("status", ""),
                proof=proof.get("result_status", "NOT_RUN"),
                obligation=proof.get("violated_obligation_id", ""),
            )
        )
    summary = campaign.get("summary", {})
    lines.extend(
        [
            "",
            f"- kill_rate_numerator: `{summary.get('kill_rate_numerator')}`",
            f"- kill_rate_denominator: `{summary.get('kill_rate_denominator')}`",
            f"- kill_rate: `{summary.get('kill_rate')}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    write_campaign_csv_tables(path.parent, campaign)
