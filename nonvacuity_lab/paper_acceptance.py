from __future__ import annotations

from dataclasses import asdict, dataclass
import re


REQUIRED_BASELINES = {"P0", "P1", "P2", "P3"}
REQUIRED_MAIN_MUTATIONS = {
    "A1", "A2", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "D1",
    "E1", "E2", "E3", "E4", "E5", "E6", "F1", "F2", "F3", "F4", "F5", "F6", "F7",
}
_CANONICAL_ID = re.compile(r"^(P[0-3]|A[12]|B[1-5]|C[1-3]|D1|E[1-6]|F[1-7])(?:_|$)")


@dataclass(frozen=True)
class AcceptanceFinding:
    finding_id: str
    passed: bool
    detail: str


def _canonicalize_mutations(campaign_report: dict) -> tuple[dict[str, dict], list[str]]:
    raw = campaign_report.get("mutations", campaign_report.get("mutation_results", {}))
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw or ())
    canonical: dict[str, dict] = {}
    duplicates: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mutation_id = str(row.get("mutation_id", ""))
        match = _CANONICAL_ID.match(mutation_id)
        key = match.group(1) if match else mutation_id
        if key in canonical:
            duplicates.append(key)
            continue
        canonical[key] = row
    return canonical, duplicates


def _proof(row: dict) -> dict:
    candidates = (
        row.get("formal_result"),
        row.get("semantic_recompile", {}).get("proof_result") if isinstance(row.get("semantic_recompile"), dict) else None,
        row.get("baseline", {}).get("proof_result") if isinstance(row.get("baseline"), dict) else None,
        row.get("integrity_reuse", {}).get("proof_result") if isinstance(row.get("integrity_reuse"), dict) else None,
        row,
    )
    return next((item for item in candidates if isinstance(item, dict) and item.get("result_status")), {})


def evaluate_paper_acceptance(campaign_report: dict) -> dict:
    mutations, duplicates = _canonicalize_mutations(campaign_report)
    findings = [
        AcceptanceFinding(
            "canonical_ids_unique",
            not duplicates,
            "duplicate canonical ids: " + ", ".join(sorted(set(duplicates))) if duplicates else "canonical ids are unique",
        )
    ]
    required = REQUIRED_BASELINES | REQUIRED_MAIN_MUTATIONS
    for mutation_id in sorted(required):
        findings.append(AcceptanceFinding(f"present:{mutation_id}", mutation_id in mutations, "present" if mutation_id in mutations else "missing"))
    for mutation_id in sorted(REQUIRED_BASELINES):
        status = _proof(mutations.get(mutation_id, {})).get("result_status")
        findings.append(AcceptanceFinding(f"baseline_proved:{mutation_id}", status == "DEPLOYED_TREE_PROVED", "positive control must remain proved"))
    for mutation_id in ("A1", "A2"):
        row = mutations.get(mutation_id, {})
        activation = row.get("activation", {}) if isinstance(row, dict) else {}
        findings.append(AcceptanceFinding(
            f"benign_accepted:{mutation_id}",
            _proof(row).get("result_status") == "DEPLOYED_TREE_PROVED"
            and (
                activation.get("status") == "ACTIVATED"
                or activation.get("activated") is True
            ),
            "mask-contained challenge must activate and remain proved",
        ))
    for mutation_id in ("F1", "F2", "F3"):
        findings.append(AcceptanceFinding(
            f"integrity_rejected:{mutation_id}",
            _proof(mutations.get(mutation_id, {})).get("result_status") == "PROOF_BUNDLE_INVALID",
            "old bundle reuse must be rejected",
        ))
    d1 = mutations.get("D1", {})
    gradient = d1.get("gradient", {}) if isinstance(d1, dict) else {}
    delta_star = d1.get("delta_star") if isinstance(d1, dict) else None
    if delta_star is None and isinstance(gradient, dict):
        delta_star = gradient.get("delta_star")
    findings.append(AcceptanceFinding("gradient_delta_star", delta_star is not None, "D1 must find a first reference failure point"))
    return {"passed": all(item.passed for item in findings), "findings": [asdict(item) for item in findings]}
