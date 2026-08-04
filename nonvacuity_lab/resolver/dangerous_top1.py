from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


RISK_ORDER = {
    "HI_BUDGET_DECREASE": 4,
    "HIGHER_PRIORITY_LO_INCREASE": 3,
    "BUDGET_INCREASE": 2,
    "BENIGN_OR_UNKNOWN": 0,
}


@dataclass(frozen=True)
class DangerousTop1Resolution:
    leaf_id: int
    action_id: int
    tree_hash: str
    activation_source: str
    witness_ref: str
    reason: str


def resolve_dangerous_top1(
    rows: Iterable[Mapping[str, Any]],
    *,
    require_hout_hit: bool = True,
) -> DangerousTop1Resolution:
    """Select a dangerous action with action-specific activation evidence.

    A leaf-level raw-top1 rejection is not sufficient for an arbitrary action:
    the selected action itself must have an observed reject count, or the row
    must name an action-specific symbolic SAT witness.
    """

    candidates = []
    for leaf in rows:
        if require_hout_hit and int(leaf.get("hout_hit_count", 0)) <= 0:
            continue
        ranking = [int(item) for item in leaf.get("action_ranking", ())]
        for risk in leaf.get("action_risks", ()):
            risk_class = str(risk.get("risk_class", "BENIGN_OR_UNKNOWN"))
            score = RISK_ORDER.get(risk_class, 0)
            action_id = int(risk["action_id"])
            if score <= 0 or (ranking and ranking[0] == action_id):
                continue
            observed_reject = int(risk.get("observed_reject_count", 0))
            symbolic_action = leaf.get("symbolic_action_id")
            symbolic_sat = (
                str(leaf.get("selected_region_status", "UNKNOWN")) == "SAT"
                and symbolic_action is not None
                and int(symbolic_action) == action_id
                and bool(leaf.get("symbolic_witness_ref"))
            )
            if observed_reject <= 0 and not symbolic_sat:
                continue
            score_tuple = (
                score,
                int(observed_reject > 0),
                int(symbolic_sat),
                int(leaf.get("hout_hit_count", 0)),
                int(leaf.get("fallback_count", 0)),
                observed_reject,
            )
            candidates.append((score_tuple, leaf, risk))
    if not candidates:
        raise ValueError("DANGEROUS_TOP1_TARGET_UNRESOLVED")
    _, leaf, risk = max(candidates, key=lambda item: item[0])
    observed_reject = int(risk.get("observed_reject_count", 0))
    return DangerousTop1Resolution(
        leaf_id=int(leaf["leaf_id"]),
        action_id=int(risk["action_id"]),
        tree_hash=str(leaf["tree_hash"]),
        activation_source="HOUT_ACTION_REJECT" if observed_reject > 0 else "SYMBOLIC_SAT",
        witness_ref=(
            f"hout:seed={leaf.get('seed')};variant={leaf.get('tree_variant')};"
            f"leaf={leaf.get('leaf_id')};action={risk.get('action_id')};count={observed_reject}"
            if observed_reject > 0
            else str(leaf["symbolic_witness_ref"])
        ),
        reason=str(risk.get("risk_class")),
    )
