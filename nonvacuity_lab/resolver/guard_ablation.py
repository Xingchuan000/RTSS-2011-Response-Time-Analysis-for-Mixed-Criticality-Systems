from __future__ import annotations

from typing import Any


def _matching_guard(reason: str, catalog: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for key, guard in catalog.get("guards", {}).items():
        prefixes = guard.get("reject_reason_prefixes")
        if not prefixes:
            one = guard.get("reject_reason_prefix")
            prefixes = [one] if one else [key]
        if any(str(reason).startswith(str(prefix)) for prefix in prefixes):
            return str(key), dict(guard)
    return None


def _as_count(mapping: Any, key: Any) -> int:
    if not isinstance(mapping, dict):
        return 0
    try:
        return int(mapping.get(str(key), mapping.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def resolve_guard_ablation(
    leaf_rows: list[dict[str, Any]],
    catalog: dict[str, Any],
    *,
    require_raw_top1: bool = True,
) -> dict[str, Any]:
    """Resolve B4 to an observed *raw top-1* rejection.

    Removing a guard does not reorder the tree.  Therefore the only action that
    can become selected after the mutation is the original leaf ranking's first
    candidate.  Older resolver code aggregated reject reasons independently of
    rejected actions and could bind B4 to a lower-ranked action that the runtime
    would never execute.  This resolver intentionally accepts only rows where
    the raw top-1 action was observed rejected in HOUT.
    """

    candidates: list[tuple[int, int, int, str, dict[str, Any], str, dict[str, Any]]] = []
    for row in leaf_rows:
        hit_count = int(row.get("hout_hit_count", 0))
        if hit_count <= 0:
            continue
        ranking = row.get("action_ranking", ())
        if not isinstance(ranking, (list, tuple)) or not ranking:
            continue
        raw_action = int(ranking[0])
        raw_reject_count = _as_count(row.get("rejected_action_histogram", {}), raw_action)
        if raw_reject_count <= 0:
            continue
        reason_hist = row.get("reject_reason_histogram", {})
        if not isinstance(reason_hist, dict):
            continue
        for reason, raw_reason_count in reason_hist.items():
            reason_count = int(raw_reason_count)
            if reason_count <= 0:
                continue
            match = _matching_guard(str(reason), catalog)
            if match is None:
                continue
            key, guard = match
            # The audit schema stores exact raw-top1 reject reasons.  We still
            # cap the evidence count by the observed raw-action count so older
            # aggregate audit files cannot overstate activation.
            evidence_count = min(reason_count, raw_reject_count)
            candidates.append(
                (
                    evidence_count,
                    hit_count,
                    -int(row.get("leaf_id", -1)),
                    str(reason),
                    row,
                    key,
                    guard,
                )
            )

    if not candidates and not require_raw_top1:
        # Legacy v1 audit fixtures did not guarantee that action_ranking and
        # rejected_action_histogram came from the same executable tree.  Keep
        # them reloadable, while v2/paper campaigns use the strict default.
        for row in leaf_rows:
            if int(row.get("hout_hit_count", 0)) <= 0:
                continue
            rejected = row.get("rejected_action_histogram", {})
            if not isinstance(rejected, dict) or not rejected:
                continue
            action_id = int(max(rejected, key=lambda item: int(rejected[item])))
            for reason, raw_count in row.get("reject_reason_histogram", {}).items():
                match = _matching_guard(str(reason), catalog)
                if match is None:
                    continue
                key, guard = match
                candidates.append((int(raw_count), int(row.get("hout_hit_count", 0)), -int(row.get("leaf_id", -1)), str(reason), row, key, guard))
                row = {**row, "_legacy_action_id": action_id}
                candidates[-1] = (*candidates[-1][:4], row, key, guard)

    if not candidates:
        raise ValueError("B4_BLOCKING_GUARD_UNRESOLVED")

    _, _, _, reason, row, key, guard = max(candidates, key=lambda item: item[:3])
    raw_action = int(row.get("_legacy_action_id", row["action_ranking"][0]))
    evidence_count = min(
        int(row.get("reject_reason_histogram", {}).get(reason, 0)),
        _as_count(row.get("rejected_action_histogram", {}), raw_action),
    )
    return {
        "reject_reason": reason,
        "hit_count": evidence_count,
        "guard_id": guard["guard_id"],
        "catalog_key": guard.get("catalog_key", key),
        "leaf_id": row.get("leaf_id"),
        "action_id": raw_action,
        "raw_top1_action_id": raw_action,
        "disabled_guard_constraint": dict(guard.get("symbolic_constraint", {"constant": True})),
        "activation_witness_ref": row.get("activation_witness_ref", row.get("witness_ref")),
    }
