"""生成 ranked first-valid 的 selected region 描述。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def selected_action_regions(guards: dict[int, Sequence[dict[str, Any]]], rankings: dict[int, Sequence[int]],
                            valid_reasons: dict[int, Sequence[str]] | None = None) -> list[dict[str, Any]]:
    regions = []
    for leaf_id in sorted(guards):
        ranking = list(rankings[leaf_id])
        for position, action_id in enumerate(ranking):
            regions.append({"leaf_id": leaf_id, "rank_position": position,
                            "action_id": int(action_id), "leaf_guard": list(guards[leaf_id]),
                            "preceding_invalid": ranking[:position],
                            "valid_reason": (valid_reasons or {}).get(leaf_id, ()),
                            "predicate": {"leaf_guard": list(guards[leaf_id]),
                                          "preceding_actions_invalid": ranking[:position],
                                          "selected_action_valid": int(action_id)},
                            "implicit_noop_predicate": False})
        regions.append({"leaf_id": leaf_id, "rank_position": len(ranking), "action_id": None,
                        "leaf_guard": list(guards[leaf_id]), "preceding_invalid": ranking,
                        "valid_reason": (valid_reasons or {}).get(leaf_id, ()),
                        "predicate": {"leaf_guard": list(guards[leaf_id]),
                                      "all_actions_invalid": list(ranking)},
                        "implicit_noop_predicate": True})
    return regions


def selected_action_regions_v2(
    guards: dict[int, Sequence[dict[str, Any]]],
    rankings: dict[int, Sequence[int]],
    valid_reasons: dict[int, Sequence[str]] | None = None,
    *, selection_semantics: str = "ranked_first_valid",
) -> dict[str, Any]:
    if selection_semantics == "raw_top1":
        regions = [{"leaf_id": leaf_id, "rank_position": 0, "action_id": int(rankings[leaf_id][0]),
                    "leaf_guard": list(guards[leaf_id]),
                    "predicate": {"leaf_guard": list(guards[leaf_id]),
                                  "raw_top1": int(rankings[leaf_id][0])},
                    "implicit_noop_predicate": False}
                   for leaf_id in sorted(guards)]
    elif selection_semantics == "top1_or_noop":
        regions = []
        for leaf_id in sorted(guards):
            top1 = int(rankings[leaf_id][0])
            regions.extend([
                {"leaf_id": leaf_id, "rank_position": 0, "action_id": top1,
                 "leaf_guard": list(guards[leaf_id]),
                 "predicate": {"leaf_guard": list(guards[leaf_id]), "top1_valid": top1},
                 "implicit_noop_predicate": False},
                {"leaf_id": leaf_id, "rank_position": 1, "action_id": None,
                 "leaf_guard": list(guards[leaf_id]),
                 "predicate": {"leaf_guard": list(guards[leaf_id]), "top1_invalid": top1},
                 "implicit_noop_predicate": True},
            ])
    else:
        regions = selected_action_regions(guards, rankings, valid_reasons)
        if selection_semantics == "first_valid_else_top1":
            for row in regions:
                if row.get("implicit_noop_predicate") is True:
                    row["action_id"] = int(rankings[int(row["leaf_id"])][0])
                    row["implicit_noop_predicate"] = False
                    row["forced_top1_on_all_invalid"] = True
    return {
        "status": "PASS",
        "schema_version": "selected_action_regions_v2",
        "universal_over_policy_inputs": True,
        "regions": regions,
        "state_enumeration_used": False,
        "selection_semantics": selection_semantics,
    }
