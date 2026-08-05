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
    if selection_semantics not in {
        "ranked_first_valid", "raw_top1", "top1_valid_else_noop",
        "all_invalid_force_top1",
    }:
        raise ValueError("UNSUPPORTED_POLICY_SELECTION_SEMANTICS")
    if selection_semantics == "ranked_first_valid":
        regions = selected_action_regions(guards, rankings, valid_reasons)
    else:
        regions = []
        for leaf_id in sorted(guards):
            ranking = list(rankings[leaf_id])
            raw = int(ranking[0])
            if selection_semantics == "raw_top1":
                regions.append({
                    "leaf_id": leaf_id, "rank_position": 0, "action_id": raw,
                    "leaf_guard": list(guards[leaf_id]), "preceding_invalid": [],
                    "predicate": {"leaf_guard": list(guards[leaf_id]), "unconditional_raw_top1": raw},
                    "implicit_noop_predicate": False,
                })
            elif selection_semantics == "top1_valid_else_noop":
                regions.extend([
                    {"leaf_id": leaf_id, "rank_position": 0, "action_id": raw,
                     "leaf_guard": list(guards[leaf_id]), "preceding_invalid": [],
                     "predicate": {"leaf_guard": list(guards[leaf_id]), "selected_action_valid": raw},
                     "implicit_noop_predicate": False},
                    {"leaf_id": leaf_id, "rank_position": len(ranking), "action_id": None,
                     "leaf_guard": list(guards[leaf_id]), "preceding_invalid": [raw],
                     "predicate": {"leaf_guard": list(guards[leaf_id]), "selected_action_invalid": raw},
                     "implicit_noop_predicate": True},
                ])
            else:
                for position, action_id in enumerate(ranking):
                    regions.append({
                        "leaf_id": leaf_id, "rank_position": position, "action_id": int(action_id),
                        "leaf_guard": list(guards[leaf_id]), "preceding_invalid": ranking[:position],
                        "predicate": {"leaf_guard": list(guards[leaf_id]),
                                      "preceding_actions_invalid": ranking[:position],
                                      "selected_action_valid": int(action_id)},
                        "implicit_noop_predicate": False,
                    })
                regions.append({
                    "leaf_id": leaf_id, "rank_position": 0, "action_id": raw,
                    "leaf_guard": list(guards[leaf_id]), "preceding_invalid": ranking,
                    "predicate": {"leaf_guard": list(guards[leaf_id]),
                                  "all_actions_invalid_force_raw_top1": ranking},
                    "implicit_noop_predicate": False,
                })
    return {
        "status": "PASS",
        "schema_version": "selected_action_regions_v2",
        "universal_over_policy_inputs": True,
        "regions": regions,
        "state_enumeration_used": False,
        "selection_semantics": selection_semantics,
    }
