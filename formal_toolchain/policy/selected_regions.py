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
) -> dict[str, Any]:
    regions = selected_action_regions(guards, rankings, valid_reasons)
    return {
        "status": "PASS",
        "schema_version": "selected_action_regions_v2",
        "universal_over_policy_inputs": True,
        "regions": regions,
        "state_enumeration_used": False,
    }
