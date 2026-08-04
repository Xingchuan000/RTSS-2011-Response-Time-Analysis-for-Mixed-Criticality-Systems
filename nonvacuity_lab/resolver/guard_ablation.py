from __future__ import annotations

from collections import Counter
from typing import Any


def resolve_guard_ablation(leaf_rows: list[dict[str, Any]], catalog: dict[str, Any]) -> dict[str, Any]:
    histogram: Counter[str] = Counter()
    for row in leaf_rows:
        if int(row.get("hout_hit_count", 0)) > 0:
            histogram.update({str(k): int(v) for k, v in row.get("reject_reason_histogram", {}).items()})
    for reason, count in histogram.most_common():
        guard = catalog.get("guards", {}).get(reason)
        patches = guard.get("patches", []) if guard else []
        if guard and any(patch.get("role") == "DEPLOYED_IMPLEMENTATION" for patch in patches):
            return {"reject_reason": reason, "hit_count": count, "guard_id": guard["guard_id"], "patch_templates": patches}
    raise ValueError("B4_BLOCKING_GUARD_UNRESOLVED")
