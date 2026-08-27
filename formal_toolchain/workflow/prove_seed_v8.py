"""V8 three-route orchestration without mixing route proof DAGs.

The mathematical theorem is a disjunction, while each proof bundle remains a
single resolved route.  This wrapper therefore runs isolated bundles in the
engineering order strict-full -> raw-prefix -> saturated-prefix and stops at
the first completely verified proof.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.routes.registry import resolve_registry
from formal_toolchain.workflow.prove_seed import prove_seed

V8_ROUTE_ORDER = ("strict_full", "raw_protected_prefix", "protected_prefix")
_ROUTE_LABEL = {
    "strict_full": "PROVED_BY_STRICT_FULL_ROUTE",
    "raw_protected_prefix": "PROVED_BY_RAW_PREFIX_ROUTE",
    "protected_prefix": "PROVED_BY_SATURATED_PREFIX_ROUTE",
}


def _read_summary(route_dir: Path) -> dict[str, Any]:
    path = route_dir / "verified" / "proof_summary.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _route_local_inconclusive(route_id: str, route_dir: Path) -> tuple[bool, list[str]]:
    """Return True only when shared proof layers are still PASS.

    FINITE_BAD_PREFIX_CONTRADICTION and FINAL_CLAIM_COMPOSITION are excluded
    because they necessarily inherit the selected terminal branch's failure.
    """

    summary = _read_summary(route_dir)
    statuses = summary.get("obligation_statuses")
    if not isinstance(statuses, dict):
        return False, ["VERIFIED_STATUS_MAP_MISSING"]
    resolved = resolve_registry(route_id)
    terminal_dependent_common = {"FINITE_BAD_PREFIX_CONTRADICTION", "FINAL_CLAIM_COMPOSITION"}
    blockers: list[str] = []
    for entry in resolved.common_entries:
        oid = str(entry["id"])
        if oid in terminal_dependent_common:
            continue
        if statuses.get(oid) != "PASS":
            blockers.append(oid)
    return not blockers, blockers


def prove_seed_v8(*, seed_dir: Path, tree_variant: str, code_root: Path, out: Path,
                  target_recipe: Path | None = None, overwrite: bool = False,
                  refresh_phase_k_map: bool = False,
                  dependency_manifest_override: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """Run the three sound V8 terminal routes in isolated workspaces."""

    out = Path(out).resolve()
    if out.exists():
        if not overwrite:
            return 2, {
                "workflow_status": "FAILED", "result_status": "PROOF_BUNDLE_INVALID",
                "failure_code": "OUTPUT_EXISTS", "proof_route": "v8_auto",
            }
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=False)

    attempts: list[dict[str, Any]] = []
    selected: str | None = None
    terminal_result: dict[str, Any] | None = None
    terminal_code = 20

    for index, route_id in enumerate(V8_ROUTE_ORDER):
        route_dir = out / route_id
        code, result = prove_seed(
            seed_dir=seed_dir, tree_variant=tree_variant, code_root=code_root,
            out=route_dir, target_recipe=target_recipe, overwrite=False,
            refresh_phase_k_map=refresh_phase_k_map,
            proof_route=route_id,
            dependency_manifest_override=dependency_manifest_override,
        )
        attempt = {
            "route_id": route_id,
            "engineering_order": index + 1,
            "exit_code": code,
            "result_status": result.get("result_status"),
            "failure_route": result.get("failure_route"),
            "failure_code": result.get("failure_code"),
            "violated_obligation_id": result.get("violated_obligation_id"),
            "bundle_dir": route_id,
        }
        attempts.append(attempt)
        if result.get("result_status") == "DEPLOYED_TREE_PROVED":
            selected = route_id
            terminal_result = result
            terminal_code = 0
            break

        route_local, common_blockers = _route_local_inconclusive(route_id, route_dir)
        attempt["route_local_inconclusive"] = route_local
        attempt["shared_blockers"] = common_blockers
        if not route_local:
            terminal_result = result
            terminal_code = code
            break

    if selected is not None and terminal_result is not None:
        aggregate = {
            "workflow_schema_version": "prove_seed_v8_three_route_v1",
            "workflow_status": "COMPLETED",
            "result_status": "DEPLOYED_TREE_PROVED",
            "proof_route": "v8_auto",
            "selected_terminal_route": selected,
            "terminal_certificate_kind": _ROUTE_LABEL[selected],
            "primary_claim": "DEPLOYED_HI_SAFETY",
            "attempts": attempts,
            "selected_result": terminal_result,
            "exit_code": 0,
        }
    elif terminal_result is not None and attempts and attempts[-1].get("route_local_inconclusive") is False:
        aggregate = {
            "workflow_schema_version": "prove_seed_v8_three_route_v1",
            "workflow_status": "FAILED",
            "result_status": terminal_result.get("result_status", "UNRESOLVED"),
            "proof_route": "v8_auto",
            "selected_terminal_route": None,
            "primary_claim": "DEPLOYED_HI_SAFETY",
            "failure_route": terminal_result.get("failure_route"),
            "failure_code": terminal_result.get("failure_code"),
            "violated_obligation_id": terminal_result.get("violated_obligation_id"),
            "shared_blockers": attempts[-1].get("shared_blockers", []),
            "attempts": attempts,
            "exit_code": terminal_code,
        }
    else:
        aggregate = {
            "workflow_schema_version": "prove_seed_v8_three_route_v1",
            "workflow_status": "COMPLETED",
            "result_status": "UNRESOLVED",
            "proof_route": "v8_auto",
            "selected_terminal_route": None,
            "primary_claim": "DEPLOYED_HI_SAFETY",
            "failure_route": "UNRESOLVED",
            "failure_code": "UNPROVED_BY_V8_THREE_ROUTE_SUFFICIENT_TESTS",
            "attempts": attempts,
            "exit_code": 20,
        }
        terminal_code = 20

    (out / "v8_three_route_result.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# V8 three-route formal proof report", "",
        f"- result_status: `{aggregate['result_status']}`",
        f"- selected_terminal_route: `{aggregate.get('selected_terminal_route')}`",
        f"- failure_code: `{aggregate.get('failure_code')}`", "", "## Attempts", "",
    ]
    for row in attempts:
        lines.append(
            f"- `{row['route_id']}`: `{row['result_status']}`"
            + (f" / `{row.get('failure_code')}`" if row.get('failure_code') else "")
        )
    (out / "human_readable_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(aggregate["exit_code"]), aggregate


__all__ = ["V8_ROUTE_ORDER", "prove_seed_v8"]
