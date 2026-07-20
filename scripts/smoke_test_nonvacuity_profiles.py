"""Smoke test for the unified non-vacuity experiment profiles.

This script is intentionally lightweight: it checks default-off fail-closed
resolution, all profile mappings, the source bindings used by Phase K, and
optionally freezes/preflights one real seed for ``off`` and ``b1_mask_bypass``.
It does not run the full RTA/bridge proof.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from amc_py.nonvacuity import (
    SUPPORTED_PROFILES,
    formal_expected_failure,
    resolve_nonvacuity_settings,
)
from formal_toolchain.binding.controller_binding import bind_controller_runtime
from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
from formal_toolchain.binding.recovery_binding import bind_recovery_runtime
from formal_toolchain.binding.removal_binding import bind_removal_runtime
from formal_toolchain.core.formal_checks import calculate_raw_evidence
from formal_toolchain.workflow.seed_workspace import freeze_seed_workspace


PROFILE_PARAMS = {
    "b4_disable_guard": {"disabled_guards": ["hi_decrease"]},
    "c1_action_step": {"action_ratio": 0.05},
    "c2_min_increment_2": {"min_budget_delta": 2},
    "e4_controller_overhead": {"controller_overhead_ticks": 1},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed-dir", type=Path)
    parser.add_argument("--tree-variant", default="best_overall")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    result: dict[str, object] = {
        "schema_version": "nonvacuity_unified_smoke_v1",
        "code_root": str(code_root),
        "profiles": {},
        "bindings": {},
        "real_seed_preflight": {},
    }

    # Default profile must reject hidden mutation parameters.
    off = resolve_nonvacuity_settings("off")
    assert off.enabled is False
    assert off.selection_semantics == "ranked_first_valid"
    assert off.step_guard_semantics == "checked"
    try:
        resolve_nonvacuity_settings("off", {"action_ratio": 0.05})
    except ValueError as exc:
        assert str(exc) == "NONVACUITY_PARAMS_REQUIRE_NON_OFF_PROFILE"
    else:
        raise AssertionError("off profile accepted mutation parameters")

    for name in SUPPORTED_PROFILES:
        settings = resolve_nonvacuity_settings(name, PROFILE_PARAMS.get(name))
        result["profiles"][name] = {
            **settings.to_dict(),
            "expected_first_rejection": formal_expected_failure(settings),
        }

    binding_functions = {
        "event_runtime": bind_event_runtime,
        "removal_runtime": bind_removal_runtime,
        "controller_runtime": bind_controller_runtime,
        "recovery_runtime": bind_recovery_runtime,
    }
    for name, function in binding_functions.items():
        evidence = function(code_root)
        result["bindings"][name] = evidence
        if evidence.get("status") != "PASS":
            raise AssertionError(f"{name} binding failed: {evidence}")

    if args.seed_dir is not None:
        for profile in ("off", "b1_mask_bypass"):
            with tempfile.TemporaryDirectory(prefix=f"nonvacuity_{profile}_") as temp:
                frozen = freeze_seed_workspace(
                    args.seed_dir,
                    args.tree_variant,
                    Path(temp) / "workspace",
                    code_root=code_root,
                    overwrite=True,
                    nonvacuity_profile=profile,
                    refresh_phase_k_map=True,
                )
                raw = calculate_raw_evidence(
                    Path(frozen["request"]), source_root=code_root, include_reference=False
                )
                result["real_seed_preflight"][profile] = {
                    "status": raw["evidence"]["PREFLIGHT"].get(
                        "obligation_status", raw["evidence"]["PREFLIGHT"].get("status")
                    ),
                    "context_hash": raw["context_hash"],
                }

    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
