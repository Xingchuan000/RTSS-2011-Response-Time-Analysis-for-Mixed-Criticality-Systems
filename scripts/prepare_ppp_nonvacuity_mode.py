from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Support direct execution after copying to <repo>/scripts/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nonvacuity_lab.audit.tree_reader import iter_leaves, load_tree
from nonvacuity_lab.canonical import (
    IGNORED_TREE_PARTS,
    file_hash,
    tree_hash,
)
from nonvacuity_lab.coherence import missing_roles
from nonvacuity_lab.config_io import (
    validate_config_kind,
    verify_config_hash,
    write_resolved_campaign,
)
from nonvacuity_lab.doctor.runner import run_doctor
from nonvacuity_lab.preflight import audit_v2_campaign_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare, reseal and fully doctor one PPP non-vacuity mode."
    )
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--mode-config", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--mutation", action="append", default=[], required=True)
    p.add_argument("--preflight-output", type=Path, required=True)
    p.add_argument("--doctor-output", type=Path, required=True)
    p.add_argument("--readiness-output", type=Path, required=True)
    return p.parse_args()


def record(checks: list[dict[str, Any]], check_id: str, ok: bool, summary: str, **details: Any) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "PASS" if ok else "FAIL",
            "summary": summary,
            "details": details,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def mutation_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("mutation_id")): item
        for item in config.get("mutations", [])
        if isinstance(item, dict)
    }


def validate_dangerous_target(
    checks: list[dict[str, Any]],
    mutation: dict[str, Any],
    *,
    label: str,
) -> None:
    target = mutation.get("resolved_target")
    required = {
        "seed_dir",
        "tree_path",
        "tree_sha256",
        "leaf_id",
        "action_id",
        "original_ranking",
    }
    ok_target = isinstance(target, dict) and required.issubset(target)
    record(
        checks,
        f"{label}_target_fields",
        ok_target,
        f"{label} target has all required fields",
        missing=sorted(required - set(target or {})),
    )
    if not ok_target:
        return

    tree_path = Path(str(target["tree_path"])).resolve()
    seed_dir = Path(str(target["seed_dir"])).resolve()
    record(checks, f"{label}_seed_dir", seed_dir.is_dir(), f"{label} seed directory exists", path=str(seed_dir))
    record(checks, f"{label}_tree_file", tree_path.is_file(), f"{label} tree exists", path=str(tree_path))
    if not tree_path.is_file():
        return

    actual_hash = file_hash(tree_path)
    record(
        checks,
        f"{label}_tree_hash",
        actual_hash == str(target["tree_sha256"]),
        f"{label} tree hash matches resolved target",
        expected=str(target["tree_sha256"]),
        actual=actual_hash,
    )

    tree = load_tree(tree_path)
    leaf_id = int(target["leaf_id"])
    action_id = int(target["action_id"])
    leaf = next(
        (
            row
            for row in iter_leaves(tree)
            if int(row.get("node_id", row.get("leaf_id", row.get("id", -1)))) == leaf_id
        ),
        None,
    )
    record(checks, f"{label}_leaf", leaf is not None, f"{label} leaf exists", leaf_id=leaf_id)
    if leaf is None:
        return
    ranking = [int(x) for x in leaf.get("action_ranking", [])]
    original = [int(x) for x in target.get("original_ranking", [])]
    record(
        checks,
        f"{label}_ranking_matches",
        ranking == original,
        f"{label} original ranking matches tree",
        tree_ranking=ranking,
        target_ranking=original,
    )
    record(
        checks,
        f"{label}_action_in_ranking",
        action_id in ranking,
        f"{label} action exists in leaf ranking",
        action_id=action_id,
    )
    record(
        checks,
        f"{label}_action_not_top1",
        bool(ranking) and action_id != ranking[0],
        f"{label} dangerous action is not already raw top-1",
        action_id=action_id,
        current_top1=ranking[0] if ranking else None,
    )


def validate_hout_profile(
    checks: list[dict[str, Any]],
    config: dict[str, Any],
    profile_id: str,
) -> None:
    profile = config.get("hout_profiles", {}).get(profile_id)
    record(checks, f"hout_{profile_id}_present", isinstance(profile, dict), f"HOUT profile {profile_id} exists")
    if not isinstance(profile, dict):
        return
    for key in ("taskset_path", "runtime_config_path"):
        path = Path(str(profile.get(key, ""))).resolve()
        record(checks, f"hout_{profile_id}_{key}", path.is_file(), f"HOUT {key} exists", path=str(path))
    scenarios = profile.get("scenario_seeds")
    valid_scenarios = isinstance(scenarios, list) and len(scenarios) == 50 and [int(x) for x in scenarios] == list(range(101550, 101600))
    record(
        checks,
        f"hout_{profile_id}_scenarios",
        valid_scenarios,
        "HOUT uses exactly actual scenarios 101550-101599",
        count=len(scenarios) if isinstance(scenarios, list) else None,
        first=scenarios[0] if isinstance(scenarios, list) and scenarios else None,
        last=scenarios[-1] if isinstance(scenarios, list) and scenarios else None,
    )
    runtime_path = Path(str(profile.get("runtime_config_path", ""))).resolve()
    if runtime_path.is_file():
        runtime = load_json(runtime_path)
        record(
            checks,
            f"hout_{profile_id}_offset",
            int(runtime.get("scenario_seed_offset", 0)) == 0,
            "Runtime replay does not apply the 100000 offset twice",
            scenario_seed_offset=runtime.get("scenario_seed_offset", 0),
        )


def validate_pair(
    checks: list[dict[str, Any]],
    producer: dict[str, Any],
    consumer: dict[str, Any],
) -> None:
    p = producer.get("resolved_target", {})
    c = consumer.get("resolved_target", {})
    keys = ("tree_path", "tree_sha256", "leaf_id", "action_id", "original_ranking")
    differences = {key: {"producer": p.get(key), "consumer": c.get(key)} for key in keys if p.get(key) != c.get(key)}
    record(
        checks,
        "a1_b1_same_tree_target",
        not differences,
        "B1 reuses the exact A1 tree/leaf/action target",
        differences=differences,
    )

    parameters = consumer.get("mutator", {}).get("parameters", {})
    patches = parameters.get("patches") if isinstance(parameters, dict) else None
    record(
        checks,
        "b1_patch_set",
        isinstance(patches, list) and bool(patches),
        "B1 coherent source patch set is present",
        patch_count=len(patches) if isinstance(patches, list) else 0,
    )
    if isinstance(patches, list):
        roles = {str(item.get("role")) for item in patches if isinstance(item, dict)}
        missing = list(missing_roles(parameters.get("semantic_change_id"), roles))
        record(
            checks,
            "b1_patch_roles",
            not missing,
            "B1 patch set covers deployed and frozen selection/apply roles",
            roles=sorted(roles),
            missing=missing,
        )


def main() -> int:
    args = parse_args()
    checks: list[dict[str, Any]] = []
    try:
        base_path = args.base_config.resolve()
        mode_path = args.mode_config.resolve()
        project_root = args.project_root.resolve()
        selected_ids = [str(x) for x in args.mutation]

        record(checks, "base_config_exists", base_path.is_file(), "Base resolved config exists", path=str(base_path))
        if not base_path.is_file():
            raise FileNotFoundError(base_path)

        base = load_json(base_path)
        try:
            validate_config_kind(base)
            verify_config_hash(base)
            valid_base = True
            base_error = None
        except Exception as exc:
            valid_base = False
            base_error = str(exc)
        record(checks, "base_config_sealed", valid_base, "Base config is a sealed RESOLVED campaign", error=base_error)
        if not valid_base:
            raise ValueError(base_error)

        ignore_ok = "experiment_data" in IGNORED_TREE_PARTS
        record(
            checks,
            "experiment_data_ignored",
            ignore_ok,
            "experiment_data is excluded from clean-source hashing",
            ignored_parts=sorted(IGNORED_TREE_PARTS),
        )
        if not ignore_ok:
            raise RuntimeError("EXPERIMENT_DATA_NOT_IGNORED_APPLY_0003_PATCH")

        config = copy.deepcopy(base)
        config["campaign_id"] = args.campaign_id
        config["enabled"] = True
        by_id = mutation_map(config)
        unknown = sorted(set(selected_ids) - set(by_id))
        record(checks, "selected_mutations_exist", not unknown, "All requested mutations exist", unknown=unknown)
        if unknown:
            raise ValueError(f"UNKNOWN_MUTATIONS:{unknown}")
        for mutation in config.get("mutations", []):
            mutation["enabled"] = str(mutation.get("mutation_id")) in selected_ids

        enabled = [m for m in config.get("mutations", []) if m.get("enabled")]
        unresolved = [
            {
                "mutation_id": m.get("mutation_id"),
                "resolution_status": m.get("resolution_status"),
                "resolution_gaps": m.get("resolution_gaps"),
                "resolution_error": m.get("resolution_error"),
            }
            for m in enabled
            if m.get("mutation_class") != "BASELINE" and not isinstance(m.get("resolved_target"), dict)
        ]
        record(checks, "enabled_targets_resolved", not unresolved, "All enabled non-baseline mutations have resolved targets", unresolved=unresolved)
        if unresolved:
            raise ValueError(f"ENABLED_TARGETS_UNRESOLVED:{unresolved}")

        validate_dangerous_target(checks, by_id["A1_s185_dangerous_top1_masked"], label="a1")
        validate_pair(checks, by_id["A1_s185_dangerous_top1_masked"], by_id["B1_s185_mask_bypass"])
        validate_hout_profile(checks, config, "s185_h5")

        # Refresh the binding only after the helper and all source patches are in place.
        config["source_binding"] = {
            "clean_source_root": str(project_root),
            "clean_source_root_sha256": tree_hash(project_root),
        }
        mode_path.parent.mkdir(parents=True, exist_ok=True)
        write_resolved_campaign(mode_path, config)
        sealed = load_json(mode_path)
        verify_config_hash(sealed)
        actual_source_hash = tree_hash(project_root)
        expected_source_hash = str(sealed["source_binding"]["clean_source_root_sha256"])
        record(
            checks,
            "mode_config_hash",
            True,
            "Mode config hash verifies after resealing",
            config_sha256=sealed.get("config_sha256"),
        )
        record(
            checks,
            "source_binding_before_doctor",
            expected_source_hash == actual_source_hash,
            "Mode config binds the current clean source",
            expected=expected_source_hash,
            actual=actual_source_hash,
        )

        # Prove that writing under experiment_data cannot invalidate source binding.
        probe_root = project_root / "experiment_data"
        probe_root.mkdir(parents=True, exist_ok=True)
        probe = probe_root / ".nonvacuity_source_hash_probe.tmp"
        before_probe = tree_hash(project_root)
        probe.write_text("probe\n", encoding="utf-8")
        during_probe = tree_hash(project_root)
        probe.unlink(missing_ok=True)
        after_probe = tree_hash(project_root)
        probe_ok = before_probe == during_probe == after_probe
        record(
            checks,
            "source_hash_output_stability",
            probe_ok,
            "Writing experiment outputs does not change clean-source hash",
            before=before_probe,
            during=during_probe,
            after=after_probe,
        )

        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        preflight = audit_v2_campaign_path(mode_path)
        args.preflight_output.write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        preflight_ok = preflight.get("status") == "PASS"
        record(checks, "preflight", preflight_ok, "S185Core preflight passes", issues=preflight.get("issues", []))

        args.doctor_output.unlink(missing_ok=True)
        doctor = run_doctor(mode_path, args.doctor_output)
        doctor_dict = doctor.to_dict()
        doctor_ok = doctor_dict.get("overall_status") == "PASS"
        doctor_failures = [row for row in doctor_dict.get("checks", []) if row.get("status") == "FAIL"]
        record(checks, "doctor", doctor_ok, "All doctor checks pass", failures=doctor_failures)

        # A second run catches self-invalidating source hashes and stale-receipt behavior.
        second_path = args.doctor_output.with_name(args.doctor_output.stem + ".stability.json")
        second_path.unlink(missing_ok=True)
        doctor2 = run_doctor(mode_path, second_path).to_dict()
        doctor2_ok = doctor2.get("overall_status") == "PASS"
        doctor2_failures = [row for row in doctor2.get("checks", []) if row.get("status") == "FAIL"]
        record(checks, "doctor_stability", doctor2_ok, "Doctor remains PASS after writing its first receipt", failures=doctor2_failures)

        overall = all(row["status"] == "PASS" for row in checks)
        report = {
            "schema_version": "ppp_mode_readiness_v1",
            "status": "READY" if overall else "NOT_READY",
            "campaign_id": args.campaign_id,
            "mode_config": str(mode_path),
            "preflight_output": str(args.preflight_output.resolve()),
            "doctor_output": str(args.doctor_output.resolve()),
            "selected_mutations": selected_ids,
            "checks": checks,
        }
        args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
        args.readiness_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if overall else 2
    except Exception as exc:
        report = {
            "schema_version": "ppp_mode_readiness_v1",
            "status": "NOT_READY",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "checks": checks,
        }
        args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
        args.readiness_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
