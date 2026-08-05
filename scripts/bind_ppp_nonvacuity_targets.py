from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nonvacuity_lab.config_io import write_resolved_campaign

RISK_ORDER = {
    "HI_BUDGET_DECREASE": 4,
    "HIGHER_PRIORITY_LO_INCREASE": 3,
    "BUDGET_INCREASE": 2,
    "BENIGN_OR_UNKNOWN": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind A1/A2 dangerous actions and exact HOUT profiles into a resolved PPP campaign."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment-data-root", type=Path, required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--a1-leaf", type=int)
    parser.add_argument("--a1-action", type=int)
    parser.add_argument("--a2-leaf", type=int)
    parser.add_argument("--a2-action", type=int)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def load_rows(audit_root: Path) -> list[dict[str, Any]]:
    raw = json.loads((audit_root / "leaf_audit.json").read_text(encoding="utf-8"))
    rows = raw.get("rows", []) if isinstance(raw, dict) else raw
    return [dict(row) for row in rows if isinstance(row, dict)]


def candidates(rows: list[dict[str, Any]], seed: int, variant: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("seed", -1)) != seed or str(row.get("tree_variant")) != variant:
            continue
        ranking = [int(x) for x in row.get("action_ranking", [])]
        if not ranking or int(row.get("hout_hit_count", 0)) <= 0:
            continue
        for risk in row.get("action_risks", []):
            risk_class = str(risk.get("risk_class", "BENIGN_OR_UNKNOWN"))
            action_id = int(risk.get("action_id", -1))
            if RISK_ORDER.get(risk_class, 0) <= 0 or action_id == ranking[0]:
                continue
            result.append(
                {
                    "seed": seed,
                    "tree_variant": variant,
                    "leaf_id": int(row["leaf_id"]),
                    "action_id": action_id,
                    "risk_class": risk_class,
                    "hout_hit_count": int(row.get("hout_hit_count", 0)),
                    "scenario_coverage_count": len(row.get("scenario_coverage", [])),
                    "training_samples": int(row.get("training_samples", 0)),
                    "observed_reject_count_before_mutation": int(risk.get("observed_reject_count", 0)),
                    "tree_path": str(row["tree_path"]),
                    "tree_hash": str(row["tree_hash"]),
                    "original_ranking": ranking,
                    "action_record": dict(risk),
                }
            )
    result.sort(
        key=lambda item: (
            item["hout_hit_count"],
            item["scenario_coverage_count"],
            RISK_ORDER.get(item["risk_class"], 0),
            item["training_samples"],
        ),
        reverse=True,
    )
    return result


def select_candidate(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    variant: str,
    leaf_override: int | None,
    action_override: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked = candidates(rows, seed, variant)
    if leaf_override is None and action_override is None:
        if not ranked:
            raise ValueError(f"no covered dangerous candidate for s{seed} {variant}")
        return ranked[0], ranked
    if leaf_override is None or action_override is None:
        raise ValueError("leaf/action override must be supplied together")
    for item in ranked:
        if item["leaf_id"] == leaf_override and item["action_id"] == action_override:
            return item, ranked
    raise ValueError(
        f"requested candidate not found or not covered: s{seed} {variant} "
        f"leaf={leaf_override} action={action_override}"
    )


def seed_dir_from_tree(tree_path: str, seed: int) -> Path:
    path = Path(tree_path).resolve()
    for parent in path.parents:
        if parent.name == f"s{seed}":
            return parent
    return path.parent.parent


def bind_target(config: dict[str, Any], mutation_id: str, choice: dict[str, Any]) -> None:
    mutation = next(item for item in config["mutations"] if item.get("mutation_id") == mutation_id)
    seed = int(choice["seed"])
    target = {
        "seed_dir": str(seed_dir_from_tree(choice["tree_path"], seed)),
        "tree_path": str(Path(choice["tree_path"]).resolve()),
        "tree_sha256": choice["tree_hash"],
        "leaf_id": int(choice["leaf_id"]),
        "action_id": int(choice["action_id"]),
        "original_ranking": list(choice["original_ranking"]),
        "manual_resolution": {
            "method": "coverage_then_risk",
            "risk_class": choice["risk_class"],
            "hout_hit_count": choice["hout_hit_count"],
            "scenario_coverage_count": choice["scenario_coverage_count"],
            "activation_status": "PENDING_MUTATED_HOUT_CONFIRMATION",
        },
    }
    mutation["resolved_target"] = target
    mutation["resolution_status"] = "RESOLVED"
    mutation.pop("resolution_gaps", None)
    mutation.pop("resolution_error", None)


def copy_pair_target(config: dict[str, Any], producer_id: str, consumer_id: str) -> None:
    producer = next(item for item in config["mutations"] if item.get("mutation_id") == producer_id)
    consumer = next(item for item in config["mutations"] if item.get("mutation_id") == consumer_id)
    consumer["resolved_target"] = json.loads(json.dumps(producer["resolved_target"]))
    consumer["resolution_status"] = "RESOLVED"
    consumer.pop("resolution_gaps", None)
    consumer.pop("resolution_error", None)


def build_profile(*, data_root: Path, scenario_seeds: list[int], seed: int, required: list[int]) -> dict[str, Any]:
    seed_root = data_root / "seeds" / f"s{seed}"
    command = [
        "python",
        "scripts/run_nonvacuity_hout.py",
        "--seed-dir",
        "{seed_dir}",
        "--tree",
        "{tree_path}",
        "--scenario-file",
        "{scenario_file}",
        "--runtime-config",
        "{runtime_config}",
        "--taskset",
        "{taskset}",
        "--output-dir",
        "{output_dir}",
    ]
    return {
        "taskset_path": str((seed_root / "formal_inputs" / "code_taskset_canonical.json").resolve()),
        "runtime_config_path": str((data_root / "runtime" / f"s{seed}_h5.json").resolve()),
        "scenario_seeds": scenario_seeds,
        "required_scenarios": required,
        "horizon": 50_000_000,
        "controller_release_times": [0],
        "worker_count": 1,
        "random_seed": seed,
        "base_command": command,
        "mutated_command": list(command),
        "required_metrics": [
            "lo_quality_qos",
            "lo_zero_service_ratio",
            "tree_raw_top1_invalid_rate",
            "tree_fallback_rate",
            "hi_deadline_misses",
        ],
    }


def bind_profiles(config: dict[str, Any], data_root: Path, scenario_file: Path) -> None:
    raw = json.loads(scenario_file.read_text(encoding="utf-8"))
    values = raw.get("scenario_seeds", raw.get("scenarios")) if isinstance(raw, dict) else raw
    if not isinstance(values, list) or not values:
        raise ValueError("scenario file must contain a non-empty list")
    scenario_seeds = [int(x) for x in values]
    profiles = config.setdefault("hout_profiles", {})
    profiles["s185_h5"] = build_profile(data_root=data_root, scenario_seeds=scenario_seeds, seed=185, required=[])
    profiles["s1264_h5"] = build_profile(data_root=data_root, scenario_seeds=scenario_seeds, seed=1264, required=[])
    profiles["s397_h5"] = build_profile(
        data_root=data_root,
        scenario_seeds=scenario_seeds,
        seed=397,
        required=[101555, 101593],
    )
    profile_bindings = {
        "A1_s185_dangerous_top1_masked": "s185_h5",
        "B1_s185_mask_bypass": "s185_h5",
        "C1_action_ratio": "s185_h5",
        "C3_retroactive_release": "s185_h5",
        "A2_s397_dangerous_top1_masked": "s397_h5",
        "B5_s397_mask_bypass": "s397_h5",
        "B2_s1264_no_first_valid": "s1264_h5",
        "B3_s1264_all_invalid_force_top1": "s1264_h5",
        "B4_s1264_guard_ablation": "s1264_h5",
    }
    for mutation in config.get("mutations", []):
        mutation_id = str(mutation.get("mutation_id"))
        if mutation_id in profile_bindings:
            mutation["hout_profile_id"] = profile_bindings[mutation_id]


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    audit_root = args.audit_root.resolve()
    project_root = args.project_root.resolve()
    data_root = args.experiment_data_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "nonvacuity_campaign_v2" or config.get("config_kind") != "RESOLVED":
        raise ValueError("--config must be a resolved nonvacuity_campaign_v2")
    rows = load_rows(audit_root)
    a1, a1_ranked = select_candidate(
        rows,
        seed=185,
        variant="best_overall",
        leaf_override=args.a1_leaf,
        action_override=args.a1_action,
    )
    a2, a2_ranked = select_candidate(
        rows,
        seed=397,
        variant="best_balanced",
        leaf_override=args.a2_leaf,
        action_override=args.a2_action,
    )
    bind_target(config, "A1_s185_dangerous_top1_masked", a1)
    bind_target(config, "A2_s397_dangerous_top1_masked", a2)
    copy_pair_target(config, "A1_s185_dangerous_top1_masked", "B1_s185_mask_bypass")
    copy_pair_target(config, "A2_s397_dangerous_top1_masked", "B5_s397_mask_bypass")
    bind_profiles(config, data_root, args.scenario_file.resolve())
    config.setdefault("output_roots", {})["nonvacuity_lab"] = str((data_root / "results").resolve())
    config["source_binding"] = {
        "clean_source_root": str(project_root),
        "clean_source_root_sha256": config.get("source_binding", {}).get("clean_source_root_sha256", ""),
    }
    config["enabled"] = False
    for mutation in config.get("mutations", []):
        mutation["enabled"] = False
    write_resolved_campaign(config_path, config)

    report = {
        "schema_version": "ppp_manual_target_binding_v1",
        "selected": {"A1": a1, "A2": a2},
        "candidates": {"A1": a1_ranked[: args.top], "A2": a2_ranked[: args.top]},
        "warning": (
            "The selected action is only an initial candidate. A1/A2 are publishable only after "
            "mutated HOUT records raw_top1_invalid for the exact leaf/action."
        ),
    }
    report_path = (args.report or (config_path.parent / "dangerous_target_candidates.json")).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "TARGETS_AND_PROFILES_BOUND",
        "config": str(config_path),
        "report": str(report_path),
        "A1": {k: a1[k] for k in ("leaf_id", "action_id", "risk_class", "hout_hit_count")},
        "A2": {k: a2[k] for k in ("leaf_id", "action_id", "risk_class", "hout_hit_count")},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
