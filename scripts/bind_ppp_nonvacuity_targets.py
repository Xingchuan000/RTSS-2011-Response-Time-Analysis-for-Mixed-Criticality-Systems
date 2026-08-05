from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from nonvacuity_lab.audit.action_risk import classify_actions
from nonvacuity_lab.canonical import file_hash, tree_hash
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


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_rows(audit_root: Path) -> list[dict[str, Any]]:
    raw = _load_json(audit_root / "leaf_audit.json")
    rows = raw.get("rows", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("leaf_audit.json must contain a rows array")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _load_actions(tree_path: Path) -> list[Mapping[str, Any]]:
    candidates = [
        tree_path.with_name("action_definitions.json"),
        tree_path.parent.parent / "formal_inputs" / "action_definitions_canonical.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        raw = _load_json(path)
        if isinstance(raw, list):
            return [dict(x) for x in raw if isinstance(x, Mapping)]
        if isinstance(raw, Mapping):
            values = raw.get("actions", raw.get("action_definitions", []))
            if isinstance(values, list):
                return [dict(x) for x in values if isinstance(x, Mapping)]
    return []


def _load_tasks(tree_path: Path) -> list[Mapping[str, Any]]:
    candidates = [
        tree_path.parent / "formal_inputs" / "code_taskset_canonical.json",
        tree_path.parent.parent / "formal_inputs" / "code_taskset_canonical.json",
        tree_path.parent / "taskset.json",
        tree_path.parent.parent / "taskset.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        raw = _load_json(path)
        values = raw.get("ordered_tasks", raw.get("tasks", [])) if isinstance(raw, Mapping) else raw
        if not isinstance(values, list):
            continue
        result: list[Mapping[str, Any]] = []
        for index, item in enumerate(values):
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            name = row.get("name", row.get("task_name", row.get("task_id", row.get("id"))))
            if name is not None:
                row["name"] = str(name)
            row.setdefault("priority", row.get("priority_index", index))
            result.append(row)
        return result
    return []


def _rebuild_action_risks(row: dict[str, Any]) -> list[dict[str, Any]]:
    tree_raw = row.get("tree_path")
    if not tree_raw:
        return []
    tree_path = Path(str(tree_raw)).resolve()
    actions = _load_actions(tree_path)
    tasks = _load_tasks(tree_path)
    if not actions:
        return []
    records = classify_actions(action_definitions=actions, tasks=tasks)
    rejected = dict(row.get("rejected_action_histogram", {}))
    result = []
    for record in records:
        payload = dict(record.__dict__)
        payload["observed_reject_count"] = int(rejected.get(str(record.action_id), 0))
        result.append(payload)
    return result


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        risks = row.get("action_risks")
        if not isinstance(risks, list) or not risks:
            row["action_risks"] = _rebuild_action_risks(row)
            row["action_risks_rebuilt"] = True
        else:
            row["action_risks"] = [dict(x) for x in risks if isinstance(x, Mapping)]
            row["action_risks_rebuilt"] = False
        normalized.append(row)
    return normalized


def candidates(rows: list[dict[str, Any]], seed: int, variant: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            row_seed = int(row.get("seed", -1))
        except (TypeError, ValueError):
            continue
        if row_seed != seed or str(row.get("tree_variant")) != variant:
            continue
        ranking = [int(x) for x in row.get("action_ranking", [])]
        hit_count = int(row.get("hout_hit_count", 0) or 0)
        if not ranking or hit_count <= 0:
            continue
        for risk in row.get("action_risks", []):
            if not isinstance(risk, Mapping):
                continue
            risk_class = str(risk.get("risk_class", "BENIGN_OR_UNKNOWN"))
            action_id = int(risk.get("action_id", -1))
            if RISK_ORDER.get(risk_class, 0) <= 0 or action_id == ranking[0] or action_id not in ranking:
                continue
            scenario_coverage = row.get("scenario_coverage", [])
            scenario_count = len(scenario_coverage) if isinstance(scenario_coverage, list) else int(row.get("scenario_coverage_count", 0) or 0)
            result.append(
                {
                    "seed": seed,
                    "tree_variant": variant,
                    "leaf_id": int(row["leaf_id"]),
                    "action_id": action_id,
                    "risk_class": risk_class,
                    "hout_hit_count": hit_count,
                    "scenario_coverage_count": scenario_count,
                    "training_samples": int(row.get("training_samples", 0) or 0),
                    "observed_reject_count_before_mutation": int(risk.get("observed_reject_count", 0) or 0),
                    "tree_path": str(row["tree_path"]),
                    "tree_hash": str(row.get("tree_hash") or file_hash(Path(str(row["tree_path"])))),
                    "original_ranking": ranking,
                    "action_record": dict(risk),
                    "action_risks_rebuilt": bool(row.get("action_risks_rebuilt", False)),
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


def scope_diagnostics(rows: list[dict[str, Any]], seed: int, variant: str) -> dict[str, Any]:
    scoped = [r for r in rows if str(r.get("tree_variant")) == variant and str(r.get("seed")) == str(seed)]
    covered = [r for r in scoped if int(r.get("hout_hit_count", 0) or 0) > 0]
    risk_counts: Counter[str] = Counter()
    non_top1_risk_counts: Counter[str] = Counter()
    for row in covered:
        ranking = [int(x) for x in row.get("action_ranking", [])]
        for risk in row.get("action_risks", []):
            if not isinstance(risk, Mapping):
                continue
            rc = str(risk.get("risk_class", "BENIGN_OR_UNKNOWN"))
            risk_counts[rc] += 1
            if ranking and int(risk.get("action_id", -1)) != ranking[0]:
                non_top1_risk_counts[rc] += 1
    return {
        "seed": seed,
        "tree_variant": variant,
        "row_count": len(scoped),
        "covered_leaf_count": len(covered),
        "hout_hit_total": sum(int(r.get("hout_hit_count", 0) or 0) for r in covered),
        "rows_with_action_risks": sum(bool(r.get("action_risks")) for r in scoped),
        "rows_with_rebuilt_action_risks": sum(bool(r.get("action_risks_rebuilt")) for r in scoped),
        "risk_counts_on_covered_leaves": dict(risk_counts),
        "non_top1_risk_counts_on_covered_leaves": dict(non_top1_risk_counts),
        "candidate_count": len(candidates(rows, seed, variant)),
    }


def select_candidate(rows: list[dict[str, Any]], *, seed: int, variant: str, leaf_override: int | None, action_override: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked = candidates(rows, seed, variant)
    if leaf_override is None and action_override is None:
        if not ranked:
            diag = scope_diagnostics(rows, seed, variant)
            raise ValueError(f"NO_COVERED_DANGEROUS_CANDIDATE:{json.dumps(diag, ensure_ascii=False, sort_keys=True)}")
        return ranked[0], ranked
    if leaf_override is None or action_override is None:
        raise ValueError("leaf/action override must be supplied together")
    for item in ranked:
        if item["leaf_id"] == leaf_override and item["action_id"] == action_override:
            return item, ranked
    raise ValueError(
        f"REQUESTED_CANDIDATE_NOT_FOUND:s{seed}:{variant}:leaf={leaf_override}:action={action_override}:"
        f"{json.dumps(scope_diagnostics(rows, seed, variant), ensure_ascii=False, sort_keys=True)}"
    )


def seed_dir_from_tree(tree_path: str, seed: int) -> Path:
    path = Path(tree_path).resolve()
    for parent in path.parents:
        if parent.name == f"s{seed}":
            return parent
    return path.parent.parent


def _find_mutation(config: dict[str, Any], mutation_id: str) -> dict[str, Any]:
    for item in config.get("mutations", []):
        if item.get("mutation_id") == mutation_id:
            return item
    raise ValueError(f"MUTATION_ID_NOT_FOUND:{mutation_id}")


def bind_target(config: dict[str, Any], mutation_id: str, choice: dict[str, Any]) -> None:
    mutation = _find_mutation(config, mutation_id)
    seed = int(choice["seed"])
    mutation["resolved_target"] = {
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
            "action_risks_rebuilt": bool(choice.get("action_risks_rebuilt", False)),
        },
    }
    mutation["resolution_status"] = "RESOLVED"
    mutation.pop("resolution_gaps", None)
    mutation.pop("resolution_error", None)


def copy_pair_target(config: dict[str, Any], producer_id: str, consumer_id: str) -> None:
    producer = _find_mutation(config, producer_id)
    consumer = _find_mutation(config, consumer_id)
    consumer["resolved_target"] = json.loads(json.dumps(producer["resolved_target"]))
    consumer["resolution_status"] = "RESOLVED"
    consumer.pop("resolution_gaps", None)
    consumer.pop("resolution_error", None)


def build_profile(*, data_root: Path, scenario_seeds: list[int], seed: int, required: list[int]) -> dict[str, Any]:
    seed_root = data_root / "seeds" / f"s{seed}"
    command = [
        "python", "scripts/run_nonvacuity_hout.py", "--seed-dir", "{seed_dir}", "--tree", "{tree_path}",
        "--scenario-file", "{scenario_file}", "--runtime-config", "{runtime_config}", "--taskset", "{taskset}",
        "--output-dir", "{output_dir}",
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
            "lo_quality_qos", "lo_zero_service_ratio", "tree_raw_top1_invalid_rate", "tree_fallback_rate", "hi_deadline_misses",
        ],
    }


def bind_profiles(config: dict[str, Any], data_root: Path, scenario_file: Path) -> None:
    raw = _load_json(scenario_file)
    values = raw.get("scenario_seeds", raw.get("scenarios")) if isinstance(raw, dict) else raw
    if not isinstance(values, list) or not values:
        raise ValueError("SCENARIO_FILE_MUST_CONTAIN_NONEMPTY_LIST")
    scenario_seeds = [int(x) for x in values]
    profiles = config.setdefault("hout_profiles", {})
    profiles["s185_h5"] = build_profile(data_root=data_root, scenario_seeds=scenario_seeds, seed=185, required=[])
    profiles["s1264_h5"] = build_profile(data_root=data_root, scenario_seeds=scenario_seeds, seed=1264, required=[])
    profiles["s397_h5"] = build_profile(data_root=data_root, scenario_seeds=scenario_seeds, seed=397, required=[101555, 101593])
    bindings = {
        "A1_s185_dangerous_top1_masked": "s185_h5", "B1_s185_mask_bypass": "s185_h5",
        "C1_action_ratio": "s185_h5", "C3_retroactive_release": "s185_h5",
        "A2_s397_dangerous_top1_masked": "s397_h5", "B5_s397_mask_bypass": "s397_h5",
        "B2_s1264_no_first_valid": "s1264_h5", "B3_s1264_all_invalid_force_top1": "s1264_h5",
        "B4_s1264_guard_ablation": "s1264_h5",
    }
    for mutation in config.get("mutations", []):
        mid = str(mutation.get("mutation_id"))
        if mid in bindings:
            mutation["hout_profile_id"] = bindings[mid]


def _failure_path(args: argparse.Namespace) -> Path:
    if args.report:
        return args.report.resolve().with_name("binding_failure.json")
    return args.config.resolve().parent / "binding_failure.json"


def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    audit_root = args.audit_root.resolve()
    project_root = args.project_root.resolve()
    data_root = args.experiment_data_root.resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != "nonvacuity_campaign_v2" or config.get("config_kind") != "RESOLVED":
        raise ValueError("CONFIG_MUST_BE_RESOLVED_NONVACUITY_CAMPAIGN_V2")
    rows = normalize_rows(load_rows(audit_root))
    diagnostics = {
        "A1": scope_diagnostics(rows, 185, "best_overall"),
        "A2": scope_diagnostics(rows, 397, "best_balanced"),
    }
    print(json.dumps({"status": "BINDING_DIAGNOSTICS", "diagnostics": diagnostics}, ensure_ascii=False, indent=2))
    a1, a1_ranked = select_candidate(rows, seed=185, variant="best_overall", leaf_override=args.a1_leaf, action_override=args.a1_action)
    a2, a2_ranked = select_candidate(rows, seed=397, variant="best_balanced", leaf_override=args.a2_leaf, action_override=args.a2_action)
    bind_target(config, "A1_s185_dangerous_top1_masked", a1)
    bind_target(config, "A2_s397_dangerous_top1_masked", a2)
    copy_pair_target(config, "A1_s185_dangerous_top1_masked", "B1_s185_mask_bypass")
    copy_pair_target(config, "A2_s397_dangerous_top1_masked", "B5_s397_mask_bypass")
    bind_profiles(config, data_root, args.scenario_file.resolve())
    config.setdefault("output_roots", {})["nonvacuity_lab"] = str((data_root / "results").resolve())
    config["source_binding"] = {
        "clean_source_root": str(project_root),
        "clean_source_root_sha256": tree_hash(project_root),
    }
    config["enabled"] = False
    for mutation in config.get("mutations", []):
        mutation["enabled"] = False
    write_resolved_campaign(config_path, config)
    report = {
        "schema_version": "ppp_manual_target_binding_v2",
        "selected": {"A1": a1, "A2": a2},
        "candidates": {"A1": a1_ranked[: args.top], "A2": a2_ranked[: args.top]},
        "diagnostics": diagnostics,
        "warning": "A1/A2 are publishable only after mutated HOUT records raw_top1_invalid for the exact leaf/action.",
    }
    report_path = (args.report or (config_path.parent / "dangerous_target_candidates.json")).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failure = _failure_path(args)
    if failure.exists():
        failure.unlink()
    print(json.dumps({
        "status": "TARGETS_AND_PROFILES_BOUND", "config": str(config_path), "report": str(report_path),
        "A1": {k: a1[k] for k in ("leaf_id", "action_id", "risk_class", "hout_hit_count")},
        "A2": {k: a2[k] for k in ("leaf_id", "action_id", "risk_class", "hout_hit_count")},
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:  # deliberately produce a stable diagnostic artifact
        payload = {
            "schema_version": "ppp_target_binding_failure_v1",
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "inputs": {
                "config": str(args.config.resolve()),
                "audit_root": str(args.audit_root.resolve()),
                "project_root": str(args.project_root.resolve()),
                "experiment_data_root": str(args.experiment_data_root.resolve()),
                "scenario_file": str(args.scenario_file.resolve()),
            },
        }
        path = _failure_path(args)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
