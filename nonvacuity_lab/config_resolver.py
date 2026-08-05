from __future__ import annotations

import copy
import json
from pathlib import Path

from .canonical import file_hash, tree_hash
from .config_io import write_resolved_campaign
from .resolver.dangerous_top1 import resolve_dangerous_top1
from .coherence import missing_roles


def require_resolved_dangerous_target(mutation: dict) -> None:
    """Fail closed unless a dangerous target is fully bound by Phase 3 data."""
    target = mutation.get("resolved_target")
    required = {"seed_dir", "tree_path", "tree_sha256", "leaf_id", "action_id", "original_ranking"}
    if not isinstance(target, dict):
        raise ValueError("DANGEROUS_TARGET_NOT_RESOLVED")
    missing = sorted(required - set(target))
    if missing:
        raise ValueError(f"DANGEROUS_TARGET_FIELDS_MISSING:{missing}")
    ranking = [int(item) for item in target["original_ranking"]]
    if ranking and int(target["action_id"]) == ranking[0]:
        raise ValueError("DANGEROUS_ACTION_ALREADY_TOP1")


def resolve_campaign(
    template_path: Path, audit_root: Path, source_root: Path, output_path: Path,
    *, require_all_resolved: bool = False,
) -> dict:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    if template.get("schema_version") == "nonvacuity_campaign_v2":
        return _resolve_v2_campaign(
            template, template_path, audit_root, source_root, output_path,
            require_all_resolved=require_all_resolved,
        )
    if template.get("config_kind") != "TEMPLATE":
        raise ValueError("resolve requires config_kind=TEMPLATE")
    resolved = copy.deepcopy(template)
    resolved["config_kind"] = "RESOLVED"
    resolved["enabled"] = False
    resolved["source_binding"] = {
        "clean_source_root": str(source_root.resolve()),
        "clean_source_root_sha256": tree_hash(source_root),
    }
    records = []
    for mutation in resolved.get("mutations", []):
        mutation["enabled"] = False
        selector = mutation.pop("selector", None)
        target = mutation.get("resolved_target")
        if selector and not target:
            # Resolve the common dangerous-top1 selector from the Phase-3
            # audit artifact.  If the audit is incomplete, fail closed and
            # keep the mutation disabled.
            try:
                audit_file = audit_root / "leaf_audit.json"
                audit = json.loads(audit_file.read_text(encoding="utf-8"))
                rows = audit.get("rows", audit if isinstance(audit, list) else [])
                seed = selector.get("seed")
                variant = selector.get("tree_variant", selector.get("variant"))
                scoped = [r for r in rows if (seed is None or int(r.get("seed", -1)) == int(seed)) and (variant is None or str(r.get("tree_variant")) == str(variant))]
                resolved_choice = resolve_dangerous_top1(scoped, require_hout_hit=False)
                candidate_path = Path(str(next((r.get("tree_path") for r in scoped if r.get("tree_path")), "")))
                target = {"leaf_id": resolved_choice.leaf_id, "action_id": resolved_choice.action_id, "tree_sha256": resolved_choice.tree_hash, "activation_witness_ref": resolved_choice.witness_ref}
                if candidate_path.as_posix() not in {"", "."}:
                    target["tree_path"] = str(candidate_path.resolve())
                mutation["resolved_target"] = target
                records.append({"mutation_id": mutation.get("mutation_id"), "status": "RESOLVED", "target": target})
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                records.append({"mutation_id": mutation.get("mutation_id"), "status": "UNRESOLVED_SELECTOR", "selector": selector, "reason": str(exc)})
        else:
            records.append({"mutation_id": mutation.get("mutation_id"), "status": "ALREADY_RESOLVED"})
    receipt = output_path.with_suffix(".resolver_receipt.json")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"schema_version": "nonvacuity_resolver_receipt_v1", "template_sha256": file_hash(template_path), "audit_root_sha256": tree_hash(audit_root), "source_root_sha256": tree_hash(source_root), "records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resolved["resolver_receipt"] = {"path": str(receipt.resolve()), "sha256": file_hash(receipt)}
    write_resolved_campaign(output_path, resolved)
    return resolved


def _resolve_v2_campaign(
    template: dict, template_path: Path, audit_root: Path, source_root: Path, output_path: Path,
    *, require_all_resolved: bool = False,
) -> dict:
    """Resolve all Phase-4--9 experiment bindings while keeping every row disabled."""
    from .mutators.catalog.selection_mutations import build_selection_catalog
    from .mutators.catalog.rounding_mutations import build_rounding_catalog
    from .mutators.catalog.guard_mutations import build_guard_catalog
    from .mutators.catalog.model_mutations import build_current_source_model_catalog
    from .resolver.guard_ablation import resolve_guard_ablation

    audit_root = Path(audit_root).resolve()
    source_root = Path(source_root).resolve()
    resolved = copy.deepcopy(template)
    resolved["config_kind"] = "RESOLVED"
    resolved["enabled"] = False
    resolved["source_binding"] = {
        "clean_source_root": str(source_root),
        "clean_source_root_sha256": tree_hash(source_root),
    }
    output_roots = resolved.setdefault("output_roots", {})
    lab_output = Path(str(output_roots.get("nonvacuity_lab", "outputs/nonvacuity")))
    if not lab_output.is_absolute():
        lab_output = source_root / lab_output
    output_roots["nonvacuity_lab"] = str(lab_output.resolve())
    for profile in resolved.get("hout_profiles", {}).values():
        if not isinstance(profile, dict):
            continue
        for key in ("taskset_path", "runtime_config_path", "demand_trace_path"):
            raw_path = profile.get(key)
            if not raw_path:
                continue
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = source_root / candidate
            profile[key] = str(candidate.resolve())
    records: list[dict] = []
    audit_rows = _load_rows_file(audit_root / "leaf_audit.json")
    audit_summary = _load_json_object(audit_root / "audit_summary.json")
    selection_catalog = build_selection_catalog(source_root)
    rounding_catalog = build_rounding_catalog(source_root)
    guard_patch_catalog = build_guard_catalog(source_root)
    model_catalog = build_current_source_model_catalog(source_root)
    guard_contract = _load_json_object(source_root / "nonvacuity_lab/catalogs/ppp_guard_catalog.json")

    for mutation in resolved.get("mutations", []):
        mutation["enabled"] = False
        mutation_id = str(mutation.get("mutation_id", ""))
        canonical = mutation_id.split("_", 1)[0]
        seed = mutation.get("seed")
        variant = mutation.get("tree_variant")
        scoped = _scope_v2_rows(audit_rows, seed=seed, variant=variant)
        try:
            if seed is not None:
                _bind_seed_target(mutation, scoped, seed=int(seed))

            parameters = mutation.setdefault("mutator", {}).setdefault("parameters", {})
            catalog_key = "B1" if canonical == "B5" else canonical
            if catalog_key in selection_catalog:
                parameters["patches"] = [dict(item) for item in selection_catalog[catalog_key]]
            elif canonical == "C2":
                parameters["patches"] = [dict(item) for item in rounding_catalog]
            elif canonical in model_catalog:
                entry = model_catalog[canonical]
                parameters["semantic_change_id"] = entry.semantic_change_id
                parameters["patches"] = [dict(item) for item in entry.patches]
                expected = mutation.setdefault("expected", {})
                expected.setdefault("allowed_result_statuses", [entry.expected_status])
                expected.setdefault("allowed_first_failing_obligations", list(entry.expected_obligations))
                expected.setdefault("require_failure", True)

            selector = mutation.pop("selector", None)
            target = mutation.setdefault("resolved_target", {})
            if selector and not {"leaf_id", "action_id"}.issubset(target):
                _bind_symbolic_target(mutation, scoped, dangerous=True)
            elif canonical == "B2" and not {"leaf_id", "action_id"}.issubset(target):
                _bind_observed_raw_target(mutation, scoped, evidence_field="raw_top1_invalid_count")
            elif canonical == "B3" and not {"leaf_id", "action_id"}.issubset(target):
                _bind_observed_raw_target(mutation, scoped, evidence_field="all_invalid_count")
            elif canonical == "C2" and not {"leaf_id", "action_id"}.issubset(target):
                _bind_c2_rounding_target(mutation, scoped)

            if canonical == "B4":
                guard_target = resolve_guard_ablation(scoped, guard_contract)
                patch_key = str(guard_target["catalog_key"])
                if patch_key not in guard_patch_catalog:
                    raise ValueError(f"B4_PATCH_CATALOG_MISSING:{patch_key}")
                parameters["patches"] = [dict(item) for item in guard_patch_catalog[patch_key]]
                target.update(guard_target)
                row = next((item for item in scoped if int(item.get("leaf_id", -1)) == int(target["leaf_id"])), None)
                if row is not None:
                    target.setdefault("original_ranking", list(row.get("action_ranking", ())))

            _bind_symbolic_activation(mutation, canonical)
            _bind_hout_profile(resolved, mutation)

            if canonical == "D1":
                proof_root = audit_summary.get("proof_bundle_root")
                metadata = mutation.setdefault("metadata", {})
                if proof_root:
                    metadata["bundle_roots"] = [str(Path(proof_root).resolve())]
                seed_dirs: dict[str, str] = {}
                variants: dict[str, list[str]] = {}
                for audit_row in audit_rows:
                    row_seed = audit_row.get("seed")
                    tree_path_raw = audit_row.get("tree_path")
                    if row_seed is None or not tree_path_raw:
                        continue
                    tree_path = Path(str(tree_path_raw)).resolve()
                    seed_dir = audit_row.get("seed_dir")
                    if not seed_dir:
                        seed_dir = next(
                            (str(parent) for parent in tree_path.parents if parent.name == f"s{int(row_seed)}"),
                            str(tree_path.parent.parent),
                        )
                    seed_dirs.setdefault(str(int(row_seed)), str(Path(seed_dir).resolve()))
                    variant_value = str(audit_row.get("tree_variant", tree_path.parent.name))
                    variants.setdefault(str(int(row_seed)), [])
                    if variant_value not in variants[str(int(row_seed))]:
                        variants[str(int(row_seed))].append(variant_value)
                metadata["seed_dirs_by_seed"] = seed_dirs
                metadata["tree_variants_by_seed"] = variants

            if canonical.startswith("F"):
                proof_root_raw = audit_summary.get("proof_bundle_root")
                if not isinstance(proof_root_raw, str) or not proof_root_raw.strip():
                    raise ValueError("PROOF_BUNDLE_ROOT_MISSING")
                proof_root = Path(proof_root_raw).resolve()
                if not proof_root.is_dir():
                    raise ValueError(f"PROOF_BUNDLE_ROOT_NOT_FOUND:{proof_root}")
                _bind_integrity_mutation(
                    mutation, canonical=canonical,
                    proof_root=proof_root,
                    source_root=source_root,
                )
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            mutation["resolution_status"] = "UNRESOLVED_BINDING"
            mutation["resolution_error"] = str(exc)
            records.append({"mutation_id": mutation_id, "status": "UNRESOLVED_BINDING", "reason": str(exc)})

    # B1/B5 consume the exact A1/A2 mutated tree target.
    by_id = {str(item.get("mutation_id")): item for item in resolved.get("mutations", [])}
    for mutation in resolved.get("mutations", []):
        pair_id = mutation.get("pair_with")
        if pair_id and pair_id in by_id and by_id[pair_id].get("resolved_target"):
            pair_target = copy.deepcopy(by_id[pair_id]["resolved_target"])
            # Preserve source-semantic activation fields added to the B row.
            pair_target.update({
                key: value for key, value in mutation.get("resolved_target", {}).items()
                if key not in pair_target
            })
            mutation["resolved_target"] = pair_target

    for mutation in resolved.get("mutations", []):
        gaps = _v2_resolution_gaps(mutation)
        if gaps:
            mutation["resolution_status"] = "UNRESOLVED"
            mutation["resolution_gaps"] = list(gaps)
        elif mutation.get("resolution_status") != "UNRESOLVED_BINDING":
            mutation["resolution_status"] = "RESOLVED"
            mutation.pop("resolution_gaps", None)
            mutation.pop("resolution_error", None)
        records.append({
            "mutation_id": str(mutation.get("mutation_id", "")),
            "status": "RESOLVED" if not gaps else "UNRESOLVED",
            "gaps": gaps,
        })
    unresolved = [item for item in records if str(item.get("status", "")).startswith("UNRESOLVED")]
    if require_all_resolved and unresolved:
        summary = {item.get("mutation_id"): item.get("gaps", item.get("reason")) for item in unresolved}
        raise ValueError(f"CAMPAIGN_NOT_FULLY_RESOLVED:{summary}")

    receipt = output_path.with_suffix(".resolver_receipt.json")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "schema_version": "nonvacuity_resolver_receipt_v3",
        "template_sha256": file_hash(template_path),
        "audit_root_sha256": tree_hash(audit_root) if audit_root.is_dir() else None,
        "source_root_sha256": tree_hash(source_root),
        "records": records,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resolved["resolver_receipt"] = {"path": str(receipt.resolve()), "sha256": file_hash(receipt)}
    write_resolved_campaign(output_path, resolved)
    return json.loads(output_path.read_text(encoding="utf-8"))


def _load_json_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


def _load_rows_file(path: Path) -> list[dict]:
    value = _load_json_object(path)
    rows = value.get("rows", [])
    return [dict(item) for item in rows if isinstance(item, dict)]


def _scope_v2_rows(rows: list[dict], *, seed, variant) -> list[dict]:
    return [
        row for row in rows
        if (seed is None or int(row.get("seed", -1)) == int(seed))
        and (variant is None or str(row.get("tree_variant")) == str(variant))
    ]


def _bind_seed_target(mutation: dict, scoped: list[dict], *, seed: int) -> None:
    row = next((item for item in scoped if item.get("tree_path")), None)
    if row is None:
        raise ValueError(f"SEED_AUDIT_ROW_MISSING:s{seed}")
    tree_path = Path(str(row["tree_path"])).resolve()
    seed_dir = row.get("seed_dir")
    if not seed_dir:
        seed_dir = next((str(parent) for parent in tree_path.parents if parent.name == f"s{seed}"), None)
    if not seed_dir:
        # Most experiment bundles use seed/<variant>/integer_tree.json even if
        # the directory is not literally named s<seed>.
        seed_dir = str(tree_path.parent.parent)
    target = mutation.setdefault("resolved_target", {})
    target.update({
        "seed_dir": str(Path(seed_dir).resolve()),
        "tree_path": str(tree_path),
        "tree_sha256": file_hash(tree_path),
    })


def _bind_symbolic_target(mutation: dict, scoped: list[dict], *, dangerous: bool) -> None:
    if not scoped:
        raise ValueError("SYMBOLIC_TARGET_AUDIT_SCOPE_EMPTY")
    if dangerous:
        choice = resolve_dangerous_top1(scoped, require_hout_hit=False)
        leaf_id, action_id = choice.leaf_id, choice.action_id
    else:
        row = max(scoped, key=lambda item: (int(item.get("hout_hit_count", 0)), int(item.get("training_samples", 0))))
        ranking = list(row.get("action_ranking", ()))
        if not ranking:
            raise ValueError("SYMBOLIC_TARGET_RANKING_MISSING")
        leaf_id, action_id = int(row["leaf_id"]), int(ranking[0])
    row = next(item for item in scoped if int(item.get("leaf_id", -1)) == int(leaf_id))
    ranking = list(row.get("action_ranking", ()))
    target = mutation.setdefault("resolved_target", {})
    target.update({"leaf_id": int(leaf_id), "action_id": int(action_id), "original_ranking": ranking})




def _bind_observed_raw_target(mutation: dict, scoped: list[dict], *, evidence_field: str) -> None:
    """Bind B2/B3 to an original raw top-1 action, never to an unpromoted action."""
    candidates = [row for row in scoped if row.get("action_ranking")]
    if not candidates:
        raise ValueError(f"SYMBOLIC_RAW_TARGET_MISSING:{evidence_field}")
    row = max(
        candidates,
        key=lambda item: (
            int(item.get(evidence_field, 0) > 0),
            int(item.get(evidence_field, 0)),
            int(item.get("hout_hit_count", 0)),
            int(item.get("fallback_count", 0)),
            int(item.get("training_samples", 0)),
        ),
    )
    ranking = [int(item) for item in row.get("action_ranking", ())]
    mutation.setdefault("resolved_target", {}).update({
        "leaf_id": int(row["leaf_id"]),
        "action_id": int(ranking[0]),
        "original_ranking": ranking,
        "activation_evidence_field": evidence_field,
        "activation_evidence_count": int(row.get(evidence_field, 0)),
    })


def _action_rows(raw) -> list[dict]:
    if isinstance(raw, dict) and isinstance(raw.get("actions"), list):
        return [dict(item) for item in raw["actions"] if isinstance(item, dict)]
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        rows = []
        for key, item in raw.items():
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row.setdefault("action_id", int(key))
            rows.append(row)
        return rows
    return []


def _task_contracts(raw) -> dict[str, dict]:
    rows = raw.get("ordered_tasks", raw.get("tasks", [])) if isinstance(raw, dict) else []
    result = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", item.get("task_id")))
        reference = int(item.get("initial_runtime_budget", item.get("reference_budget", item.get("code_c_lo", item.get("c_lo", 0)))))
        result[name] = {
            "criticality": str(item.get("criticality", "LO")).upper(),
            "reference": reference,
            "floor": int(item.get("budget_floor", item.get("minimum_budget", reference))),
            "upper": int(item.get("action_hard_upper", item.get("certified_upper_bound", item.get("code_c_hi", item.get("c_hi", reference))))),
        }
    return result


def _normalized_action(row: dict, default_ratio: float = 0.02) -> tuple[str | None, str, float, int] | None:
    if bool(row.get("is_noop")):
        return None
    if row.get("increase_task") is not None:
        return str(row["increase_task"]), "increase", float(row.get("increase_ratio", default_ratio)), int(row.get("minimum_increment", 1))
    decreases = row.get("decrease_tasks")
    if isinstance(decreases, (list, tuple)) and decreases:
        return str(decreases[0]), "decrease", float(row.get("decrease_ratio", default_ratio)), int(row.get("minimum_increment", 1))
    task = row.get("task_id", row.get("target_task"))
    direction = str(row.get("operation", row.get("direction", ""))).lower()
    if task is None or direction not in {"increase", "decrease"}:
        return None
    return str(task), direction, float(row.get("ratio", default_ratio)), int(row.get("minimum_increment", 1))


def _round_candidate(current: int, *, direction: str, ratio: float, minimum_delta: int, nearest: bool) -> int:
    import math
    raw = current * (1.0 + ratio if direction == "increase" else 1.0 - ratio)
    rounded = int(round(raw)) if nearest else (math.ceil(raw) if direction == "increase" else math.floor(raw))
    return max(rounded, current + minimum_delta) if direction == "increase" else min(rounded, current - minimum_delta)


def _bind_c2_rounding_target(mutation: dict, scoped: list[dict]) -> None:
    target = mutation.setdefault("resolved_target", {})
    binding = _discover_symbolic_binding(target)
    actions_raw = json.loads(Path(binding["action_definitions_path"]).read_text(encoding="utf-8"))
    tasks_raw = json.loads(Path(binding["taskset_path"]).read_text(encoding="utf-8"))
    actions = {int(item["action_id"]): item for item in _action_rows(actions_raw)}
    contracts = _task_contracts(tasks_raw)
    candidates = sorted(
        (row for row in scoped if row.get("action_ranking")),
        key=lambda item: (int(item.get("hout_hit_count", 0)), int(item.get("training_samples", 0))),
        reverse=True,
    )
    for row in candidates:
        ranking = [int(item) for item in row.get("action_ranking", ())]
        action_id = ranking[0]
        default_ratio_raw = binding.get("default_action_ratio", 0.02)
        if isinstance(default_ratio_raw, str) and "/" in default_ratio_raw:
            numerator, denominator = default_ratio_raw.split("/", 1)
            default_ratio = float(int(numerator) / int(denominator))
        else:
            default_ratio = float(default_ratio_raw)
        normalized = _normalized_action(actions.get(action_id, {}), default_ratio)
        if normalized is None:
            continue
        task_id, direction, ratio, minimum_delta = normalized
        contract = contracts.get(str(task_id))
        if contract is None:
            continue
        for current in range(int(contract["floor"]), int(contract["upper"]) + 1):
            normal = _round_candidate(current, direction=direction, ratio=ratio, minimum_delta=minimum_delta, nearest=False)
            nearest = _round_candidate(current, direction=direction, ratio=ratio, minimum_delta=minimum_delta, nearest=True)
            if normal == nearest:
                continue
            lower = max(int(contract["floor"]), int(contract["reference"]) if contract["criticality"] == "HI" else int(contract["floor"]))
            upper = int(contract["upper"])
            if lower <= normal <= upper and lower <= nearest <= upper:
                target.update({
                    "leaf_id": int(row["leaf_id"]),
                    "action_id": action_id,
                    "original_ranking": ranking,
                    "rounding_witness_current_budget": current,
                    "rounding_witness_ceil_floor": normal,
                    "rounding_witness_nearest": nearest,
                })
                return
    raise ValueError("C2_ROUNDING_DIFFERENCE_TARGET_UNRESOLVED")


def _discover_symbolic_binding(target: dict) -> dict:
    tree_path = Path(str(target["tree_path"])).resolve()
    variant_dir = tree_path.parent
    seed_dir = Path(str(target.get("seed_dir", variant_dir.parent))).resolve()
    action_path = next((p for p in (
        variant_dir / "action_definitions.json", seed_dir / "action_definitions.json"
    ) if p.is_file()), None)
    feature_path = next((p for p in (
        variant_dir / "feature_schema.json", variant_dir / "feature_names.json",
        seed_dir / "feature_schema.json", seed_dir / "feature_names.json",
    ) if p.is_file()), None)
    taskset_path = next((p for p in (
        seed_dir / "formal_inputs/code_taskset_canonical.json",
        variant_dir / "formal_inputs/code_taskset_canonical.json",
        seed_dir / "code_taskset_canonical.json", seed_dir / "taskset.json",
    ) if p.is_file()), None)
    if action_path is None or feature_path is None or taskset_path is None:
        raise ValueError(
            f"SYMBOLIC_ARTIFACTS_MISSING:action={action_path}:feature={feature_path}:taskset={taskset_path}"
        )
    scale = 1000
    fixed = variant_dir / "fixed_point_config.json"
    if fixed.is_file():
        raw = json.loads(fixed.read_text(encoding="utf-8"))
        scale = int(raw.get("config", raw).get("scale", scale))
    return {
        "taskset_path": str(taskset_path),
        "feature_schema_path": str(feature_path),
        "action_definitions_path": str(action_path),
        "feature_scale": scale,
        "default_action_ratio": "1/50",
    }


def _bind_symbolic_activation(mutation: dict, canonical: str) -> None:
    activation = mutation.get("activation")
    if not isinstance(activation, dict) or str(activation.get("mode", "")).lower() != "symbolic_auto":
        return
    target = mutation.get("resolved_target")
    if not isinstance(target, dict) or not {"tree_path", "leaf_id", "action_id"}.issubset(target):
        raise ValueError("SYMBOLIC_TARGET_NOT_RESOLVED")
    binding = _discover_symbolic_binding(target)
    formula_by_id = {
        "B2": "B2_NO_FIRST_VALID_DIFFERENCE",
        "B3": "B3_ALL_INVALID",
        "B4": "B4_GUARD_NECESSITY",
        "C2": "C2_ROUNDING_DIFFERENCE",
    }
    overlay_by_id = {
        "B2": "top1_valid_else_noop",
        "B3": "all_invalid_force_top1",
        "B4": "ranked_first_valid",
        "C2": "ranked_first_valid",
    }
    if canonical not in formula_by_id:
        raise ValueError(f"SYMBOLIC_FORMULA_NOT_MAPPED:{canonical}")
    binding["overlay_semantics"] = overlay_by_id[canonical]
    if canonical in {"B3"}:
        binding["overlay_unchecked_apply"] = True
    if canonical == "B4":
        binding["disabled_guard"] = target.get("catalog_key", target.get("guard_id", "selected_guard"))
        target.setdefault("disabled_guard_constraint", {"constant": True})
    if canonical == "C2":
        binding["overlay_rounding_mode"] = "nearest"
    activation.update({
        "binding": binding,
        "formula_kind": formula_by_id[canonical],
        "runtime_binding_factory": "nonvacuity_lab.activation.default_runtime_binding:build_default_runtime_binding",
    })


def _bind_hout_profile(config: dict, mutation: dict) -> None:
    mode = str(mutation.get("activation", {}).get("mode", "")).lower()
    if "hout" not in mode or mutation.get("hout_profile_id"):
        return
    seed = mutation.get("seed")
    if seed is None:
        return
    matches = [
        profile_id for profile_id in config.get("hout_profiles", {})
        if str(profile_id).startswith(f"s{seed}_")
    ]
    if len(matches) == 1:
        mutation["hout_profile_id"] = matches[0]


def _proof_run_roots(proof_root: Path) -> list[Path]:
    if not proof_root.is_dir():
        return []
    roots = set()
    for request in proof_root.rglob("proof_request.json"):
        parent = request.parent.parent if request.parent.name == "request" else request.parent
        roots.add(parent.resolve())
    if not roots and (proof_root / "proof_request.json").is_file():
        roots.add(proof_root.resolve())
    return sorted(roots)


def _candidate_dir(run_root: Path) -> Path:
    candidate = run_root / "candidate"
    return candidate if candidate.is_dir() else run_root


def _choose_base_run(runs: list[Path]) -> Path:
    if not runs:
        raise ValueError("PROOF_RUNS_NOT_FOUND")
    return next((p for p in runs if "s185" in p.as_posix().lower()), runs[0])


def _find_json_pointer(value, key_names: set[str], pointer: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{pointer}/{str(key).replace('~','~0').replace('/','~1')}"
            if key in key_names:
                return child, item
            found = _find_json_pointer(item, key_names, child)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_json_pointer(item, key_names, f"{pointer}/{index}")
            if found:
                return found
    return None


def _bundle_json_target(run_root: Path, patterns: tuple[str, ...], keys: set[str]) -> tuple[str, str, object]:
    candidate = _candidate_dir(run_root)
    files = [p for pattern in patterns for p in candidate.rglob(pattern)]
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found = _find_json_pointer(raw, keys)
        if found:
            pointer, value = found
            return "candidate/" + path.relative_to(candidate).as_posix(), pointer, value
    raise ValueError(f"BUNDLE_JSON_TARGET_NOT_FOUND:{patterns}:{keys}")


def _bind_integrity_mutation(mutation: dict, *, canonical: str, proof_root: Path, source_root: Path) -> None:
    runs = _proof_run_roots(proof_root)
    base = _choose_base_run(runs)
    mutation["reuse_source_bundle"] = str(base)
    parameters = mutation.setdefault("mutator", {}).setdefault("parameters", {})
    candidate = _candidate_dir(base)
    if canonical == "F1":
        target, pointer, value = _bundle_json_target(base, ("*integer_tree*.json",), {"threshold_int", "threshold"})
        parameters.update({"target_file": target, "json_pointer": pointer, "value": int(value) + 1})
    elif canonical == "F4":
        target, pointer, value = _bundle_json_target(base, ("*priority*.json", "proof_request.json"), {"priority_order"})
        if isinstance(value, list):
            pointer = pointer + "/0"
            value = "NONVACUITY_TAMPERED_TASK"
        else:
            value = "NONVACUITY_TAMPERED_PRIORITY"
        parameters.update({"target_file": target, "json_pointer": pointer, "value": value})
    elif canonical == "F5":
        from .audit.bundle_inventory import build_bundle_inventory, resolve_f5_witness_pointer
        target = resolve_f5_witness_pointer(build_bundle_inventory(candidate))
        parameters.update({
            "target_file": "candidate/" + target["artifact"],
            "json_pointer": target["json_pointer"], "value": "NONVACUITY_TAMPERED_WITNESS",
        })
    elif canonical == "F6":
        from .audit.bundle_inventory import build_bundle_inventory, resolve_f6_obligation_artifact
        target = resolve_f6_obligation_artifact(build_bundle_inventory(candidate))
        parameters.update({"target_file": "candidate/" + target["artifact"]})
    elif canonical in {"F2", "F3"}:
        base_files = sorted(candidate.rglob("*.json"))
        target_path = next((p for p in base_files if "proof_summary" in p.name), base_files[0] if base_files else None)
        if target_path is None:
            raise ValueError("CROSS_BUNDLE_TARGET_MISSING")
        if canonical == "F2":
            source_run = next((p for p in runs if _seed_token(p) != _seed_token(base)), None)
        else:
            source_run = next((p for p in runs if _variant_token(p) != _variant_token(base)), None)
        if source_run is None:
            raise ValueError(f"{canonical}_SOURCE_RUN_MISSING")
        source_candidate = _candidate_dir(source_run)
        relative = target_path.relative_to(candidate)
        source_file = source_candidate / relative
        if not source_file.is_file():
            source_file = next(iter(sorted(source_candidate.rglob(target_path.name))), None)
        if source_file is None or not source_file.is_file():
            raise ValueError(f"{canonical}_SOURCE_FILE_MISSING")
        parameters.update({
            "target_file": "candidate/" + relative.as_posix(),
            "source_file": str(source_file.resolve()),
        })
    elif canonical == "F7":
        target = source_root / "amc_py/rl/actions.py"
        before = '    if mode == "ceil_floor":\n'
        if target.read_text(encoding="utf-8").count(before) != 1:
            raise ValueError("F7_SOURCE_SNIPPET_NOT_UNIQUE")
        parameters.update({
            "target_file": "amc_py/rl/actions.py",
            "before_snippet": before,
            "after_snippet": '    if mode in {"ceil_floor"}:\n',
        })
    else:
        raise ValueError(f"UNKNOWN_INTEGRITY_MUTATION:{canonical}")


def _seed_token(path: Path) -> str:
    import re
    match = re.search(r"(?:^|[/_])s?(\d{2,})(?:$|[/_])", path.as_posix().lower())
    return match.group(1) if match else path.name


def _variant_token(path: Path) -> str:
    text = path.as_posix().lower()
    for value in ("best_overall", "compact", "best_balanced", "balanced", "best_performance"):
        if value in text:
            return value
    return "unknown"


def _v2_resolution_gaps(mutation: dict) -> list[str]:
    gaps: list[str] = []
    mutation_class = str(mutation.get("mutation_class", ""))
    target = mutation.get("resolved_target")
    seed = mutation.get("seed")
    if seed is not None:
        for field in ("seed_dir", "tree_path", "tree_sha256"):
            if not isinstance(target, dict) or target.get(field) in (None, ""):
                gaps.append(f"resolved_target.{field}")
    if mutation_class == "DANGEROUS_TOP1":
        for field in ("leaf_id", "action_id", "original_ranking"):
            if not isinstance(target, dict) or target.get(field) in (None, ""):
                gaps.append(f"resolved_target.{field}")
    mutator = mutation.get("mutator", {})
    parameters = mutator.get("parameters", {}) if isinstance(mutator, dict) else {}
    kind = str(mutator.get("kind", "")) if isinstance(mutator, dict) else ""
    if kind == "coherent_source_patch":
        patches = parameters.get("patches")
        if not patches:
            gaps.append("mutator.parameters.patches")
        else:
            roles = [str(item.get("role", "")) for item in patches if isinstance(item, dict)]
            for role in missing_roles(parameters.get("semantic_change_id"), roles):
                gaps.append(f"mutator.parameters.patches.role:{role}")
    if kind == "action_step" and seed is None:
        gaps.append("seed")
    if mutation_class == "ENVELOPE_GRADIENT" and not mutation.get("metadata", {}).get("bundle_roots"):
        gaps.append("metadata.bundle_roots")
    if mutation_class.startswith("BUNDLE_") or mutation_class == "SOURCE_BINDING_TAMPER":
        if not mutation.get("reuse_source_bundle"):
            gaps.append("reuse_source_bundle")
        if not parameters.get("target_file"):
            gaps.append("mutator.parameters.target_file")
    mode = str(mutation.get("activation", {}).get("mode", "")).lower()
    if "hout" in mode and not mutation.get("hout_profile_id"):
        gaps.append("hout_profile_id")
    if mode == "symbolic_auto":
        activation = mutation.get("activation", {})
        for field in ("binding", "formula_kind", "runtime_binding_factory"):
            if activation.get(field) in (None, ""):
                gaps.append(f"activation.{field}")
        for field in ("leaf_id", "action_id"):
            if not isinstance(target, dict) or target.get(field) in (None, ""):
                gaps.append(f"resolved_target.{field}")
    return sorted(set(gaps))

def seal_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    write_resolved_campaign(path, config)
    return json.loads(path.read_text(encoding="utf-8"))
