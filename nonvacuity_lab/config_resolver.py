from __future__ import annotations

import copy
import json
from pathlib import Path

from .canonical import file_hash, tree_hash
from .config_io import write_resolved_campaign
from .resolver.dangerous_top1 import resolve_dangerous_top1


def resolve_campaign(template_path: Path, audit_root: Path, source_root: Path, output_path: Path) -> dict:
    template = json.loads(template_path.read_text(encoding="utf-8"))
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
    receipt.write_text(json.dumps({"schema_version": "nonvacuity_resolver_receipt_v1", "template_sha256": file_hash(template_path), "audit_root_sha256": tree_hash(audit_root), "source_root_sha256": tree_hash(source_root), "records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resolved["resolver_receipt"] = {"path": str(receipt.resolve()), "sha256": file_hash(receipt)}
    write_resolved_campaign(output_path, resolved)
    return resolved


def seal_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    write_resolved_campaign(path, config)
    return json.loads(path.read_text(encoding="utf-8"))
