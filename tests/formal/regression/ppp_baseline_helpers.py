from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


NONDETERMINISTIC_KEYS = {
    "created_at", "timestamp", "started_at", "finished_at",
    "duration_seconds", "elapsed_seconds", "pid", "cwd",
    "stdout_path", "stderr_path", "log_dir",
}

SEMANTIC_RESULT_KEYS = (
    "workflow_status", "result_status", "failure_route", "failure_code",
    "violated_obligation_id", "primary_claim", "profile", "proof_route",
    "target_kind", "tree_variant", "phase_k_map_refreshed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in NONDETERMINISTIC_KEYS
        }
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return value.name
    return value


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_ppp_baseline(bundle_root: Path) -> dict[str, Any]:
    root = Path(bundle_root).resolve()
    request_path = root / "request" / "proof_request.json"
    result_path = root / "proof_result.json"
    summary_path = root / "verified" / "proof_summary.json"
    if not request_path.is_file():
        raise FileNotFoundError(request_path)
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    tree_dir = root / str(request["tree_artifact_dir"])
    tree_files = {
        name: sha256_file(tree_dir / name)
        for name in ("integer_tree.json", "action_definitions.json", "feature_names.json", "fixed_point_config.json")
        if (tree_dir / name).is_file()
    }
    return {
        "schema_version": "ppp_clean_baseline_v1",
        "request_schema_version": request.get("schema_version"),
        "request_keys": sorted(request),
        "target_recipe_keys": sorted(dict(request.get("target_recipe", {}))),
        "target_recipe_kwarg_keys": sorted(dict(request.get("target_recipe", {}).get("kwargs", {}))),
        "request_semantic_hash": canonical_hash(request),
        "semantic_result": {key: result.get(key) for key in SEMANTIC_RESULT_KEYS if key in result},
        "summary_semantic_hash": canonical_hash(summary),
        "tree_file_hashes": tree_files,
        "expected_tree_file_sha256": request.get("expected_tree_file_sha256"),
        "outer_bundle_root": result.get("outer_bundle_root"),
    }
