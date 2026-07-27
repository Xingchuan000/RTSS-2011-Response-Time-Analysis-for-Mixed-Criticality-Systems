"""A1/A2 dangerous top-1 ranking mutation on copied tree artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from amc_py.viper.integer_tree import integer_tree_hash, load_integer_tree_json

from ..canonical import canonical_json_hash, file_hash, json_pointer_diff
from .base import MutationContext, MutationResult, PreflightResult


class DangerousTop1Mutation:
    def preflight(self, context: MutationContext) -> PreflightResult:
        try:
            artifact_dir = self._artifact_dir(context)
            tree = _read_json(artifact_dir / "integer_tree.json")
            leaf = self.choose_leaf(tree, context.parameters)
            action = self.choose_dangerous_action(leaf, context.parameters)
            if action in leaf["action_ranking"] and int(leaf["action_ranking"][0]) == action:
                raise ValueError("dangerous action 已经是 raw top-1")
            return PreflightResult("PASS", {"leaf_id": leaf["node_id"], "action_id": action})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return PreflightResult("FAIL", {"reason": str(exc)})

    def apply(self, context: MutationContext) -> MutationResult:
        preflight = self.preflight(context)
        if preflight.status != "PASS":
            raise ValueError(str(preflight.details.get("reason", "tree mutation preflight failed")))
        artifact_dir = self._artifact_dir(context)
        tree_path = artifact_dir / "integer_tree.json"
        manifest_path = artifact_dir / "artifact_manifest.json"
        before_tree = _read_json(tree_path)
        after_tree = json.loads(json.dumps(before_tree))
        leaf = self.choose_leaf(after_tree, context.parameters)
        action = self.choose_dangerous_action(leaf, context.parameters)
        ranking = [int(item) for item in leaf["action_ranking"]]
        ranking.remove(action)
        leaf["action_ranking"] = [action, *ranking]
        leaf["raw_action_id"] = action
        before_hash = canonical_json_hash(before_tree)
        after_hash = canonical_json_hash(after_tree)
        _write_json(tree_path, after_tree, compact=True)
        parsed_tree = load_integer_tree_json(tree_path)
        semantic_tree_hash = integer_tree_hash(parsed_tree)

        manifest = _read_json(manifest_path)
        tree_file_hash = file_hash(tree_path)
        hashes = manifest.get("file_hashes", manifest.get("files"))
        if not isinstance(hashes, dict):
            raise ValueError("artifact manifest 缺少 file_hashes/files")
        entry = hashes.get("integer_tree.json")
        if isinstance(entry, dict):
            entry["sha256"] = tree_file_hash
        else:
            hashes["integer_tree.json"] = tree_file_hash
        if "integer_tree_file_hash" in manifest:
            manifest["integer_tree_file_hash"] = tree_file_hash
        if "integer_tree_hash" in manifest:
            manifest["integer_tree_hash"] = semantic_tree_hash
        _write_json(manifest_path, manifest)

        diff = json_pointer_diff(before_tree, after_tree)
        leaf_index = next(
            index
            for index, item in enumerate(after_tree["leaves"])
            if int(item["node_id"]) == int(leaf["node_id"])
        )
        allowed = {
            f"/leaves/{leaf_index}/action_ranking/{index}"
            for index in range(len(leaf["action_ranking"]))
        } | {f"/leaves/{leaf_index}/raw_action_id"}
        unexpected = [row["pointer"] for row in diff if row["pointer"] not in allowed]
        status = "PASS" if not unexpected and before_hash != after_hash else "FAIL"
        return MutationResult(
            status=status,
            before_hash=before_hash,
            after_hash=after_hash,
            changed_files=(
                str(tree_path.relative_to(context.mutated_seed)),
                str(manifest_path.relative_to(context.mutated_seed)),
            ),
            changed_pointers=tuple(row["pointer"] for row in diff),
            semantic_change_count=1,
            artifact_manifest_validation="PASS" if _manifest_tree_hash(manifest) == tree_file_hash else "FAIL",
            rollback_information={"original_tree_hash": before_hash},
            details={
                "leaf_id": int(leaf["node_id"]),
                "action_id": action,
                "unexpected_pointers": unexpected,
                "ranking_only": True,
                "training_counts_preserved": True,
            },
        )

    def verify_single_change(self, result: MutationResult) -> PreflightResult:
        valid = (
            result.status == "PASS"
            and result.semantic_change_count == 1
            and result.artifact_manifest_validation == "PASS"
            and result.before_hash != result.after_hash
            and not result.details.get("unexpected_pointers")
        )
        return PreflightResult("PASS" if valid else "FAIL", result.to_dict())

    def choose_leaf(
        self,
        tree: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        leaves = tree.get("leaves")
        if not isinstance(leaves, list) or not leaves:
            raise ValueError("integer tree 缺少 leaves")
        explicit = parameters.get("leaf_id")
        if explicit is not None:
            match = next(
                (leaf for leaf in leaves if int(leaf.get("node_id", -1)) == int(explicit)),
                None,
            )
            if match is None:
                raise ValueError(f"leaf_id 不存在: {explicit}")
            return match
        audit_rows = parameters.get("leaf_candidates", ())
        scores = {
            int(row["leaf_id"]): (
                int(row.get("hout_hit_count", row.get("coverage", 0))),
                int(row.get("baseline_reject_count", 0) > 0),
                int(row.get("fallback_count", 0)),
                int(row.get("target_scenario_hits", 0)),
                -int(row.get("guard_complexity", 0)),
            )
            for row in audit_rows
            if isinstance(row, Mapping) and "leaf_id" in row
        }
        return max(
            leaves,
            key=lambda leaf: scores.get(
                int(leaf.get("node_id", -1)),
                (int(leaf.get("n_node_samples", 0)), 0, 0, 0, 0),
            ),
        )

    def choose_dangerous_action(
        self,
        leaf: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> int:
        explicit = parameters.get("action_id")
        ranking = [int(item) for item in leaf.get("action_ranking", ())]
        if not ranking:
            raise ValueError("leaf 缺少 action_ranking")
        if explicit is not None:
            action = int(explicit)
            if action not in ranking:
                raise ValueError(f"action_id 不在 ranking: {action}")
            return action
        candidates = parameters.get("dangerous_actions", ())
        if candidates:
            scored = sorted(
                (
                    (
                        int(row.get("reduces_hi_budget", False)),
                        int(row.get("limiting_hi_interference", False)),
                        int(row.get("exceeds_envelope", False)),
                        int(row.get("triggers_safety_checker", False)),
                        int(row.get("all_invalid_difference", False)),
                        int(row["action_id"]),
                    )
                    for row in candidates
                    if isinstance(row, Mapping) and int(row["action_id"]) in ranking
                ),
                reverse=True,
            )
            if scored:
                return scored[0][-1]
        raise ValueError("必须显式提供 action_id 或 dangerous_actions 证据")

    @staticmethod
    def _artifact_dir(context: MutationContext) -> Path:
        if context.mutated_seed is None:
            raise ValueError("tree mutation 需要 copied mutated_seed")
        variant = str(context.parameters.get("tree_variant", "best_overall"))
        artifact = context.mutated_seed / variant
        if not artifact.is_dir() or artifact.is_symlink():
            raise ValueError(f"tree artifact 目录无效: {artifact}")
        return artifact


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    kwargs = {"ensure_ascii": False, "sort_keys": True}
    if not compact:
        kwargs["indent"] = 2
    path.write_text(json.dumps(value, **kwargs) + "\n", encoding="utf-8")


def _manifest_tree_hash(manifest: Mapping[str, Any]) -> str | None:
    hashes = manifest.get("file_hashes", manifest.get("files", {}))
    entry = hashes.get("integer_tree.json") if isinstance(hashes, Mapping) else None
    return str(entry.get("sha256")) if isinstance(entry, Mapping) else str(entry)
