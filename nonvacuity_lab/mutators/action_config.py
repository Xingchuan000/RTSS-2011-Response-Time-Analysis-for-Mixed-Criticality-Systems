"""Single-field JSON action/runtime configuration mutations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..canonical import canonical_json_hash, file_hash, json_pointer_diff
from .base import MutationContext, MutationResult, PreflightResult


class JsonPatchMutation:
    def preflight(self, context: MutationContext) -> PreflightResult:
        try:
            path = self._target(context)
            data = json.loads(path.read_text(encoding="utf-8"))
            pointer = str(context.parameters["json_pointer"])
            before = get_pointer(data, pointer)
            expected = context.parameters.get("expected_before", before)
            if before != expected:
                raise ValueError(f"JSON pointer 原值不匹配: {pointer}")
            return PreflightResult("PASS", {"target": str(path), "before": before})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return PreflightResult("FAIL", {"reason": str(exc)})

    def apply(self, context: MutationContext) -> MutationResult:
        preflight = self.preflight(context)
        if preflight.status != "PASS":
            raise ValueError(str(preflight.details.get("reason")))
        path = self._target(context)
        before_data = json.loads(path.read_text(encoding="utf-8"))
        after_data = json.loads(json.dumps(before_data))
        pointer = str(context.parameters["json_pointer"])
        set_pointer(after_data, pointer, context.parameters["value"])
        diff = json_pointer_diff(before_data, after_data)
        path.write_text(
            json.dumps(after_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        valid = len(diff) == 1 and diff[0]["pointer"] == pointer
        root = context.mutated_seed or context.source_overlay
        assert root is not None
        return MutationResult(
            status="PASS" if valid else "FAIL",
            before_hash=canonical_json_hash(before_data),
            after_hash=canonical_json_hash(after_data),
            changed_files=(str(path.relative_to(root)),),
            changed_pointers=tuple(row["pointer"] for row in diff),
            semantic_change_count=len(diff),
            details={"declared_pointer": pointer},
        )

    def verify_single_change(self, result: MutationResult) -> PreflightResult:
        valid = result.status == "PASS" and result.semantic_change_count == 1
        return PreflightResult("PASS" if valid else "FAIL", result.to_dict())

    @staticmethod
    def _target(context: MutationContext) -> Path:
        relative = Path(str(context.parameters["target_file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("target_file 必须是无 .. 的相对路径")
        root_kind = str(context.parameters.get("root", "seed"))
        root = context.mutated_seed if root_kind == "seed" else context.source_overlay
        if root is None:
            raise ValueError(f"缺少 mutation root: {root_kind}")
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("target_file 逃逸 mutation root")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"target_file 不存在或为软链接: {path}")
        return path


class ActionConfigMutation(JsonPatchMutation):
    """Apply one conceptual action semantic change to all declared pointers."""

    def preflight(self, context: MutationContext) -> PreflightResult:
        patches = context.parameters.get("patches")
        if patches is None:
            return super().preflight(context)
        if not isinstance(patches, list) or not patches:
            return PreflightResult("FAIL", {"reason": "patches 必须为非空 array"})
        try:
            path = self._target(context)
            data = json.loads(path.read_text(encoding="utf-8"))
            pointers = []
            for patch in patches:
                if not isinstance(patch, dict):
                    raise ValueError("每个 action patch 必须为 object")
                pointer = str(patch["json_pointer"])
                before = get_pointer(data, pointer)
                if "expected_before" in patch and before != patch["expected_before"]:
                    raise ValueError(f"action patch 原值不匹配: {pointer}")
                pointers.append(pointer)
            if len(pointers) != len(set(pointers)):
                raise ValueError("action patch pointer 不得重复")
            return PreflightResult("PASS", {"target": str(path), "pointers": pointers})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return PreflightResult("FAIL", {"reason": str(exc)})

    def apply(self, context: MutationContext) -> MutationResult:
        patches = context.parameters.get("patches")
        if patches is None:
            return super().apply(context)
        preflight = self.preflight(context)
        if preflight.status != "PASS":
            raise ValueError(str(preflight.details.get("reason")))
        path = self._target(context)
        before_data = json.loads(path.read_text(encoding="utf-8"))
        after_data = json.loads(json.dumps(before_data))
        declared = set()
        for patch in patches:
            pointer = str(patch["json_pointer"])
            declared.add(pointer)
            set_pointer(after_data, pointer, patch["value"])
        diff = json_pointer_diff(before_data, after_data)
        actual = {row["pointer"] for row in diff}
        path.write_text(
            json.dumps(after_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        root_kind = str(context.parameters.get("root", "seed"))
        root = context.mutated_seed if root_kind == "seed" else context.source_overlay
        assert root is not None
        changed_files = [str(path.relative_to(root))]
        hash_updates = []
        for update in context.parameters.get("hash_updates", ()):
            update_path = (root / Path(str(update["target_file"]))).resolve()
            if root.resolve() not in update_path.parents or not update_path.is_file():
                raise ValueError("hash update target 非法")
            update_data = json.loads(update_path.read_text(encoding="utf-8"))
            replacement = (
                canonical_json_hash(after_data)
                if update.get("hash_kind") == "canonical_json"
                else file_hash(path)
            )
            set_pointer(update_data, str(update["json_pointer"]), replacement)
            update_path.write_text(
                json.dumps(update_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            changed_files.append(str(update_path.relative_to(root)))
            hash_updates.append(
                {
                    "file": str(update_path.relative_to(root)),
                    "pointer": str(update["json_pointer"]),
                    "value": replacement,
                }
            )
        valid = actual == declared
        return MutationResult(
            status="PASS" if valid else "FAIL",
            before_hash=canonical_json_hash(before_data),
            after_hash=canonical_json_hash(after_data),
            changed_files=tuple(changed_files),
            changed_pointers=tuple(sorted(actual)),
            semantic_change_count=1,
            artifact_manifest_validation="PASS" if hash_updates else "NOT_APPLICABLE",
            details={
                "semantic_group": context.parameters.get("semantic_group"),
                "declared_pointers": sorted(declared),
                "hash_updates": hash_updates,
            },
        )


class ActionStepMutation:
    """C1 exact inc/dec ratio mutation across copied recipe and tree schema."""

    def preflight(self, context: MutationContext) -> PreflightResult:
        try:
            if context.mutated_seed is None:
                raise ValueError("action step mutation 需要 copied seed")
            direction = str(context.parameters.get("direction", "inc_only"))
            if direction not in {"inc_only", "dec_only", "both"}:
                raise ValueError("direction 必须为 inc_only/dec_only/both")
            before = float(context.parameters.get("before_ratio", 0.02))
            after = float(context.parameters.get("after_ratio", 0.05))
            if before <= 0 or after <= 0 or before == after:
                raise ValueError("action ratio 必须为不同的正数")
            variant = str(context.parameters.get("tree_variant", "best_overall"))
            for path in (
                context.mutated_seed / variant / "action_definitions.json",
                context.mutated_seed / variant / "artifact_manifest.json",
                context.mutated_seed / "formal_inputs" / "target_recipe.json",
            ):
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"C1 input 缺失或为软链接: {path}")
            return PreflightResult(
                "PASS",
                {"direction": direction, "before_ratio": before, "after_ratio": after},
            )
        except (OSError, ValueError, TypeError) as exc:
            return PreflightResult("FAIL", {"reason": str(exc)})

    def apply(self, context: MutationContext) -> MutationResult:
        preflight = self.preflight(context)
        if preflight.status != "PASS":
            raise ValueError(str(preflight.details.get("reason")))
        assert context.mutated_seed is not None
        direction = str(context.parameters.get("direction", "inc_only"))
        before_ratio = float(context.parameters.get("before_ratio", 0.02))
        after_ratio = float(context.parameters.get("after_ratio", 0.05))
        variant = str(context.parameters.get("tree_variant", "best_overall"))
        action_path = context.mutated_seed / variant / "action_definitions.json"
        recipe_path = context.mutated_seed / "formal_inputs" / "target_recipe.json"
        manifest_path = context.mutated_seed / variant / "artifact_manifest.json"
        action_before = json.loads(action_path.read_text(encoding="utf-8"))
        recipe_before = json.loads(recipe_path.read_text(encoding="utf-8"))
        action_after = json.loads(json.dumps(action_before))
        recipe_after = json.loads(json.dumps(recipe_before))
        action_rows = _action_rows(action_after)
        expected_rows = recipe_after["kwargs"]["expected_action_definitions"]
        changed_action_ids = []
        for row, expected in zip(action_rows, expected_rows, strict=True):
            action_id = int(row["action_id"])
            changed = False
            if direction in {"inc_only", "both"} and row.get("increase_task") is not None:
                if float(row["increase_ratio"]) != before_ratio:
                    raise ValueError(f"C1 inc ratio 原值不匹配: action {action_id}")
                row["increase_ratio"] = after_ratio
                expected["increase_ratio"] = after_ratio
                changed = True
            if direction in {"dec_only", "both"} and row.get("decrease_tasks"):
                if float(row["decrease_ratio"]) != before_ratio:
                    raise ValueError(f"C1 dec ratio 原值不匹配: action {action_id}")
                row["decrease_ratio"] = after_ratio
                expected["decrease_ratio"] = after_ratio
                changed = True
            if changed:
                changed_action_ids.append(action_id)
        runtime_args = recipe_after["kwargs"]["runtime_args"]
        if direction in {"inc_only", "both"}:
            runtime_args["budget_increase_ratio"] = after_ratio
        if direction in {"dec_only", "both"}:
            runtime_args["budget_decrease_ratio"] = after_ratio
        action_path.write_text(
            json.dumps(action_after, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        recipe_path.write_text(
            json.dumps(recipe_after, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = manifest.get("file_hashes", manifest.get("files"))
        if not isinstance(hashes, dict):
            raise ValueError("artifact manifest 缺少 action hash map")
        entry = hashes.get("action_definitions.json")
        digest = file_hash(action_path)
        if isinstance(entry, dict):
            entry["sha256"] = digest
        else:
            hashes["action_definitions.json"] = digest
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        details = {
            "direction": direction,
            "before_ratio": before_ratio,
            "after_ratio": after_ratio,
            "changed_action_ids": changed_action_ids,
            "action_diff": json_pointer_diff(action_before, action_after),
            "recipe_diff": json_pointer_diff(recipe_before, recipe_after),
        }
        return MutationResult(
            status="PASS" if changed_action_ids else "FAIL",
            before_hash=canonical_json_hash(
                {"actions": action_before, "recipe": recipe_before}
            ),
            after_hash=canonical_json_hash(
                {"actions": action_after, "recipe": recipe_after}
            ),
            changed_files=(
                str(action_path.relative_to(context.mutated_seed)),
                str(recipe_path.relative_to(context.mutated_seed)),
                str(manifest_path.relative_to(context.mutated_seed)),
            ),
            semantic_change_count=1,
            artifact_manifest_validation="PASS",
            details=details,
        )

    def verify_single_change(self, result: MutationResult) -> PreflightResult:
        valid = (
            result.status == "PASS"
            and result.semantic_change_count == 1
            and result.artifact_manifest_validation == "PASS"
        )
        return PreflightResult("PASS" if valid else "FAIL", result.to_dict())


def get_pointer(value: Any, pointer: str) -> Any:
    current = value
    for token in _tokens(pointer):
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def set_pointer(value: Any, pointer: str, replacement: Any) -> None:
    tokens = _tokens(pointer)
    if not tokens:
        raise ValueError("不允许替换 JSON 根")
    current = value
    for token in tokens[:-1]:
        current = current[int(token)] if isinstance(current, list) else current[token]
    final = tokens[-1]
    if isinstance(current, list):
        current[int(final)] = replacement
    else:
        if final not in current:
            raise KeyError(final)
        current[final] = replacement


def _tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer 必须以 / 开头")
    if pointer == "/":
        return [""]
    return [item.replace("~1", "/").replace("~0", "~") for item in pointer[1:].split("/")]


def _action_rows(value: Any) -> list[dict[str, Any]]:
    rows = value.get("actions") if isinstance(value, dict) else value
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("action definitions 必须为 object array")
    return rows
