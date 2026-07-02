"""VIPER tree artifact 的保存与加载。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.tree import export_text

from amc_py.viper.tree_policy import TreeBudgetPolicy


def export_tree_rules_text(classifier, feature_names: tuple[str, ...]) -> str:
    """导出带特征名的可读规则文本。"""

    return export_text(classifier, feature_names=list(feature_names))


def save_tree_policy_artifact(
    output_dir: Path,
    *,
    classifier,
    metadata: dict[str, object],
    feature_names: tuple[str, ...],
    action_definitions: list[dict[str, object]],
) -> Path:
    """把一棵树保存成计划要求的 artifact 目录。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output_dir / "model.joblib")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(list(feature_names), handle, ensure_ascii=False, indent=2)
    with (output_dir / "action_definitions.json").open("w", encoding="utf-8") as handle:
        json.dump(action_definitions, handle, ensure_ascii=False, indent=2)
    with (output_dir / "rules.txt").open("w", encoding="utf-8") as handle:
        handle.write(export_tree_rules_text(classifier, feature_names))
    return output_dir


def load_tree_policy_artifact(tree_artifact_dir: Path) -> TreeBudgetPolicy:
    """从 artifact 目录恢复 tree policy。"""

    classifier = joblib.load(tree_artifact_dir / "model.joblib")
    with (tree_artifact_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (tree_artifact_dir / "feature_names.json").open("r", encoding="utf-8") as handle:
        feature_names = tuple(json.load(handle))
    with (tree_artifact_dir / "action_definitions.json").open("r", encoding="utf-8") as handle:
        action_definitions = list(json.load(handle))
    return TreeBudgetPolicy(
        classifier=classifier,
        metadata=metadata,
        feature_names=feature_names,
        action_definitions=action_definitions,
    )
