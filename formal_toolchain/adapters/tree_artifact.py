"""s185 整数树 artifact inventory。

本模块只读取文件并校验声明，不重排 feature、task 或 action，也不读取 HOUT。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file
from amc_py.viper.fixed_point import fixed_point_config_from_dict, fixed_point_config_hash

REQUIRED_FILES = (
    "artifact_manifest.json", "integer_tree.json", "feature_names.json",
    "action_definitions.json", "fixed_point_config.json", "metadata.json",
)


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def inspect_tree_artifact(artifact_dir: Path, *, expected_state_dim: int = 128,
                          expected_action_dim: int | None = None, expected_seed: int | None = None) -> dict[str, Any]:
    """返回确定性的 inventory；任何缺失或结构矛盾都会明确失败。"""
    artifact_dir = artifact_dir.resolve()
    missing = [name for name in REQUIRED_FILES if not (artifact_dir / name).is_file()]
    if missing:
        raise ValueError(f"缺少必需 artifact: {', '.join(missing)}")
    manifest = _read(artifact_dir / "artifact_manifest.json")
    # 真实 s185 artifact 使用 provider 的 ``file_hashes`` 字段，synthetic
    # fixture 使用 ``files``。两种已存在的 manifest schema 都是显式输入，
    # 这里只做字段名兼容，不放宽文件集合或 hash 校验。
    files = (manifest.get("files", manifest.get("file_hashes", manifest))
             if isinstance(manifest, dict) else {})
    hashes: dict[str, str] = {}
    for name in REQUIRED_FILES:
        actual = sha256_file(artifact_dir / name)
        declared = files.get(name) if isinstance(files, dict) else None
        if name != "artifact_manifest.json" and declared is None:
            raise ValueError(f"artifact manifest 缺少 hash: {name}")
        if isinstance(declared, dict):
            declared = declared.get("sha256")
        if declared is not None and declared != actual:
            raise ValueError(f"artifact hash 不匹配: {name}")
        hashes[name] = actual
    tree = _read(artifact_dir / "integer_tree.json")
    features = _read(artifact_dir / "feature_names.json")
    actions = _read(artifact_dir / "action_definitions.json")
    fixed = _read(artifact_dir / "fixed_point_config.json")
    metadata = _read(artifact_dir / "metadata.json")
    feature_names = features.get("feature_names", features) if isinstance(features, dict) else features
    action_table = actions.get("actions", actions) if isinstance(actions, dict) else actions
    if not isinstance(feature_names, list) or len(feature_names) != expected_state_dim:
        raise ValueError(f"feature count 必须为 {expected_state_dim}")
    if not isinstance(action_table, list):
        raise ValueError("action_definitions 必须为 list")
    if len(action_table) <= 0:
        raise ValueError("action_definitions 不允许为空")
    resolved_action_dim = (
        len(action_table)
        if expected_action_dim is None
        else int(expected_action_dim)
    )
    if len(action_table) != resolved_action_dim:
        raise ValueError(
            "action_definitions 数量与 expected_action_dim 不一致: "
            f"artifact={len(action_table)}, expected={resolved_action_dim}"
        )
    noop_ids = [
        index for index, action in enumerate(action_table)
        if isinstance(action, dict) and action.get("is_noop") is True
    ]
    explicit_noop_declared = (
        isinstance(metadata, dict) and metadata.get("explicit_noop") is True
    ) or bool(noop_ids)
    if explicit_noop_declared and len(noop_ids) != 1:
        raise ValueError("explicit-noop artifact 必须存在且仅存在一个 noop action")
    rankings = tree.get("rankings", tree.get("leaf_rankings")) if isinstance(tree, dict) else None
    expected_ids = set(range(resolved_action_dim))

    def check_ranking(ranking: Any, message: str) -> None:
        if not isinstance(ranking, list):
            raise ValueError(f"{message} 必须为 list")
        try:
            ranking_ids = [int(action_id) for action_id in ranking]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{message} 必须只包含整数 action-id") from exc
        if len(ranking_ids) != resolved_action_dim:
            raise ValueError(
                f"{message} 长度错误: got={len(ranking_ids)}, expected={resolved_action_dim}"
            )
        if set(ranking_ids) != expected_ids:
            raise ValueError(
                f"{message} 必须是完整 action-id 排列: "
                f"expected=0..{resolved_action_dim - 1}"
            )
        if len(set(ranking_ids)) != len(ranking_ids):
            raise ValueError(f"{message} 中存在重复 action-id")

    if rankings is not None:
        for ranking in rankings.values() if isinstance(rankings, dict) else rankings:
            check_ranking(ranking, "每个 leaf ranking")
    leaves = tree.get("leaves") if isinstance(tree, dict) else None
    if isinstance(leaves, list):
        for leaf in leaves:
            ranking = leaf.get("action_ranking") if isinstance(leaf, dict) else None
            if ranking is not None:
                check_ranking(ranking, "每个 leaf action_ranking")
    seed = metadata.get("taskset_seed") if isinstance(metadata, dict) else None
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(f"metadata taskset_seed 必须为 {expected_seed}，实际为 {seed!r}")
    state_dim = tree.get("state_dim") if isinstance(tree, dict) else None
    action_dim = tree.get("action_dim") if isinstance(tree, dict) else None
    if state_dim != expected_state_dim or action_dim != resolved_action_dim:
        raise ValueError(f"tree state/action dimension 必须为 {expected_state_dim}/{resolved_action_dim}，实际为 {state_dim}/{action_dim}")
    tree_fp_hash = tree.get("fixed_point_config_hash") if isinstance(tree, dict) else None
    fixed_data = fixed.get("config", fixed) if isinstance(fixed, dict) else fixed
    if not isinstance(fixed_data, dict):
        raise ValueError("fixed_point_config.json 必须是 object")
    expected_fp_hash = fixed_point_config_hash(fixed_point_config_from_dict(fixed_data))
    if tree_fp_hash != expected_fp_hash:
        raise ValueError("tree fixed-point semantic hash 与 fixed_point_config.json 不一致")
    return {"artifact_dir_name": artifact_dir.name, "files": hashes,
            "state_dim": state_dim, "action_dim": resolved_action_dim,
            "feature_count": len(feature_names), "action_count": len(action_table),
            "feature_names": list(feature_names), "action_definitions": list(action_table),
            "metadata_taskset_seed": seed, "fixed_point_config_hash": expected_fp_hash,
            "fixed_point_config": fixed_data}
