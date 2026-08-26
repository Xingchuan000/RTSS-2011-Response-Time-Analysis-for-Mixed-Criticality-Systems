from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact


FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_p0"
ARTIFACT_FILES = (
    "artifact_manifest.json",
    "integer_tree.json",
    "feature_names.json",
    "action_definitions.json",
    "fixed_point_config.json",
    "metadata.json",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _artifact(tmp_path: Path, action_dim: int, *, broken_ranking: str | None = None) -> Path:
    root = tmp_path / f"artifact_{action_dim}"
    root.mkdir()
    for name in ARTIFACT_FILES:
        shutil.copy2(FIXTURE / name, root / name)

    actions = [
        {"action_id": index, "direction": "increase", "target_task": f"T{index}", "is_noop": False}
        for index in range(action_dim)
    ]
    if action_dim == 25:
        actions[-1] = {"action_id": 24, "direction": "noop", "target_task": None, "is_noop": True}
    _write_json(root / "action_definitions.json", {"actions": actions})

    tree = json.loads((root / "integer_tree.json").read_text(encoding="utf-8"))
    tree["action_dim"] = action_dim
    for leaf in tree["leaves"]:
        leaf["action_ranking"] = list(range(action_dim))
    if broken_ranking == "missing":
        tree["leaves"][0]["action_ranking"][-1] = action_dim - 2
    elif broken_ranking == "duplicate":
        tree["leaves"][0]["action_ranking"][1] = tree["leaves"][0]["action_ranking"][0]
    _write_json(root / "integer_tree.json", tree)

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if action_dim == 25:
        metadata["explicit_noop"] = True
    _write_json(root / "metadata.json", metadata)

    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ARTIFACT_FILES
        if name != "artifact_manifest.json"
    }
    _write_json(root / "artifact_manifest.json", manifest)
    return root


def test_legacy_24_action_tree_still_loads(tmp_path: Path) -> None:
    inventory = inspect_tree_artifact(_artifact(tmp_path, 24), expected_state_dim=28)
    assert inventory["action_dim"] == 24


def test_explicit_noop_25_action_tree_loads(tmp_path: Path) -> None:
    inventory = inspect_tree_artifact(_artifact(tmp_path, 25), expected_state_dim=28)
    assert inventory["action_dim"] == 25


@pytest.mark.parametrize("expected,actual", [(25, 24), (24, 25)])
def test_formal_target_dimension_mismatch_fails_closed(tmp_path: Path, expected: int, actual: int) -> None:
    with pytest.raises(ValueError, match="action_definitions 数量与 expected_action_dim 不一致"):
        inspect_tree_artifact(
            _artifact(tmp_path, actual),
            expected_state_dim=28,
            expected_action_dim=expected,
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_ranking_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    with pytest.raises(ValueError):
        inspect_tree_artifact(
            _artifact(tmp_path, 25, broken_ranking=mutation),
            expected_state_dim=28,
        )
