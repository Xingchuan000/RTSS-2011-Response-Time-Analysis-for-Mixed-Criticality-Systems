from __future__ import annotations

from pathlib import Path

from formal_toolchain.adapters.source_manifest import build_source_manifest


def test_lab_is_outside_formal_semantic_source_manifest():
    root = Path(__file__).resolve().parents[3]
    manifest = build_source_manifest(root)
    paths = {row["path"] for row in manifest["files"]}
    assert not any(path.startswith("nonvacuity_lab/") for path in paths)
    assert not any(path.startswith("configs/nonvacuity/") for path in paths)
