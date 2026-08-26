from __future__ import annotations

import json
from pathlib import Path

from formal_toolchain.core.registry import load_registry, registry_fingerprint


ROOT = Path(__file__).resolve().parents[3]


def _load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_theory_and_migration_manifests_bind_current_registry():
    registry = load_registry(ROOT / "formal_toolchain/specs/obligation_registry.json")
    current = registry_fingerprint(registry)
    theory = _load("formal_toolchain/theory/theory_manifest.json")
    migration = _load("formal_toolchain/specs/migration_manifest.json")

    assert theory["registry_fingerprint"] == current
    assert migration["registry_fingerprint"] == current
    assert migration["previous_registry_fingerprint"] != current
    assert theory["library_version"] == "p0-theory-r12"
    assert migration["migration_id"] == "r12-explicit-noop-controller-n3-registry-sync"
