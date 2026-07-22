from __future__ import annotations

from pathlib import Path


def test_global_audit_before_exists():
    path = Path(__file__).resolve().parents[3] / "build" / "formal" / "global_audit_before.json"
    assert path.is_file()


def test_global_audit_before_contains_registry_snapshot():
    path = Path(__file__).resolve().parents[3] / "build" / "formal" / "global_audit_before.json"
    data = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert data["registry"]["total_entries"] == 93
    assert data["registry"]["active_required"] == 89
    assert data["registry"]["deprecated"] == 3
    assert data["registry"]["conditional"] == 1

