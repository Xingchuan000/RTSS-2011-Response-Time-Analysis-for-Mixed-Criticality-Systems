from __future__ import annotations

from pathlib import Path


def test_global_audit_snapshot_is_generatable():
    from scripts.generate_global_audit_snapshot import build_snapshot
    root = Path(__file__).resolve().parents[3]
    snapshot = build_snapshot(root)
    assert isinstance(snapshot["registry_entry_count"], int) and snapshot["registry_entry_count"] > 0
    assert isinstance(snapshot["active_count"], int) and snapshot["active_count"] > 0
    assert isinstance(snapshot["required_count"], int)
    assert isinstance(snapshot["claim_closure"], list)
    assert "FINAL_CLAIM_COMPOSITION" in snapshot["claim_closure"]
    assert isinstance(snapshot["zero_predecessor_required"], list)

