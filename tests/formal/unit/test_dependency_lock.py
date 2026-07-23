from __future__ import annotations

from pathlib import Path
import json

from formal_toolchain.adapters.runtime_manifest import (
    build_dependency_manifest,
    check_dependency_policy,
)


def test_dependency_lock_file_exists():
    lock_path = Path(__file__).resolve().parents[3] / "formal_toolchain" / "specs" / "proof_dependency_lock.json"
    assert lock_path.is_file()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["schema_version"] == "proof_dependency_lock_v1"
    assert "z3-solver" in lock["packages"]
    assert "jsonschema" in lock["packages"]


def test_dependency_manifest_contains_required_packages():
    manifest = build_dependency_manifest()
    packages = manifest.get("packages", {})
    assert "z3-solver" in packages
    assert "jsonschema" in packages


def test_check_dependency_policy_with_lock_rejects_mismatch():
    manifest = {"packages": {"z3-solver": None, "jsonschema": "1.0.0"}}
    lock = {"packages": {"z3-solver": "4.13.4.0", "jsonschema": "4.26.0"}}
    result = check_dependency_policy(manifest, lock=lock)
    assert result["status"] == "FAIL"
    assert result["code"] == "PROOF_DEPENDENCY_LOCK_MISMATCH"
    assert "z3-solver" in result["mismatches"]
    assert "jsonschema" in result["mismatches"]


def test_check_dependency_policy_with_exact_lock_passes():
    manifest = {"packages": {"z3-solver": "4.13.4.0", "jsonschema": "4.26.0"}}
    lock = {"packages": {"z3-solver": "4.13.4.0", "jsonschema": "4.26.0"}}
    result = check_dependency_policy(manifest, lock=lock)
    assert result["status"] == "PASS"


def test_check_dependency_policy_with_compatible_range_passes():
    manifest = {"packages": {"z3-solver": "4.16.0.0", "jsonschema": "4.26.0"}}
    lock = {"packages": {"z3-solver": ">=4.13,<5", "jsonschema": "4.26.0"}}
    result = check_dependency_policy(manifest, lock=lock)
    assert result["status"] == "PASS"


def test_check_dependency_policy_rejects_incompatible_major():
    manifest = {"packages": {"z3-solver": "5.0.0", "jsonschema": "4.26.0"}}
    lock = {"packages": {"z3-solver": ">=4.13,<5", "jsonschema": "4.26.0"}}
    result = check_dependency_policy(manifest, lock=lock)
    assert result["status"] == "FAIL"
    assert "z3-solver" in result["mismatches"]


def test_check_dependency_policy_missing_returns_fail():
    manifest = {"packages": {"numpy": "1.26.0"}}
    result = check_dependency_policy(manifest, lock=None)
    assert result["status"] == "FAIL"
    assert result["code"] == "DEPENDENCY_LOCK_INCOMPLETE"
