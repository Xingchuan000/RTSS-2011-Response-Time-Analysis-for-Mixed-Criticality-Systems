from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.verifier import bridge_replay
from formal_toolchain.verifier import bridge_proof_checker


def _write_case_map(tmp_path: Path) -> None:
    case_map = tmp_path / "request" / "inputs" / "formal_inputs"
    case_map.mkdir(parents=True, exist_ok=True)
    (case_map / "phase_k_case_map.json").write_text(
        json.dumps({"source_hash": "s" * 64}),
        encoding="utf-8",
    )


def _raw_inputs(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        workspace=tmp_path,
        source_root=tmp_path,
        source_manifest={"semantic_hash": "s" * 64},
        contexts={
            "semantic_context": {"hash": "a" * 64},
            "reference_context": {"hash": "b" * 64},
            "bridge_context": {"hash": "c" * 64},
        },
    )


def _prefix_candidate() -> dict[str, object]:
    theorem_hash = "d" * 64
    return {
        "artifact_schema_version": "synthetic_phase_k_v1",
        "obligation_id": "REFERENCE_PREFIX_EXTENSION",
        "obligation_status": "PASS",
        "certificate_context_hash": "c" * 64,
        "direct_predecessor_hashes": {
            "event_order": "e" * 64,
            "time_progress": "f" * 64,
        },
        "checker_id": "test",
        "checker_version": "1",
        "inputs": {
            "theorem_id": "REFERENCE_PREFIX_EXTENSION",
            "theorem": {
                "theorem_id": "REFERENCE_PREFIX_EXTENSION",
                "statement_hash": theorem_hash,
                "assumption_hash": "1" * 64,
            },
        },
        "witness": {
            "case_ids": ["case_a"],
            "coverage": {"status": "PASS", "missing": [], "unknown": [], "duplicate": [], "unresolved_cases": []},
        },
        "evidence": [{"status": "PASS"}],
        "failure": None,
    }


def _closed_candidate() -> dict[str, object]:
    theorem_hash = "d" * 64
    return {
        "artifact_schema_version": "synthetic_phase_k_v1",
        "obligation_id": "CLOSED_PREFIX_REFINEMENT",
        "obligation_status": "PASS",
        "certificate_context_hash": "c" * 64,
        "direct_predecessor_hashes": {
            "base_relation": "1" * 64,
            "same_timestamp": "2" * 64,
            "positive_time": "3" * 64,
            "controller_postclosure": "4" * 64,
            "event_projection": "5" * 64,
            "release_mapping": "6" * 64,
        },
        "checker_id": "test",
        "checker_version": "1",
        "source_hash": "7" * 64,
        "inputs": {
            "source_branch_count": 1,
            "branch_map_hash": "8" * 64,
            "release_mapping_hash": "9" * 64,
            "theorem_hash": theorem_hash,
        },
        "theorem_hash": theorem_hash,
        "witness": {
            "case_ids": ["case_a"],
            "coverage": {"status": "PASS", "missing": [], "unknown": [], "duplicate": [], "unresolved_cases": []},
            "transition_case_certificates": [
                {
                    "inputs": {"case_id": "case_a"},
                    "witness": {
                        "precondition_formula": "p",
                        "concrete_delta": "c",
                        "projected_reference_delta": "r",
                        "relation_preservation_formula": "s",
                    },
                }
            ],
        },
        "evidence": [{"status": "PASS"}],
        "failure": None,
    }


def test_prefix_extension_requires_verified_global_predecessors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_case_map(tmp_path)
    monkeypatch.setattr(bridge_proof_checker, "verify_obligation_certificate", lambda candidate: True)
    monkeypatch.setattr(
        bridge_proof_checker,
        "_THEORY_HASHES",
        {
            "REFERENCE_PREFIX_EXTENSION": {"statement_hash": "d" * 64, "assumption_hash": "1" * 64},
        },
    )

    monkeypatch.setattr(
        bridge_replay,
        "replay_all_transition_cases",
        lambda inputs: {"status": "PASS", "cases": [{"case_id": "case_a", "formula_hash": "x" * 64, "effect_ir_hash": "y" * 64}]},
    )

    result = bridge_proof_checker.verify_prefix_extension_proof_object(
        candidate=_prefix_candidate(),
        bridge_context_hash="c" * 64,
        contexts=_raw_inputs(tmp_path).contexts,
        predecessors={},
        raw_inputs=_raw_inputs(tmp_path),
        reference_taskset={"tasks": []},
    )

    assert result["status"] != "PASS"


def test_closed_prefix_still_requires_transition_case_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_case_map(tmp_path)
    monkeypatch.setattr(bridge_proof_checker, "verify_obligation_certificate", lambda candidate: True)
    monkeypatch.setattr(bridge_proof_checker, "REQUIRED_P0_CASE_IDS", ["case_a"])
    monkeypatch.setattr(
        bridge_proof_checker,
        "_THEORY_HASHES",
        {
            "CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT": {"statement_hash": "d" * 64},
        },
    )

    candidate = _closed_candidate()
    candidate["witness"].pop("transition_case_certificates", None)

    result = bridge_proof_checker.verify_closed_prefix_proof_object(
        candidate=candidate,
        bridge_context_hash="c" * 64,
        raw_inputs=_raw_inputs(tmp_path),
        reference_taskset={"tasks": []},
    )

    assert result["status"] != "PASS"
