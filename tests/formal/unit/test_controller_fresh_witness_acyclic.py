from __future__ import annotations

import json
from types import SimpleNamespace

from formal_toolchain.verifier import semantic_checkers


def _pass_predecessor(*, witness=None, artifact_hash="a" * 64):
    return {
        "obligation_status": "PASS",
        "artifact_hash": artifact_hash,
        "witness": dict(witness or {}),
    }


def test_controller_invisibility_fresh_witness_is_acyclic(monkeypatch):
    monkeypatch.setattr(semantic_checkers, "_p0_runtime", lambda raw: {})
    monkeypatch.setattr(
        semantic_checkers,
        "check_controller_invisibility",
        lambda runtime: {"status": "PASS", "detail": {"n3": "proved"}},
    )
    predecessors = {
        "CONTROLLER_WRITE_SET": _pass_predecessor(),
        "CONTROLLER_BOUNDARY": _pass_predecessor(
            witness={"preclosed_scheduler_consistent": True}
        ),
        "CONTROLLER_PATH_UNIQUENESS": _pass_predecessor(),
        "UPDATE_PAYLOAD_TOTALITY": _pass_predecessor(),
        "TOKEN_REFRESH_PROJECTION": _pass_predecessor(
            witness={"effective_frontier_preserved_if_preclosed": True}
        ),
    }
    result = semantic_checkers.verify_controller_invisibility(
        raw_inputs=SimpleNamespace(),
        verified_predecessors=predecessors,
        expected_context_hash="b" * 64,
    )
    assert result["status"] == "PASS"
    json.dumps(result)
    assert "witness" not in result["witness"]["source_candidate"]


def test_controller_postclosure_fresh_witness_is_acyclic(monkeypatch):
    monkeypatch.setattr(semantic_checkers, "_p0_runtime", lambda raw: {})
    monkeypatch.setattr(
        semantic_checkers,
        "check_controller_postclosure",
        lambda runtime: {"status": "PASS", "detail": {"closure": "proved"}},
    )
    predecessors = {
        "CONTROLLER_INVISIBILITY": _pass_predecessor(
            witness={"selected_action_n3_resolved": True},
            artifact_hash="c" * 64,
        )
    }
    result = semantic_checkers.verify_controller_postclosure(
        raw_inputs=SimpleNamespace(),
        verified_predecessors=predecessors,
        expected_context_hash="d" * 64,
    )
    assert result["status"] == "PASS"
    json.dumps(result)
    assert "witness" not in result["witness"]["source_candidate"]
