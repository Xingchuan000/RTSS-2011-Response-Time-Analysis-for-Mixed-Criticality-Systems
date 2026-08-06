from types import SimpleNamespace

import formal_toolchain.verifier.recompute as recompute


def _inputs():
    return SimpleNamespace(
        proof_route=SimpleNamespace(route="protected_prefix"),
        contexts={
            "semantic_context": {"hash": "semantic"},
            "invariant_context": {"hash": "invariant"},
        },
    )


def test_early_replay_returns_first_decisive_failure(monkeypatch):
    monkeypatch.setattr(recompute, "resolve_route", lambda route: object())
    monkeypatch.setattr(
        recompute,
        "expected_context_for_obligation",
        lambda obligation_id, contexts: "ctx-" + obligation_id,
    )

    def catalog(obligation_id, route_strategy=None):
        def checker(**kwargs):
            if obligation_id == "DEADLINE_OBSERVATION":
                return {
                    "status": "FAIL",
                    "route": "MODEL_CONFORMANCE_FAILED",
                    "code": "DEADLINE_OBSERVATION_VIOLATED",
                    "witness": {"fresh": True},
                }
            return {"status": "PASS"}
        return checker

    monkeypatch.setattr(recompute, "checker_for", catalog)
    result = recompute._replay_early_decisive_failure(
        inputs=_inputs(),
        active=["HI_NONTRUNCATION", "DEADLINE_OBSERVATION"],
        order=["HI_NONTRUNCATION", "DEADLINE_OBSERVATION"],
    )
    assert result == {
        "obligation_id": "DEADLINE_OBSERVATION",
        "route": "MODEL_CONFORMANCE_FAILED",
        "code": "DEADLINE_OBSERVATION_VIOLATED",
        "witness": {"fresh": True},
    }


def test_early_replay_does_not_turn_unresolved_or_exception_into_failure(monkeypatch):
    monkeypatch.setattr(recompute, "resolve_route", lambda route: object())
    monkeypatch.setattr(
        recompute,
        "expected_context_for_obligation",
        lambda obligation_id, contexts: "ctx",
    )

    def catalog(obligation_id, route_strategy=None):
        if obligation_id == "HI_NONTRUNCATION":
            return lambda **kwargs: {"status": "UNRESOLVED", "code": "NO_EVIDENCE"}
        return lambda **kwargs: (_ for _ in ()).throw(ValueError("bad input"))

    monkeypatch.setattr(recompute, "checker_for", catalog)
    assert recompute._replay_early_decisive_failure(
        inputs=_inputs(),
        active=["HI_NONTRUNCATION", "DEADLINE_OBSERVATION"],
        order=["HI_NONTRUNCATION", "DEADLINE_OBSERVATION"],
    ) is None
