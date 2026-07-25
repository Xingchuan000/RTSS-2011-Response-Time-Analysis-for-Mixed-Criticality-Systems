import json

from formal_toolchain.verifier.recompute import _load_candidate


def test_candidate_root_failure_precedes_missing_component_contexts(tmp_path):
    failure = {
        "route": "MODEL_CONFORMANCE_FAILED",
        "code": "CANDIDATE_INPUT_REPLAY_FAILED",
        "exception_type": "ValueError",
        "message": "root cause",
    }
    (tmp_path / "candidate_failure.json").write_text(
        json.dumps({"schema_version": "candidate_failure_v1", "failure": failure}),
        encoding="utf-8",
    )
    contexts, candidates, error = _load_candidate(tmp_path, [], [])
    assert contexts is None
    assert candidates == {}
    assert error["code"] == "CANDIDATE_INPUT_REPLAY_FAILED"
    assert error["candidate_failure"]["message"] == "root cause"
