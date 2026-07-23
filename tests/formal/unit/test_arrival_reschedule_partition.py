from __future__ import annotations

import pytest

from formal_toolchain.bridge.handler_decomposition import (
    HANDLER_COMPOSITION_CASES,
    prove_arrival_reschedule_partition,
)


def _valid_batch_certificate():
    return {
        "status": "PASS",
        "batch_nonempty": True,
        "one_release_substep_per_event": True,
        "release_keys_unique": True,
        "every_element_creates_fresh_job": True,
        "fold_certificate": {
            "status": "PASS",
            "iterable_is_finite": True,
            "body_called_once_per_element": True,
            "loop_has_no_early_exit": True,
            "element_case_partition_complete": True,
            "element_case_partition_exclusive": True,
            "fold_extends_job_map": True,
            "fold_preserves_relation": True,
        },
    }


def test_arrival_idle_is_not_an_executable_composition():
    assert "arrival_no_switch_idle" not in HANDLER_COMPOSITION_CASES
    assert "arrival_switch_s0_idle" not in HANDLER_COMPOSITION_CASES


def test_nonempty_release_fold_excludes_idle():
    pytest.importorskip("z3")
    result = prove_arrival_reschedule_partition(_valid_batch_certificate())
    assert result["status"] == "PASS"
    assert result["idle_unreachable"] is True
    assert result["keep_dispatch_exhaustive"] is True
    assert result["keep_dispatch_exclusive"] is True


def test_missing_nonempty_fold_cannot_authorize_partition():
    pytest.importorskip("z3")
    cert = _valid_batch_certificate()
    cert["batch_nonempty"] = False
    result = prove_arrival_reschedule_partition(cert)
    assert result["status"] == "UNRESOLVED"
