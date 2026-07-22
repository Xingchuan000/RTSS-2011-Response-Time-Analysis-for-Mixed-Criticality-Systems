from __future__ import annotations

import pytest

from formal_toolchain.core.obligation_ids import LEGACY_PROTECTED_HI_IDS
from formal_toolchain.reference.rta_production import all_task_reference_rta
from formal_toolchain.reference.rta_replay import replay_all_task_rta
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.verifier.aggregator import reject_legacy_core_for_current_claim


def toy_taskset() -> ReferenceTaskset:
    tasks = (
        ReferenceTask(
            name="T1",
            period=10,
            deadline=10,
            c_lo=1,
            c_hi=1,
            criticality="LO",
            priority_index=0,
            code_c_lo=1,
            code_c_hi=1,
        ),
        ReferenceTask(
            name="T2",
            period=20,
            deadline=20,
            c_lo=1,
            c_hi=1,
            criticality="HI",
            priority_index=1,
            code_c_lo=1,
            code_c_hi=1,
        ),
    )
    return ReferenceTaskset(tasks, source_context_hash="0" * 64)


def test_legacy_protected_hi_pass_cannot_prove_current_claim():
    certs = {
        "PROTECTED_HI_RTA_ARITHMETIC": {"obligation_status": "PASS"},
        "PROTECTED_HI_SAFETY_COROLLARY": {"obligation_status": "PASS"},
    }
    result = reject_legacy_core_for_current_claim(certs)
    assert result is not None
    assert result["code"] == "LEGACY_PROTECTED_HI_CORE_NOT_ELIGIBLE"
    assert set(result["legacy_obligations"]) <= set(LEGACY_PROTECTED_HI_IDS)


def test_all_task_reference_rta_replay_round_trip_passes():
    taskset = toy_taskset()
    production = all_task_reference_rta(taskset)
    assert production["status"] == "PASS"
    replay = replay_all_task_rta(taskset, production)
    assert replay["status"] == "PASS"


def test_all_task_reference_rta_replay_rejects_mutated_trace():
    taskset = toy_taskset()
    production = all_task_reference_rta(taskset)
    mutated = production.copy()
    mutated["tasks"] = [dict(row) for row in production["tasks"]]
    mutated["tasks"][1] = dict(mutated["tasks"][1])
    mutated["tasks"][1]["case1"] = [dict(step) for step in mutated["tasks"][1]["case1"]]
    mutated["tasks"][1]["case1"][0] = dict(mutated["tasks"][1]["case1"][0])
    mutated["tasks"][1]["case1"][0]["trace"][0]["raw_f"] += 1
    replay = replay_all_task_rta(taskset, mutated)
    assert replay["status"] == "FAIL"


def test_reference_taskset_offset_validation():
    with pytest.raises(ValueError):
        ReferenceTask(
            name="BAD",
            period=10,
            deadline=0,
            c_lo=1,
            c_hi=1,
            criticality="LO",
            priority_index=0,
            code_c_lo=1,
            code_c_hi=1,
        )
