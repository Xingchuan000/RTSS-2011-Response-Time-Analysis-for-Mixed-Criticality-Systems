from formal_toolchain.reference.protected_priority_prefix.construction import build_saturated_protected_prefix
from formal_toolchain.reference.protected_priority_prefix.observable import _hi_class
from formal_toolchain.reference.rta_production import all_task_protected_prefix_rta
from formal_toolchain.reference.rta_replay import replay_all_task_rta
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.routes.protected_prefix_checkers import (
    check_runtime_schema_conformance, check_weak_forward_simulation,
    check_hi_bad_prefix_reflection,
)


def _pass_predecessor():
    return {"obligation_status": "PASS"}


def test_theorem_receipts_cannot_close_quantified_pp_obligations():
    runtime = check_runtime_schema_conformance(
        verified_predecessors={"SATURATED_PROTECTED_PREFIX_REFERENCE": _pass_predecessor()}
    )
    assert runtime["status"] == "UNRESOLVED"
    assert runtime["code"] == "PROTECTED_PREFIX_RUNTIME_SCHEMA_PARAMETRIC_PROOF_MISSING"

    weak = check_weak_forward_simulation(
        verified_predecessors={"FULL_TO_PREFIX_SIMULATION_DOMAIN": _pass_predecessor()}
    )
    assert weak["status"] == "UNRESOLVED"

    reflection = check_hi_bad_prefix_reflection(
        verified_predecessors={"PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED": _pass_predecessor()}
    )
    assert reflection["status"] == "UNRESOLVED"


def test_rta_replay_does_not_pass_an_unschedulable_prefix():
    taskset = ReferenceTaskset((
        ReferenceTask("hi", 10, 1, 2, 3, "HI", 0, 2, 3, None, 0),
    ), "a" * 64)
    production = all_task_protected_prefix_rta(
        taskset, certificate_context_hash="b" * 64
    )
    assert production["status"] == "FAIL"
    replay = replay_all_task_rta(
        taskset, production,
        expected_obligation_id="PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        expected_route_id="protected_prefix",
    )
    assert replay["status"] == "FAIL"
    assert replay["code"] == "ALL_TASK_SUFFICIENT_TEST_FAILED"
    assert replay["witness"]["all_deadlines_met"] is False


def test_protected_observable_forgets_trigger_identity_but_keeps_hi_class():
    assert _hi_class("HI", "HI_ABNORMAL_SWITCH_TRIGGER") == "ABNORMAL"
    assert _hi_class("HI", "HI_ABNORMAL") == "ABNORMAL"
    assert _hi_class("HI", "HI_NORMAL") == "NORMAL"
    assert _hi_class("LO", "LO_PRIMARY_NORMAL") is None
