from formal_toolchain.bridge.handler_decomposition import (
    HANDLER_COMPOSITION_CASES,
    RESCHEDULE_ALTERNATIVES,
)


def test_each_handler_sequence_contains_at_most_one_reschedule_outcome():
    for component, case_ids in HANDLER_COMPOSITION_CASES.items():
        outcomes = [case_id for case_id in case_ids if case_id in RESCHEDULE_ALTERNATIVES]
        assert len(outcomes) <= 1, (component, case_ids)


def test_arrival_controller_and_completion_outcomes_are_separate_alternatives():
    assert HANDLER_COMPOSITION_CASES["arrival_no_switch_keep"] == (
        "ARRIVAL_BATCH_NO_SWITCH",
        "RESCHEDULE_KEEP_SAME",
    )
    assert HANDLER_COMPOSITION_CASES["arrival_no_switch_idle"] == (
        "ARRIVAL_BATCH_NO_SWITCH",
        "RESCHEDULE_TO_IDLE",
    )
    assert HANDLER_COMPOSITION_CASES["arrival_no_switch_dispatch"] == (
        "ARRIVAL_BATCH_NO_SWITCH",
        "PREEMPTION_DISPATCH",
    )
    assert HANDLER_COMPOSITION_CASES["controller_no_action_idle"] == (
        "CONTROLLER_NO_ACTION",
        "RESCHEDULE_TO_IDLE",
    )
    assert HANDLER_COMPOSITION_CASES["controller_no_action_dispatch"] == (
        "CONTROLLER_NO_ACTION",
        "PREEMPTION_DISPATCH",
    )
    assert HANDLER_COMPOSITION_CASES["normal_completion_keep"] == (
        "NORMAL_COMPLETION",
        "RESCHEDULE_KEEP_SAME",
    )
    assert HANDLER_COMPOSITION_CASES["normal_completion_dispatch"] == (
        "NORMAL_COMPLETION",
        "PREEMPTION_DISPATCH",
    )
