from nonvacuity_lab.activation.hout_activation import evaluate_hout_activation


def test_hout_activation_requires_target_reject_and_mutated_selection():
    result = evaluate_hout_activation(
        mutation_id="B1",
        base_events=[
            {
                "leaf_id": 7,
                "raw_action_id": 13,
                "raw_top1_invalid": True,
                "selected_action_id": 2,
                "reject_reason": "HI_FLOOR",
            }
        ],
        mutated_events=[
            {
                "leaf_id": 7,
                "raw_action_id": 13,
                "selected_action_id": 13,
                "time": 4,
            }
        ],
        rule={
            "required_leaf_id": 7,
            "required_action_id": 13,
            "require_baseline_reject": True,
            "require_selected_after_mutation": True,
        },
    )
    assert result.status.value == "ACTIVATED"
    assert result.baseline_reject_count == 1
    assert result.selected_after_mutation_count == 1
    assert result.details["first_intervention_time"] == 4


def test_hout_runtime_field_names_are_normalized():
    result = evaluate_hout_activation(
        mutation_id="B1",
        base_events=[
            {
                "tree_leaf_id": 7,
                "raw_top1_action_id": 3,
                "tree_raw_top1_invalid": True,
                "tree_selected_action_id": 1,
            }
        ],
        mutated_events=[
            {
                "tree_leaf_id": 7,
                "raw_top1_action_id": 3,
                "tree_raw_top1_invalid": True,
                "tree_selected_action_id": 3,
            }
        ],
        rule={
            "required_leaf_id": 7,
            "required_action_id": 3,
            "require_baseline_reject": True,
            "require_selected_after_mutation": True,
        },
    )
    assert result.status.value == "ACTIVATED"
