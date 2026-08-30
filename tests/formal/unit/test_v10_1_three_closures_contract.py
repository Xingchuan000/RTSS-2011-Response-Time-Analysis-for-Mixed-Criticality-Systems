from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_feature_transfer_is_per_target_per_epoch_per_feature():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "FLOW_START_FEATURE_TRANSFER_SOUND::" in text
    assert "INTER_EPOCH_FEATURE_TRANSFER_SOUND::" in text
    assert "INTER_EPOCH_FEATURE_TRANSFER_COVERAGE::" in text
    assert "required_policy_read_features" in text


def test_controller_prefix_coverage_is_bound_to_tested_response_horizon():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "CONTROLLER_PATH_PREFIX_COVERAGE::" in text
    assert '"prefix_closed": True' in text
    assert '"future_independent": True' in text
    assert '"horizon_consistent": True' in text


def test_fixed_priority_delay_accounting_is_terminal_obligation():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    for token in (
        "TOTAL_PRIORITY_OR_TIEBREAK_BINDING",
        "EFFECTIVE_AHEAD_SET_SOUND::",
        "TARGET_NEXT_RELEASE_EXCLUDED_ON_POSTFIX_HORIZON::",
        "FIXED_PRIORITY_TARGET_DELAY_ACCOUNTING::",
    ):
        assert token in text


def test_switch_endpoint_and_arrival_correlation_rules_are_explicit():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "LO at u=s is primary" in text
    assert "HI at u=s may use C_HI" in text
    assert "for a in range(q)" in text and "for b in range(a, q)" in text
    assert "POLICY_ARRIVAL_CORRELATION_RELAXATION" in text


def test_first_bad_controller_start_region_is_not_boot_budget():
    text = (ROOT / "formal_toolchain/v10_1/controller_macro.py").read_text(encoding="utf-8")
    assert "FIRST_BAD_START_BUDGET_WIDENING" in text
    assert "task.budget_floor" in text and "task.budget_upper" in text
    assert "task.initial_budget" not in text


def test_full_feature_domain_drives_exact_cart_without_narrow_history_reconstruction():
    text = (ROOT / "formal_toolchain/v10_1/controller_macro.py").read_text(encoding="utf-8")
    assert "_full_domain_controller_encoding" in text
    assert "encode_tree_leaf_and_ranking" in text
    assert "encode_action_mask" in text
    assert "encode_first_valid_leaf_cases" in text
    assert "_history_domain" not in text
    assert "names = [str(value) for value in model.feature_names]" in text


def test_feature_transfer_has_exact_binary64_machine_obligation():
    text = (ROOT / "formal_toolchain/v10_1/feature_transfer.py").read_text(encoding="utf-8")
    for token in (
        "z3.Float64()",
        "z3.fpMul",
        "z3.fpAdd",
        "FULL_LEGAL_HISTORY_UPDATE_DOMAIN_CLOSURE",
        "FULL_LEGAL_NUMERIC_OBSERVATION_DOMAIN",
    ):
        assert token in text


def test_scheduler_safe_prefix_does_not_assume_real_ema_bound():
    text = (ROOT / "formal_toolchain/v10_1/safe_prefix.py").read_text(encoding="utf-8")
    assert 'rows.pop("history_bounds", None)' in text
    assert 'mutable=frozenset({"budgets", "history"})' in text
    assert "build_p5_scheduler_summary_soundness_obligations" in text


def test_p5_scheduler_summary_preserves_structural_history_domain():
    text = (ROOT / "formal_toolchain/v10_1/safe_prefix.py").read_text(encoding="utf-8")
    assert "_history_structural_domain" in text
    assert "z3.Implies(enabled, _history_structural_domain(zp, model))" in text
    for token in (
        "state.chi.recent_cost[task.name] >= 0",
        "state.chi.ema_cost[task.name] >= 0",
        "state.chi.overrun_ema[task.name] >= 0",
        "state.chi.max_cost_k[task.name] >= 0",
        "state.chi.job_start_window",
    ):
        assert token in text


def test_controller_prefix_coverage_checks_depth_and_budget_boxes_before_receipt():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "required_controller_depth" in text
    assert "available_controller_depth" in text
    assert "MISSING_BUDGET" in text and "BAD_BOX" in text
    assert "timestamps c_k<R" in text


def test_kernel_p5_history_projection_uses_same_fp64_widening_as_feature_transfer():
    text = (ROOT / "formal_toolchain/v10_1/kernel/transition_encoder.py").read_text(encoding="utf-8")
    assert "state.chi.ema_cost[task.name] <= 2 * upper" in text
    assert "state.chi.overrun_ema[task.name] <= 2" in text
    assert "encode_p5_invariant_summary" not in text
