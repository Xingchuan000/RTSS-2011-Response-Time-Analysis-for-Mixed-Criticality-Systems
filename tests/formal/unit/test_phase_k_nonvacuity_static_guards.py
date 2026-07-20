from types import SimpleNamespace

from formal_toolchain.bridge.transition_compiler import (
    build_phase_k_static_guard_bindings,
    compile_source_guards,
)


def _guard(source: str, polarity: bool = True):
    return [{"test_source": source, "test_ast_hash": "a" * 64, "polarity": polarity}]


def test_off_profile_static_guards_compile_to_false():
    cfg = SimpleNamespace(
        nonvacuity_deadline_cleanup_remove=False,
        nonvacuity_hi_budget_cap_truncate=False,
        nonvacuity_recover_without_quiescence=False,
    )
    bindings = build_phase_k_static_guard_bindings(cfg)
    deadline = compile_source_guards(
        _guard("self.config.nonvacuity_deadline_cleanup_remove", False),
        static_guard_bindings=bindings,
    )
    hi_cap = compile_source_guards(
        _guard(
            "self.config.nonvacuity_hi_budget_cap_truncate and "
            "job.task.criticality is Criticality.HI",
            False,
        ),
        static_guard_bindings=bindings,
    )
    assert "(not false)" in deadline.formula
    assert "(not (and false (= task_criticality 1)))" in hi_cap.formula


def test_enabled_profile_static_guards_compile_to_true_branch_condition():
    cfg = SimpleNamespace(
        nonvacuity_deadline_cleanup_remove=True,
        nonvacuity_hi_budget_cap_truncate=True,
        nonvacuity_recover_without_quiescence=True,
    )
    bindings = build_phase_k_static_guard_bindings(cfg)
    deadline = compile_source_guards(
        _guard("self.config.nonvacuity_deadline_cleanup_remove"),
        static_guard_bindings=bindings,
    )
    hi_cap = compile_source_guards(
        _guard(
            "self.config.nonvacuity_hi_budget_cap_truncate and "
            "job.task.criticality is Criticality.HI"
        ),
        static_guard_bindings=bindings,
    )
    recovery = compile_source_guards(
        _guard(
            "cfg.nonvacuity_recover_without_quiescence and "
            "state.mode is SystemMode.HI"
        ),
        static_guard_bindings=bindings,
    )
    assert "true" in deadline.formula
    assert "(and true (= task_criticality 1))" in hi_cap.formula
    assert "(and true (= c_mode 1))" in recovery.formula


def test_legacy_direct_calls_default_to_safe_off():
    result = compile_source_guards(
        _guard("self.config.nonvacuity_deadline_cleanup_remove", False)
    )
    assert "(not false)" in result.formula
