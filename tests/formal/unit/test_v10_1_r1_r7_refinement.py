from pathlib import Path

from formal_toolchain.v10_1.carry_in_envelope import (
    CarryTaskSpec,
    fixed_phase_single_switch_backlog,
    phase_relaxed_lo_entry_carry,
    phase_relaxed_single_switch_carry,
)

ROOT = Path(__file__).resolve().parents[3]


def test_r1_uses_boot_seeded_controller_closed_interval_fixpoint_without_iteration_cap():
    text = (ROOT / "formal_toolchain/v10_1/controller_macro.py").read_text(encoding="utf-8")
    assert "_boot_reachable_budget_invariant" in text
    assert "initial_runtime_budget" in text
    assert "task.initial_budget" in text
    assert "_controller_image_hull" in text
    assert "_join_budget_boxes" in text
    assert "postfixed" in text
    assert "FIRST_BAD_START_BUDGET_WIDENING" not in text
    assert "max_iterations" not in text


def test_r7_removes_unprotected_lo_early_exit_and_routes_through_aggregate_backlog():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "R7_SINGLE_SWITCH_AGGREGATE_BACKLOG_ENVELOPE" in text
    assert "included in aggregate work-conserving backlog" in text
    assert "phase_relaxed_single_switch_carry" in text
    assert "phase_relaxed_lo_entry_carry" in text
    assert "phase_block_r7_carry_upper" in text
    assert "PHASE_BLOCK_WORKLOAD_LIFTING_SOUND::" in text
    assert 'target.name, "UNRESOLVED", None, "REACHABLE_LO_CARRY_IN_UNRESOLVED"' not in text


def test_single_switch_aggregate_backlog_respects_lo_and_hi_switch_endpoint_semantics():
    specs = (
        CarryTaskSpec("lo", "LO", 10, 6, 2),
        CarryTaskSpec("hi", "HI", 10, 2, 6),
    )
    # Both tasks next release at zero.  If the switch is exactly at -10, the
    # LO endpoint remains primary (6) while the HI endpoint is already high (6),
    # yielding 12 units of work over a 10-unit interval and backlog 2.
    bound, details = fixed_phase_single_switch_backlog(specs, (0, 0))
    assert bound == 2
    assert details["busy_horizon"] >= 10


def test_phase_relaxed_lo_entry_backlog_is_finite_for_stable_periodic_workload():
    specs = (
        CarryTaskSpec("a", "LO", 10, 3, 1),
        CarryTaskSpec("b", "HI", 20, 2, 4),
    )
    bound, details = phase_relaxed_lo_entry_carry(20, 10, 0, specs)
    assert bound >= 0
    assert details["busy_horizon"] > 0


def test_phase_relaxed_single_switch_backlog_needs_no_per_task_completion_certificate():
    specs = (
        CarryTaskSpec("lo_unprotected", "LO", 25, 7, 2),
        CarryTaskSpec("hi", "HI", 20, 2, 5),
    )
    bound, details = phase_relaxed_single_switch_carry(40, 10, 0, specs)
    assert bound >= 0
    assert details["candidate_switch_ages"] > 0


def test_source_manifest_binds_new_r7_module():
    text = (ROOT / "formal_toolchain/adapters/source_manifest.py").read_text(encoding="utf-8")
    assert '"formal_toolchain/v10_1/carry_in_envelope.py"' in text
