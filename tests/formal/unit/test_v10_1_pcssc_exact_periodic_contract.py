from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_pcssc_uses_bound_exact_periodic_release_model_without_inner_arrival_smt():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "EXACT_PERIODIC_PHASE_ZERO" in text
    assert "compatible_release_phases" in text
    assert "_exact_periodic_phase_workload_cached" in text
    assert "_exact_periodic_task_workload" in text
    assert "z3.Int(" not in text
    assert "_maximize_counts_cached" not in text
    assert "maximize_weighted_counts" not in text
    assert "make_solver" not in text


def test_exact_periodic_workload_couples_previous_carry_and_future_releases():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "if phase == 0:" in text
    assert "carry = 0" in text
    assert "age = int(task_period) - int(phase)" in text
    assert "residual_time = int(completion_bound) - age" in text
    assert "future += _weight_at_release(release, cells, weights)" in text
    assert "release += int(task_period)" in text
    assert "PHASE_BLOCK_WORKLOAD_LIFTING_SOUND::" in text
    assert '"global_q_enumerated": False' in text


def test_periodic_cross_task_phase_relaxation_is_explicitly_recorded():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "FAST_ROUTE_CROSS_TASK_PERIODIC_PHASE_RELAXATION" in text
    assert "deferred to the V10.16 adaptive phase-block terminal" in text
    assert "ONLY_ADDS_CROSS_TASK_PHASE_COMBINATIONS" in text
    assert "PER_TASK_PERIODIC_PHASE_MAX_NOT_REQUIRED_TO_REPRODUCE_FEATURE_HISTORY" in text
