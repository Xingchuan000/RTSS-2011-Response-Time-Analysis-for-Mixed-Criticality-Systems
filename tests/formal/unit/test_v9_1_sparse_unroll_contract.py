from pathlib import Path


def test_window_uses_known_phase_dispatch_and_gcd_controller_pruning():
    source = Path("formal_toolchain/v9_1/window_encoder.py").read_text(encoding="utf-8")
    assert "encode_phase_step" in source
    assert "controller_stride = gcd(model.agent_period, task.period)" in source
    assert "allowed_ticks_by_task=allowed_ticks" in source
    assert "encode_step(states[index]" not in source


def test_boot_reachability_uses_exact_known_phase_controller_schedule():
    source = Path("formal_toolchain/v9_1/safe_prefix_reachability.py").read_text(encoding="utf-8")
    assert "encode_phase_step" in source
    assert "absolute_tick % model.agent_period == 0" in source


def test_environment_supports_sparse_per_task_demand_ticks():
    source = Path("formal_toolchain/v9_1/environment_encoder.py").read_text(encoding="utf-8")
    assert "allowed_ticks_by_task" in source
    assert "for (name, tick), demand in env.actual_demands.items()" in source
