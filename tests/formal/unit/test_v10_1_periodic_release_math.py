from math import gcd, lcm

from formal_toolchain.v10_1.periodic_release import compatible_release_phases


def _bruteforce_phases(target_period: int, task_period: int, controller_period: int, theta: int):
    target_cycle = controller_period // gcd(target_period, controller_period)
    task_cycle = task_period // gcd(target_period, task_period)
    n_cycle = lcm(target_cycle, task_cycle)
    return tuple(sorted({
        (-n * target_period) % task_period
        for n in range(n_cycle)
        if (-n * target_period) % controller_period == theta
    }))


def test_modular_release_phase_orbit_matches_bruteforce_on_small_periods():
    for target_period in range(1, 13):
        for task_period in range(1, 13):
            for controller_period in range(1, 10):
                cycle = controller_period // gcd(target_period, controller_period)
                thetas = {
                    (-n * target_period) % controller_period
                    for n in range(cycle)
                }
                for theta in thetas:
                    assert compatible_release_phases(
                        target_period, task_period, controller_period, theta
                    ) == _bruteforce_phases(
                        target_period, task_period, controller_period, theta
                    )


def test_incompatible_controller_phase_is_rejected():
    # T_i=6 and T_c=10 can only induce even theta values.
    try:
        compatible_release_phases(6, 7, 10, 1)
    except ValueError as exc:
        assert "EXACT_PERIODIC_CONTROLLER_PHASE_INCOMPATIBLE" in str(exc)
    else:
        raise AssertionError("incompatible theta was accepted")
