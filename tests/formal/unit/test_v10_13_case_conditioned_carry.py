import sys
import types
from math import gcd
import inspect

_fake_z3_added = "z3" not in sys.modules
if _fake_z3_added:
    sys.modules["z3"] = types.ModuleType("z3")

from formal_toolchain.v10_1.carry_in_envelope import (
    CarryTaskSpec,
    exact_joint_lo_entry_max_with_periodic_future,
    fixed_phase_lo_entry_backlog,
    target_release_joint_phase_parameters,
    target_release_joint_phases_at_q,
)
from formal_toolchain.v10_1 import pcssc
from formal_toolchain.v10_1.completion_certificates import (
    PCSSC_GUARDED_COMPLETION_THEOREM_V10_17,
)
from formal_toolchain.v10_1.constants import (
    FRAMEWORK_REVISION,
    TARGET_PROVED_PCSSC_CASE_CONDITIONED_CARRY,
)

if _fake_z3_added:
    sys.modules.pop("z3", None)


def test_fixed_phase_lo_entry_backlog_excludes_release_at_zero():
    specs = (
        CarryTaskSpec("a", "HI", 10, 4, 7),
        CarryTaskSpec("b", "LO", 15, 5, 2),
    )
    # Both tasks release at zero.  Their zero-time releases are future work,
    # not carry.  With periods greater than the 1-unit candidate suffix there
    # is no pending pre-zero workload in that suffix.
    value, details = fixed_phase_lo_entry_backlog(specs, (0, 0))
    assert value >= 0
    assert details["busy_horizon"] > 0


def test_joint_phase_coordinates_use_one_target_release_index_for_all_tasks():
    specs = (
        CarryTaskSpec("a", "HI", 7, 2, 4),
        CarryTaskSpec("b", "LO", 9, 3, 1),
    )
    n0, step, cycle = target_release_joint_phase_parameters(12, 5, 0, specs)
    assert cycle > 0
    for q in range(cycle):
        phases = target_release_joint_phases_at_q(
            12, specs, n0=n0, q_step=step, q=q
        )
        assert len(phases) == 2
        assert 0 <= phases[0] < 7
        assert 0 <= phases[1] < 9
        # The row is generated from a single target release index, not a
        # Cartesian product of independently selected task phases.
        assert isinstance(q, int)


def test_exact_crt_factorization_matches_bruteforce_v10_13_maximum():
    specs = (
        CarryTaskSpec("a", "HI", 6, 2, 3),
        CarryTaskSpec("b", "LO", 10, 3, 1),
        CarryTaskSpec("c", "HI", 7, 1, 2),
    )
    target_period = 13
    controller_period = 5
    theta = 0
    n0, step, cycle = target_release_joint_phase_parameters(
        target_period, controller_period, theta, specs
    )

    q_periods = tuple(
        spec.period // gcd(spec.period, step * target_period)
        for spec in specs
    )
    # Synthetic exact future tables.  The helper is agnostic to how each
    # task-local future value was obtained; it only requires Q_j periodicity.
    future_tables = tuple(
        tuple((index + 1) * ((r * 3 + index) % 5) for r in range(q_period))
        for index, q_period in enumerate(q_periods)
    )

    optimized, details = exact_joint_lo_entry_max_with_periodic_future(
        target_period, step, n0, specs, future_tables
    )

    brute = -1
    for q in range(cycle):
        phases = target_release_joint_phases_at_q(
            target_period, specs, n0=n0, q_step=step, q=q
        )
        carry, _ = fixed_phase_lo_entry_backlog(specs, phases)
        future = sum(
            future_tables[index][q % q_periods[index]]
            for index in range(len(specs))
        )
        brute = max(brute, carry + future)

    assert optimized == brute
    witness_q = details["witness_q"]
    phases = target_release_joint_phases_at_q(
        target_period, specs, n0=n0, q_step=step, q=witness_q
    )
    carry, _ = fixed_phase_lo_entry_backlog(specs, phases)
    future = sum(
        future_tables[index][witness_q % q_periods[index]]
        for index in range(len(specs))
    )
    assert carry + future == optimized


def test_exact_crt_factorization_avoids_large_global_joint_cycle():
    # q-periods are exactly 9,14,29,33,9,9,53,7 when target_period is coprime
    # to these task periods and q_step=1.  Their global lcm is 2,130,282, but
    # the dependency components have periods only 99,14,29,53.
    periods = (9, 14, 29, 33, 9, 9, 53, 7)
    specs = tuple(
        CarryTaskSpec(f"t{index}", "HI", period, 0, 0)
        for index, period in enumerate(periods)
    )
    future_tables = tuple(tuple(0 for _ in range(period)) for period in periods)
    value, details = exact_joint_lo_entry_max_with_periodic_future(
        1, 1, 0, specs, future_tables
    )
    assert value == 0
    assert details["joint_phase_cycle"] == 2_130_282
    assert sorted(details["component_periods"]) == [14, 29, 53, 99]
    assert details["residues_per_candidate"] == 195


def test_v10_13_terminal_does_not_materialize_or_scan_global_joint_q():
    source = inspect.getsource(pcssc._case_conditioned_joint_phase_interference)
    helper = inspect.getsource(exact_joint_lo_entry_max_with_periodic_future)
    assert "target_release_joint_phase_orbit" not in source
    assert "range(int(joint_cycle))" not in source
    assert "range(int(joint_cycle))" not in helper


def test_v10_13_case_postfix_fallback_can_close_a_v10_12_failed_case(monkeypatch):
    class Target:
        name = "hi0"
        deadline = 20

    class Switch:
        kind = "LO_SWITCH"
        lower = 0
        upper = 0
        id = "LO_SWITCH[0,0]"

    class Case:
        theta = 0
        switch = Switch()
        target_classification = "ABNORMAL"
        id = "THETA_0__LO_SWITCH[0,0]__TARGET_ABNORMAL"
        canonical_deadline = 20
        def as_dict(self):
            return {
                "case_id": self.id,
                "theta": 0,
                "switch_kind": "LO_SWITCH",
                "switch_lower": 0,
                "switch_upper": 0,
                "target_classification": "ABNORMAL",
                "canonical_deadline": 20,
            }

    monkeypatch.setattr(pcssc, "_target_cap", lambda target, cls: 5)
    monkeypatch.setattr(
        pcssc,
        "_controller_prefix_coverage_receipt",
        lambda *args, **kwargs: {"obligation_id": "PREFIX", "status": "PASS"},
    )
    monkeypatch.setattr(
        pcssc,
        "_workload_case_conditioned_v10_13",
        lambda *args, horizon, **kwargs: (
            12 if horizon < 12 else 11,
            {"selected_hp_bound": "V10_13_CASE_CONDITIONED_JOINT_PHASE"},
        ),
    )
    cert, path, failure = pcssc._case_conditioned_postfix_search_v10_13(
        pcssc._TargetWorkloadCache.empty(),
        object(), Target(), (), object(), set(), {}, Case()
    )
    assert failure is None
    assert cert is not None
    assert cert["R"] == 12
    assert cert["W"] == 11
    assert cert["case_theorem_basis"] == "V10_13_CASE_CONDITIONED_CARRY_FUTURE"
    assert path[-1]["postfixed"] is True


def test_v10_13_identifiers_are_distinct_and_exportable():
    assert FRAMEWORK_REVISION == "V10.17_CRT_PHASE_FAMILY_TERMINAL"
    assert TARGET_PROVED_PCSSC_CASE_CONDITIONED_CARRY.endswith("V10_13")
    assert PCSSC_GUARDED_COMPLETION_THEOREM_V10_17.endswith("V10_17")


def test_v10_13_is_not_used_as_a_soundness_error_fallback(monkeypatch):
    class Target:
        name = "hi0"
        deadline = 10

    class Switch:
        kind = "LO_SWITCH"
        lower = 0
        upper = 0
        id = "LO_SWITCH[0,0]"

    class Case:
        theta = 0
        switch = Switch()
        target_classification = "ABNORMAL"
        id = "C0"
        canonical_deadline = 10
        def as_dict(self):
            return {
                "case_id": self.id,
                "theta": 0,
                "switch_kind": "LO_SWITCH",
                "switch_lower": 0,
                "switch_upper": 0,
                "target_classification": "ABNORMAL",
                "canonical_deadline": 10,
            }

    monkeypatch.setattr(
        pcssc,
        "_deadline_canonical_case_domain",
        lambda *args, **kwargs: ((Case(),), (), "h" * 64),
    )
    monkeypatch.setattr(
        pcssc,
        "_case_postfix_search",
        lambda *args, **kwargs: (None, [], "CONTROLLER_PREFIX_COVERAGE_UNRESOLVED"),
    )
    called = {"refined": False}
    def refined(*args, **kwargs):
        called["refined"] = True
        raise AssertionError("V10.13 must not mask a soundness failure")
    monkeypatch.setattr(pcssc, "_case_conditioned_postfix_search_v10_13", refined)

    bound, tested, receipts, failure = pcssc._case_consistent_postfix_search(
        object(), Target(), (), object(), set(), {}
    )
    assert bound is None
    assert called["refined"] is False
    assert "CONTROLLER_PREFIX_COVERAGE_UNRESOLVED" in failure
    assert tested[0]["v10_13_refinement_attempted"] is False


def test_target_local_v10_13_cache_reuses_joint_hp_across_direct_recheck(monkeypatch):
    class Target:
        name = "hi0"
        period = 20
        deadline = 20
        c_lo = 5
        c_hi = 8
        actual_demand_min = 1
        actual_demand_upper = 8

    class Model:
        agent_period = 10

    class Path:
        boxes = ({},)

    switch = pcssc.SwitchCell("LO_SWITCH", 0, 0)
    calls = {"exact": 0}

    monkeypatch.setattr(pcssc, "candidate_controller_times", lambda *args: ())
    monkeypatch.setattr(pcssc, "_macro_cells", lambda *args: (pcssc.MacroCell(0, 20),))
    monkeypatch.setattr(pcssc, "_carry_task_specs", lambda *args: ())
    monkeypatch.setattr(pcssc, "target_release_joint_phase_parameters", lambda *args: (0, 1, 1))
    monkeypatch.setattr(pcssc, "phase_block_task_projections", lambda *args, **kwargs: ())
    monkeypatch.setattr(pcssc, "target_release_joint_phases_at_q", lambda *args, **kwargs: ())
    monkeypatch.setattr(pcssc, "fixed_phase_lo_entry_backlog", lambda *args, **kwargs: (0, {}))

    def exact(*args, **kwargs):
        calls["exact"] += 1
        return 7, {
            "witness_q": 0,
            "component_periods": (),
            "component_tasks": (),
            "candidate_lengths": 1,
            "residues_per_candidate": 0,
        }

    monkeypatch.setattr(pcssc, "exact_joint_lo_entry_max_with_periodic_future", exact)
    cache = pcssc._TargetWorkloadCache.empty()
    first = pcssc._case_conditioned_joint_phase_interference(
        cache, Model(), Target(), (), Path(), horizon=12, theta=0, switch=switch
    )
    second = pcssc._case_conditioned_joint_phase_interference(
        cache, Model(), Target(), (), Path(), horizon=12, theta=0, switch=switch
    )
    assert first == second
    assert calls["exact"] == 1


def test_lo_entry_carry_cache_is_shared_by_target_classifications(monkeypatch):
    class Target:
        name = "hi0"
        period = 20
        deadline = 20
        c_lo = 5
        c_hi = 8
        actual_demand_min = 1
        actual_demand_upper = 8

    class Model:
        agent_period = 10

    class Path:
        boxes = ({},)

    calls = {"carry": 0}
    monkeypatch.setattr(pcssc, "candidate_controller_times", lambda *args: ())
    monkeypatch.setattr(pcssc, "_macro_cells", lambda *args: (pcssc.MacroCell(0, 12),))

    def carry(*args, **kwargs):
        calls["carry"] += 1
        return 3, {"basis": "TEST", "carry_in": 3}

    monkeypatch.setattr(pcssc, "_lo_entry_aggregate_carry_bound", carry)
    cache = pcssc._TargetWorkloadCache.empty()
    normal, _ = pcssc._workload_case(
        cache, Model(), Target(), (), Path(), set(), {},
        horizon=12, theta=0, switch=pcssc.SwitchCell("LO_SWITCH", 0, 0),
        classification="NORMAL",
    )
    abnormal, _ = pcssc._workload_case(
        cache, Model(), Target(), (), Path(), set(), {},
        horizon=12, theta=0, switch=pcssc.SwitchCell("LO_SWITCH", 0, 0),
        classification="ABNORMAL",
    )
    assert normal == 5
    assert abnormal == 8
    assert calls["carry"] == 1



def test_exact_crt_carry_plan_is_reused_across_future_tables(monkeypatch):
    import formal_toolchain.v10_1.carry_in_envelope as carry

    specs = (
        CarryTaskSpec("a", "HI", 9, 2, 3),
        CarryTaskSpec("b", "LO", 14, 2, 1),
        CarryTaskSpec("c", "HI", 29, 1, 2),
    )
    n0, step, _ = target_release_joint_phase_parameters(25, 8, 0, specs)
    q_periods = tuple(
        spec.period // gcd(spec.period, step * 25) for spec in specs
    )
    future_a = tuple(tuple(0 for _ in range(period)) for period in q_periods)
    future_b = tuple(
        tuple((index + residue) % 4 for residue in range(period))
        for index, period in enumerate(q_periods)
    )

    carry._exact_joint_lo_entry_carry_plan.cache_clear()
    calls = {"count": 0}
    original = carry._count_releases

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(carry, "_count_releases", counted)
    exact_joint_lo_entry_max_with_periodic_future(25, step, n0, specs, future_a)
    after_first = calls["count"]
    exact_joint_lo_entry_max_with_periodic_future(25, step, n0, specs, future_b)
    assert after_first > 0
    assert calls["count"] == after_first
