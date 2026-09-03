import inspect
import random
import sys
import types
from math import lcm

_fake_z3_added = "z3" not in sys.modules
if _fake_z3_added:
    sys.modules["z3"] = types.ModuleType("z3")

from formal_toolchain.v10_1 import pcssc
import formal_toolchain.v10_1.carry_in_envelope as carry
from formal_toolchain.v10_1.carry_in_envelope import (
    CarryTaskSpec,
    crt_phase_family_plan,
    exact_crt_phase_family_pre_hi_max,
    fixed_phase_pre_hi_interference,
    fixed_phase_single_switch_backlog,
    phase_block_task_projections as real_phase_block_task_projections,
    target_release_joint_phase_parameters,
    target_release_joint_phases_at_q,
)
from formal_toolchain.v10_1.completion_certificates import (
    PCSSC_GUARDED_COMPLETION_THEOREM_V10_17,
)
from formal_toolchain.v10_1.constants import (
    FRAMEWORK_REVISION,
    TARGET_PROVED_PCSSC_MIXED_PHASE_TERMINALS_V10_17,
)

if _fake_z3_added:
    sys.modules.pop("z3", None)


def _case(deadline=10):
    return pcssc.CaseKey(
        theta=0,
        switch=pcssc.SwitchCell("PRE_HI"),
        target_classification="ABNORMAL",
        canonical_deadline=deadline,
    )


class _Task:
    def __init__(self, name, period=2, priority=0, criticality="HI"):
        self.name = name
        self.period = period
        self.priority = priority
        self.criticality = criticality


class _Model:
    agent_period = 5
    tasks = ()


class _Target10:
    name = "hi0"
    period = 10
    deadline = 10
    priority = 0
    criticality = "HI"


class _HP:
    name = "hp"
    period = 2
    priority = 0
    criticality = "HI"


class _TargetSplit:
    name = "hi0"
    period = 1
    deadline = 10
    priority = 1
    criticality = "HI"


def test_v10_17_empty_ahead_has_q1_lifted_root_only(monkeypatch):
    monkeypatch.setattr(pcssc, "_carry_task_specs", lambda *args: ())
    monkeypatch.setattr(pcssc, "_target_cap", lambda *args: 4)
    monkeypatch.setattr(
        pcssc,
        "_controller_prefix_coverage_receipt",
        lambda *args, **kwargs: {"obligation_id": "PREFIX", "status": "PASS"},
    )

    cert, attempts, failure = pcssc._phase_block_postfix_search_v10_17(
        _Model(), _Target10(), (), object(), {}, _case()
    )

    assert failure is None
    assert cert is not None
    assert cert["R"] == 4
    assert cert["phase_block_joint_period"] == 1
    assert cert["phase_block_leaf_count"] == 1
    assert cert["crt_phase_family_terminal_count"] == 0
    assert cert["global_q_enumerated"] is False
    assert attempts == [{
        "block_id": "M1_A0", "M": 1, "a": 0, "depth": 0,
        "parent_id": None, "split_factor": None, "status": "PASS",
        "R": 4, "W": 4, "projection_sizes": [], "candidate_steps": 1,
    }]
    ids = [r["obligation_id"] for r in cert["phase_block_receipts"]]
    assert any(x.startswith("MIXED_PHASE_TERMINAL_COVER::") for x in ids)
    assert not any(x.startswith("CRT_PHASE_FAMILY_RESPONSE_CERTIFICATE::") for x in ids)


def test_v10_17_lifted_failure_can_refine_without_crt(monkeypatch):
    spec = CarryTaskSpec("hp", "HI", 2, 1, 1)
    monkeypatch.setattr(pcssc, "_carry_task_specs", lambda *args: (spec,))
    monkeypatch.setattr(pcssc, "_target_cap", lambda *args: 4)
    monkeypatch.setattr(
        pcssc,
        "target_release_joint_phase_parameters",
        lambda *args, **kwargs: (0, 1, 2),
    )
    monkeypatch.setattr(pcssc, "phase_block_task_projections", real_phase_block_task_projections)

    def fake_r7(specs, projections):
        projection = projections[0]
        carry_value = 20 if projection.phase_count > 1 else (2 if projection.phase_residue == 0 else 5)
        return carry_value, {
            "candidate_domain_kind": "PROVED_BOUNDARY_UNION",
            "busy_horizon": 2,
            "candidate_boundary_count": 1,
            "witness_length": 1,
        }

    monkeypatch.setattr(pcssc, "phase_block_r7_carry_upper", fake_r7)
    monkeypatch.setattr(pcssc, "phase_block_post_switch_future_upper", lambda *args: 0)
    monkeypatch.setattr(
        pcssc,
        "_crt_phase_family_postfix_search_v10_17",
        lambda *args, **kwargs: (None, [], "CRT_PHASE_FAMILY_POSTFIX_FAIL:TEST"),
    )
    monkeypatch.setattr(
        pcssc,
        "_controller_prefix_coverage_receipt",
        lambda *args, **kwargs: {"obligation_id": "PREFIX", "status": "PASS"},
    )

    model = _Model()
    model.tasks = (_HP(), _TargetSplit())
    cert, attempts, failure = pcssc._phase_block_postfix_search_v10_17(
        model, _TargetSplit(), (_HP(),), object(), {}, _case()
    )

    assert failure is None
    assert cert is not None
    assert cert["R"] == 9
    assert cert["phase_block_leaf_count"] == 2
    assert cert["case_theorem_basis"] == "V10_17_MIXED_PHASE_TERMINAL"
    assert cert["lifted_block_terminal_count"] == 2
    assert cert["crt_phase_family_terminal_count"] == 0
    assert attempts[0]["status"] == "FAILED_BLOCK"
    assert {row["block_id"] for row in attempts[1:]} == {"M2_A0", "M2_A1"}


def test_v10_17_crt_exact_family_matches_bruteforce_random_small_domains():
    rng = random.Random(17017)
    checked = 0
    for _ in range(400):
        specs = []
        for index in range(rng.randint(0, 4)):
            period = rng.randint(2, 12)
            cap_limit = max(0, period // 4)
            specs.append(CarryTaskSpec(
                f"t{index}", rng.choice(("HI", "LO")), period,
                rng.randint(0, cap_limit), rng.randint(0, cap_limit),
            ))
        specs = tuple(specs)
        target_period = rng.randint(1, 10)
        controller_period = rng.randint(1, 8)
        try:
            n0, q_step, Q = target_release_joint_phase_parameters(
                target_period, controller_period, 0, specs
            )
            divisors = [value for value in range(1, Q + 1) if Q % value == 0]
            M = rng.choice(divisors)
            a = rng.randrange(M)
            plan = crt_phase_family_plan(target_period, q_step, n0, specs, Q, M, a)
        except carry.CarryInEnvelopeUnresolved:
            continue
        K = Q // M
        if K > 80:
            continue
        horizon = rng.randint(1, 15)
        exact, details = exact_crt_phase_family_pre_hi_max(
            target_period, q_step, n0, specs, Q, M, a, horizon
        )
        brute = 0
        for k in range(K):
            q = (a + M * k) % Q
            phases = target_release_joint_phases_at_q(
                target_period, specs, n0=n0, q_step=q_step, q=q
            )
            value, _ = fixed_phase_pre_hi_interference(
                specs, phases, horizon, completion_bounds=None
            )
            brute = max(brute, value)
            common_r7 = carry._fixed_phase_r7_on_common_candidate_domain(specs, phases, plan)
            production_r7, _ = fixed_phase_single_switch_backlog(specs, phases)
            assert common_r7 == production_r7
        assert exact == brute
        assert details["witness_reevaluation"] == exact
        checked += 1
    assert checked >= 300


def test_v10_17_component_partition_uses_gcd_connected_components():
    # Same structure that matters for s603 hi3 before the final factor-11 split.
    periods = (9, 14, 29, 33, 9, 9, 53, 7)
    specs = tuple(CarryTaskSpec(f"t{i}", "HI", p, 0, 0) for i, p in enumerate(periods))
    Q = lcm(*periods)
    plan = crt_phase_family_plan(1, 1, 0, specs, Q, 1, 0)
    assert plan.family_size == Q
    assert sorted(plan.component_periods) == [14, 29, 53, 99]
    product = 1
    for period in plan.component_periods:
        product *= period
    assert product == Q


def test_v10_17_crt_terminal_is_r7_only_and_emits_direct_postfix(monkeypatch):
    specs = (CarryTaskSpec("hp", "HI", 2, 1, 1),)
    target = types.SimpleNamespace(
        name="hi0", period=1, deadline=10, actual_demand_upper=4, c_lo=2, c_hi=4,
    )
    block = pcssc.PhaseBlock(modulus=1, residue=0, depth=0)
    monkeypatch.setattr(pcssc, "_target_cap", lambda *args: 4)
    cert, rows, failure = pcssc._crt_phase_family_postfix_search_v10_17(
        target, _case(), specs, block, n0=0, q_step=1, joint_period=2,
        runtime_total_order_hash="a" * 64,
        controller_period=5,
        recurrence_after_deadline_failure=True,
    )
    assert failure is None
    assert cert is not None
    assert cert["terminal_type"] == "CRT_PHASE_FAMILY"
    ids = [row["obligation_id"] for row in cert["receipts"]]
    assert any(x.startswith("COMMON_TIMEBASE_EXACT::") for x in ids)
    assert any(x.startswith("C_AMC_SEM_SWITCH_ENDPOINT_BINDING::") for x in ids)
    assert any(x.startswith("CRT_PHASE_FAMILY_DIRECT_POSTFIX::") for x in ids)
    direct = next(row for row in cert["receipts"] if row["obligation_id"].startswith("CRT_PHASE_FAMILY_DIRECT_POSTFIX::"))
    assert direct["Q"] == 2
    assert direct["K"] == 2
    acnf = next(row for row in cert["receipts"] if row["obligation_id"].startswith("CRT_PHASE_FAMILY_ACNF_SOUND::"))
    assert acnf["carry_mode"] == "FIXED_Q_R7_CRT"
    assert "completion" not in repr(acnf).lower()


def test_v10_17_structural_budget_fails_closed_if_crt_plan_is_not_available(monkeypatch):
    spec = CarryTaskSpec("hp", "HI", 2, 1, 1)
    monkeypatch.setattr(pcssc, "_carry_task_specs", lambda *args: (spec,))
    monkeypatch.setattr(pcssc, "_target_cap", lambda *args: 4)
    monkeypatch.setattr(
        pcssc, "target_release_joint_phase_parameters", lambda *args, **kwargs: (0, 2, 1)
    )
    monkeypatch.setattr(pcssc, "phase_block_r7_carry_upper", lambda *args: (
        20,
        {"candidate_domain_kind": "PROVED_BOUNDARY_UNION", "busy_horizon": 2,
         "candidate_boundary_count": 1, "witness_length": 1},
    ))
    monkeypatch.setattr(pcssc, "phase_block_post_switch_future_upper", lambda *args: 0)
    monkeypatch.setattr(
        pcssc, "_crt_phase_family_postfix_search_v10_17",
        lambda *args, **kwargs: (None, [], "CRT_PHASE_FAMILY_POSTFIX_FAIL:TEST"),
    )
    model = _Model()
    model.tasks = (_HP(), _Target10())
    cert, attempts, failure = pcssc._phase_block_postfix_search_v10_17(
        model, _Target10(), (_HP(),), object(), {}, _case()
    )
    assert cert is None
    assert attempts[-1]["status"] == "FAILED_BLOCK"
    assert failure is not None
    assert "CRT_PHASE_FAMILY_POSTFIX_FAIL" in failure or "NO_LEGAL_SPLIT" in failure


def test_v10_17_outer_case_aggregation_exports_guarded_completion(monkeypatch):
    case = _case()
    monkeypatch.setattr(
        pcssc, "_deadline_canonical_case_domain", lambda *args, **kwargs: ((case,), (), "d" * 64)
    )
    monkeypatch.setattr(
        pcssc, "_phase_block_postfix_search_v10_17",
        lambda *args, **kwargs: ({
            **case.as_dict(), "status": "PASS", "R": 7, "W": 7,
            "candidate_path": [{"block_id": "M2_A0", "status": "PASS"}],
            "controller_prefix_receipt": {"obligation_id": "PREFIX", "status": "PASS"},
            "case_theorem_basis": "V10_17_MIXED_PHASE_TERMINAL",
            "phase_block_joint_period": 2, "phase_block_leaf_count": 1,
            "phase_block_leaf_digest_sha256": "a" * 64,
            "worst_phase_block": "M2_A0", "uniform_R_is_common_postfix": False,
            "lifted_block_terminal_count": 0, "crt_phase_family_terminal_count": 1,
            "phase_block_receipts": [{
                "obligation_id": f"CRT_PHASE_FAMILY_DIRECT_POSTFIX::{case.id},M2_A0",
                "status": "PASS", "R_B": 7, "W_B_R_B": 7,
            }],
            "global_q_enumerated": False,
        }, [{"block_id": "M2_A0", "status": "PASS"}], None),
    )

    class Target:
        name = "hi0"
        deadline = 10
        criticality = "HI"

    bound, tested, receipts, failure = pcssc._case_consistent_postfix_search(
        object(), Target(), (), object(), set(), {}
    )
    assert failure is None
    assert bound == 7
    ids = [row["obligation_id"] for row in receipts]
    assert any(x.startswith("GUARDED_SAFE_PREFIX_WINDOW_REDUCTION::hi0") for x in ids)
    export = next(row for row in receipts if row["obligation_id"].startswith("GUARDED_SAFE_PREFIX_COMPLETION_EXPORT::hi0"))
    assert export["theorem_basis"] == PCSSC_GUARDED_COMPLETION_THEOREM_V10_17
    assert export["conditional_safe_prefix_completion"] is True
    all_cases = next(row for row in receipts if row["obligation_id"] == "ALL_CASES_POSTFIX_COVERED::hi0")
    assert all_cases["v10_17_mixed_phase_terminal_case_ids"] == [case.id]
    assert tested[0]["case_theorem_basis"] == "V10_17_MIXED_PHASE_TERMINAL"


def test_v10_17_identifiers_and_source_do_not_restore_global_q_scan():
    assert FRAMEWORK_REVISION == "V10.17_CRT_PHASE_FAMILY_TERMINAL"
    assert TARGET_PROVED_PCSSC_MIXED_PHASE_TERMINALS_V10_17.endswith("V10_17")
    assert PCSSC_GUARDED_COMPLETION_THEOREM_V10_17.endswith("V10_17")
    phase_source = inspect.getsource(pcssc._phase_block_postfix_search_v10_17)
    exact_source = inspect.getsource(exact_crt_phase_family_pre_hi_max)
    assert "for q in range" not in phase_source
    assert "range(int(plan.joint_period))" not in exact_source
    assert "completion_bounds" not in exact_source


def test_v10_17_formula_hash_binds_completion_bounds_and_target_cap():
    spec = CarryTaskSpec("hp", "HI", 7, 2, 4)
    projections = real_phase_block_task_projections(
        10, 1, 0, (spec,), block_modulus=1, block_residue=0
    )

    class Target:
        name = "hi0"
        actual_demand_upper = 8
        c_lo = 5
        c_hi = 8

    details = {"candidate_domain_kind": "PROVED_BOUNDARY_UNION", "busy_horizon": 5}
    c1 = pcssc._phase_block_binding_context(
        Target(), _case(), (spec,), carry_mode="R7_INTERSECT_COMPLETION", completion_bounds=(3,)
    )
    c2 = pcssc._phase_block_binding_context(
        Target(), _case(), (spec,), carry_mode="R7_INTERSECT_COMPLETION", completion_bounds=(4,)
    )
    b1 = pcssc._phase_block_binding(projections, details, c1)
    b2 = pcssc._phase_block_binding(projections, details, c2)
    assert b1["service_cap_binding_hash"] != b2["service_cap_binding_hash"]
    assert b1["formula_hash"] != b2["formula_hash"]


def test_v10_17_crt_search_schedules_root_probe_and_nonroot_recurrence(monkeypatch):
    spec = CarryTaskSpec("hp", "HI", 4, 1, 1)
    monkeypatch.setattr(pcssc, "_carry_task_specs", lambda *args: (spec,))
    monkeypatch.setattr(pcssc, "_target_cap", lambda *args: 4)
    monkeypatch.setattr(
        pcssc,
        "target_release_joint_phase_parameters",
        lambda *args, **kwargs: (0, 1, 4),
    )
    monkeypatch.setattr(pcssc, "phase_block_task_projections", real_phase_block_task_projections)
    monkeypatch.setattr(
        pcssc,
        "phase_block_r7_carry_upper",
        lambda *args: (
            20,
            {
                "candidate_domain_kind": "PROVED_BOUNDARY_UNION",
                "busy_horizon": 4,
                "candidate_boundary_count": 1,
                "witness_length": 1,
            },
        ),
    )
    monkeypatch.setattr(pcssc, "phase_block_post_switch_future_upper", lambda *args: 0)
    monkeypatch.setattr(
        pcssc,
        "_controller_prefix_coverage_receipt",
        lambda *args, **kwargs: {"obligation_id": "PREFIX", "status": "PASS"},
    )

    calls = []

    def fake_crt(*args, **kwargs):
        block = args[3]
        calls.append((block.modulus, block.residue, kwargs["recurrence_after_deadline_failure"]))
        if block.modulus == 1:
            return None, [{"stage": "DEADLINE_DIRECT_PROBE"}], "CRT_PHASE_FAMILY_POSTFIX_FAIL:ROOT"
        return ({
            **block.as_dict(),
            "terminal_type": "CRT_PHASE_FAMILY",
            "R": 7,
            "W": 7,
            "formula_hash": f"f{block.residue}",
            "candidate_steps": 2,
            "family_size": 2,
            "component_periods": [2],
            "streaming_residue_work": 1,
            "receipts": [{
                "obligation_id": f"CRT_PHASE_FAMILY_DIRECT_POSTFIX::{block.id}",
                "status": "PASS",
            }],
        }, [{"stage": "CANDIDATE_RECURRENCE"}], None)

    monkeypatch.setattr(pcssc, "_crt_phase_family_postfix_search_v10_17", fake_crt)

    class Target:
        name = "hi0"
        period = 1
        deadline = 10
        priority = 1
        criticality = "HI"
        actual_demand_upper = 4
        c_lo = 2
        c_hi = 4

    class HP:
        name = "hp"
        period = 4
        priority = 0
        criticality = "HI"

    model = _Model()
    model.tasks = (HP(), Target())
    cert, _, failure = pcssc._phase_block_postfix_search_v10_17(
        model, Target(), (HP(),), object(), {}, _case()
    )

    assert failure is None
    assert cert is not None
    assert calls[0] == (1, 0, False)
    assert {row for row in calls[1:]} == {(2, 0, True), (2, 1, True)}
    assert cert["crt_phase_family_terminal_count"] == 2
