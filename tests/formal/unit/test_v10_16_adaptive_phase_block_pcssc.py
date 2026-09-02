import inspect
import sys
import types

_fake_z3_added = "z3" not in sys.modules
if _fake_z3_added:
    sys.modules["z3"] = types.ModuleType("z3")

from formal_toolchain.v10_1 import pcssc
from formal_toolchain.v10_1.carry_in_envelope import (
    CarryTaskSpec,
    phase_block_task_projections as real_phase_block_task_projections,
)
from formal_toolchain.v10_1.completion_certificates import (
    PCSSC_REFINED_CASE_COMPLETION_THEOREM_V10_16,
)
from formal_toolchain.v10_1.constants import (
    FRAMEWORK_REVISION,
    TARGET_PROVED_PCSSC_REFINED_CASES_V10_16,
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


class _Model:
    agent_period = 5


class _Target10:
    name = "hi0"
    period = 10
    deadline = 10


class _HP:
    name = "hp"
    period = 2


class _TargetSplit:
    name = "hi0"
    period = 1
    deadline = 10


def test_v10_16_empty_ahead_has_q1_root_only(monkeypatch):
    monkeypatch.setattr(pcssc, "_carry_task_specs", lambda *args: ())
    monkeypatch.setattr(pcssc, "_target_cap", lambda *args: 4)
    monkeypatch.setattr(
        pcssc,
        "_controller_prefix_coverage_receipt",
        lambda *args, **kwargs: {"obligation_id": "PREFIX", "status": "PASS"},
    )

    cert, attempts, failure = pcssc._phase_block_postfix_search_v10_16(
        _Model(), _Target10(), (), object(), {}, _case()
    )

    assert failure is None
    assert cert is not None
    assert cert["R"] == 4
    assert cert["phase_block_joint_period"] == 1
    assert cert["phase_block_leaf_count"] == 1
    assert cert["global_q_enumerated"] is False
    assert attempts == [{
        "block_id": "M1_A0", "M": 1, "a": 0, "depth": 0,
        "parent_id": None, "split_factor": None, "status": "PASS",
        "R": 4, "W": 4, "projection_sizes": [], "candidate_steps": 1,
    }]
    root = next(
        row for row in cert["phase_block_receipts"]
        if row["obligation_id"].startswith("PHASE_BLOCK_ROOT_DOMAIN::")
    )
    assert root["Q"] == 1
    assert root["lcm_empty_convention"] == 1
    assert root["empty_ahead"] is True


def test_v10_16_failed_root_refines_and_aggregates_independent_leaf_postfixes(monkeypatch):
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
        if projection.phase_count > 1:
            carry = 20
        elif projection.phase_residue == 0:
            carry = 2
        else:
            carry = 5
        return carry, {
            "candidate_domain_kind": "PROVED_BOUNDARY_UNION",
            "busy_horizon": 2,
            "candidate_boundary_count": 1,
            "witness_length": 1,
        }

    monkeypatch.setattr(pcssc, "phase_block_r7_carry_upper", fake_r7)
    monkeypatch.setattr(pcssc, "phase_block_post_switch_future_upper", lambda *args: 0)
    monkeypatch.setattr(
        pcssc,
        "_controller_prefix_coverage_receipt",
        lambda *args, **kwargs: {"obligation_id": "PREFIX", "status": "PASS"},
    )

    cert, attempts, failure = pcssc._phase_block_postfix_search_v10_16(
        _Model(), _TargetSplit(), (_HP(),), object(), {}, _case()
    )

    assert failure is None
    assert cert is not None
    assert cert["R"] == 9
    assert cert["phase_block_leaf_count"] == 2
    assert cert["uniform_R_is_common_postfix"] is False
    assert cert["case_theorem_basis"] == "V10_16_ADAPTIVE_PHASE_BLOCK"
    assert attempts[0]["status"] == "FAILED_BLOCK"
    assert {row["block_id"] for row in attempts[1:]} == {"M2_A0", "M2_A1"}
    assert all(row["status"] == "PASS" for row in attempts[1:])

    obligations = [row["obligation_id"] for row in cert["phase_block_receipts"]]
    assert any(value.startswith("PHASE_BLOCK_SPLIT::") for value in obligations)
    assert sum(value.startswith("PHASE_BLOCK_WORKLOAD_LIFTING_SOUND::") for value in obligations) == 2
    assert sum(value.startswith("PHASE_BLOCK_POSTFIX::") for value in obligations) == 2
    assert any(value.startswith("PHASE_BLOCK_LEAF_COVERAGE::") for value in obligations)
    postfixes = [
        row for row in cert["phase_block_receipts"]
        if row["obligation_id"].startswith("PHASE_BLOCK_POSTFIX::")
    ]
    assert {row["R_B"] for row in postfixes} == {6, 9}
    assert all(row["direct_recheck"] for row in postfixes)


def test_v10_16_structural_budget_fails_closed_without_singleton_fallback(monkeypatch):
    spec = CarryTaskSpec("hp", "HI", 2, 1, 1)
    monkeypatch.setattr(pcssc, "_carry_task_specs", lambda *args: (spec,))
    monkeypatch.setattr(pcssc, "_target_cap", lambda *args: 4)
    monkeypatch.setattr(
        pcssc,
        "target_release_joint_phase_parameters",
        lambda *args, **kwargs: (0, 2, 1),
    )
    monkeypatch.setattr(pcssc, "phase_block_r7_carry_upper", lambda *args: (
        20,
        {
            "candidate_domain_kind": "PROVED_BOUNDARY_UNION",
            "busy_horizon": 2,
            "candidate_boundary_count": 1,
            "witness_length": 1,
        },
    ))
    monkeypatch.setattr(pcssc, "phase_block_post_switch_future_upper", lambda *args: 0)

    cert, attempts, failure = pcssc._phase_block_postfix_search_v10_16(
        _Model(), _Target10(), (_HP(),), object(), {}, _case()
    )

    assert cert is None
    assert attempts[-1]["status"] == "FAILED_BLOCK"
    assert failure is not None
    assert failure.startswith("PHASE_BLOCK_REFINEMENT_INSUFFICIENT:")
    assert "NO_LEGAL_SPLIT" in failure


def test_v10_16_identifiers_and_default_terminal_remove_global_q_loop():
    assert FRAMEWORK_REVISION == "V10.16_ADAPTIVE_PHASE_BLOCK_PCSSC"
    assert TARGET_PROVED_PCSSC_REFINED_CASES_V10_16.endswith("V10_16")
    assert PCSSC_REFINED_CASE_COMPLETION_THEOREM_V10_16.endswith("V10_16")
    source = inspect.getsource(pcssc._phase_block_postfix_search_v10_16)
    assert "for q in range" not in source
    assert "target_release_joint_phases_at_q" not in source
    assert "W_child" not in source


def test_v10_16_outer_case_aggregation_exports_only_unified_target_bound(monkeypatch):
    case = _case()
    monkeypatch.setattr(
        pcssc,
        "_deadline_canonical_case_domain",
        lambda *args, **kwargs: ((case,), (), "d" * 64),
    )
    monkeypatch.setattr(
        pcssc,
        "_case_postfix_search",
        lambda *args, **kwargs: (
            None,
            [{"R": 10, "W": 11, "postfixed": False}],
            f"CASE_POSTFIX_NOT_FOUND_BY_DEADLINE:{case.id}:W=11:D=10",
        ),
    )
    phase_receipts = [{
        "obligation_id": f"PHASE_BLOCK_POSTFIX::{case.id},M2_A0",
        "status": "PASS",
        "R_B": 7,
        "W_B_R_B": 7,
    }]
    monkeypatch.setattr(
        pcssc,
        "_phase_block_postfix_search_v10_16",
        lambda *args, **kwargs: ({
            **case.as_dict(),
            "status": "PASS",
            "R": 7,
            "W": 7,
            "candidate_path": [{"block_id": "M2_A0", "status": "PASS"}],
            "controller_prefix_receipt": {"obligation_id": "PREFIX", "status": "PASS"},
            "case_theorem_basis": "V10_16_ADAPTIVE_PHASE_BLOCK",
            "phase_block_joint_period": 2,
            "phase_block_leaf_count": 1,
            "phase_block_leaf_digest_sha256": "a" * 64,
            "worst_phase_block": "M2_A0",
            "uniform_R_is_common_postfix": False,
            "phase_block_receipts": phase_receipts,
            "global_q_enumerated": False,
        }, [{"block_id": "M2_A0", "status": "PASS"}], None),
    )

    class Target:
        name = "hi0"
        deadline = 10

    bound, tested, receipts, failure = pcssc._case_consistent_postfix_search(
        object(), Target(), (), object(), set(), {}
    )

    assert failure is None
    assert bound == 7
    ids = [row["obligation_id"] for row in receipts]
    assert f"PCSSC_REFINED_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_16::hi0" in ids
    assert f"PHASE_BLOCK_POSTFIX::{case.id},M2_A0" in ids
    assert not any(value.startswith("CASE_WORKLOAD_DOMINANCE::hi0") for value in ids)
    all_cases = next(row for row in receipts if row["obligation_id"] == "ALL_CASES_POSTFIX_COVERED::hi0")
    assert all_cases["Rcert"] == 7
    assert all_cases["v10_16_adaptive_phase_block_case_ids"] == [case.id]
    assert tested[0]["case_theorem_basis"] == "V10_16_ADAPTIVE_PHASE_BLOCK"


def test_v10_16_formula_hash_binds_completion_bounds_and_target_cap():
    spec = CarryTaskSpec("hp", "HI", 7, 2, 4)
    projections = real_phase_block_task_projections(
        10, 1, 0, (spec,), block_modulus=1, block_residue=0
    )

    class Target:
        name = "hi0"
        actual_demand_upper = 8
        c_lo = 5
        c_hi = 8

    details = {
        "candidate_domain_kind": "PROVED_BOUNDARY_UNION",
        "busy_horizon": 5,
    }
    b1 = pcssc._phase_block_binding(
        Target(), _case(), (spec,), projections, details,
        carry_mode="R7_INTERSECT_COMPLETION", completion_bounds=(3,),
    )
    b2 = pcssc._phase_block_binding(
        Target(), _case(), (spec,), projections, details,
        carry_mode="R7_INTERSECT_COMPLETION", completion_bounds=(4,),
    )
    assert b1["service_cap_binding_hash"] != b2["service_cap_binding_hash"]
    assert b1["formula_hash"] != b2["formula_hash"]
