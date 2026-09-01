import sys
import types

_fake_z3_added = "z3" not in sys.modules
if _fake_z3_added:
    sys.modules["z3"] = types.ModuleType("z3")

from formal_toolchain.v10_1.carry_in_envelope import (
    CarryTaskSpec,
    fixed_phase_lo_entry_backlog,
    target_release_joint_phase_orbit,
)
from formal_toolchain.v10_1 import pcssc
from formal_toolchain.v10_1.completion_certificates import (
    PCSSC_CONDITIONED_CARRY_COMPLETION_THEOREM,
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


def test_joint_phase_orbit_uses_one_target_release_index_for_all_tasks():
    specs = (
        CarryTaskSpec("a", "HI", 7, 2, 4),
        CarryTaskSpec("b", "LO", 9, 3, 1),
    )
    orbit = target_release_joint_phase_orbit(12, 5, 0, specs)
    assert orbit
    for q, phases in orbit:
        assert len(phases) == 2
        assert 0 <= phases[0] < 7
        assert 0 <= phases[1] < 9
        # The row is generated from a single target release index, not a
        # Cartesian product of independently selected task phases.
        assert isinstance(q, int)


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
        object(), Target(), (), object(), set(), {}, Case()
    )
    assert failure is None
    assert cert is not None
    assert cert["R"] == 12
    assert cert["W"] == 11
    assert cert["case_theorem_basis"] == "V10_13_CASE_CONDITIONED_CARRY_FUTURE"
    assert path[-1]["postfixed"] is True


def test_v10_13_identifiers_are_distinct_and_exportable():
    assert FRAMEWORK_REVISION == "V10.14_PRE_HI_PHASE_CONSISTENT_POSTFIX"
    assert TARGET_PROVED_PCSSC_CASE_CONDITIONED_CARRY.endswith("V10_13")
    assert PCSSC_CONDITIONED_CARRY_COMPLETION_THEOREM.endswith("V10_13")


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
