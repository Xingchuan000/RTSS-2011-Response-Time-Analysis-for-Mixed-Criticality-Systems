import sys
import types

_fake_z3_added = "z3" not in sys.modules
if _fake_z3_added:
    sys.modules["z3"] = types.ModuleType("z3")

from formal_toolchain.v10_1 import pcssc
from formal_toolchain.v10_1.carry_in_envelope import (
    CarryTaskSpec,
    fixed_phase_pre_hi_interference,
    target_release_joint_phase_parameters,
    target_release_joint_phases_at_q,
)
from formal_toolchain.v10_1.completion_certificates import (
    PCSSC_REFINED_CASE_COMPLETION_THEOREM_V10_14,
)
from formal_toolchain.v10_1.constants import (
    FRAMEWORK_REVISION,
    TARGET_PROVED_PCSSC_REFINED_CASES_V10_14,
)

if _fake_z3_added:
    sys.modules.pop("z3", None)


def test_v10_14_joint_phase_parameters_stream_same_orbit_as_q_formula():
    specs = (
        CarryTaskSpec("a", "HI", 7, 2, 4),
        CarryTaskSpec("b", "LO", 9, 3, 1),
    )
    n0, step, cycle = target_release_joint_phase_parameters(12, 5, 0, specs)
    assert cycle > 0
    seen = set()
    for q in range(cycle):
        phases = target_release_joint_phases_at_q(
            12, specs, n0=n0, q_step=step, q=q
        )
        assert len(phases) == 2
        assert 0 <= phases[0] < 7
        assert 0 <= phases[1] < 9
        seen.add(phases)
    assert len(seen) == cycle


def test_fixed_phase_pre_hi_interference_completion_is_sound_intersection():
    specs = (
        CarryTaskSpec("h", "HI", 10, 3, 6),
        CarryTaskSpec("l", "LO", 15, 5, 2),
    )
    phases = (4, 7)
    raw, raw_details = fixed_phase_pre_hi_interference(specs, phases, 20, None)
    protected, protected_details = fixed_phase_pre_hi_interference(
        specs, phases, 20, (8, 10)
    )
    assert protected <= raw
    assert protected_details["carry_in"] <= raw_details["carry_in"]


def test_v10_14_pre_hi_phase_consistent_search_aggregates_per_q_postfix(monkeypatch):
    class Target:
        name = "hi0"
        period = 20
        deadline = 20

    class Switch:
        kind = "PRE_HI"
        id = "PRE_HI"

    class Case:
        theta = 0
        switch = Switch()
        target_classification = "ABNORMAL"
        id = "THETA_0__PRE_HI__TARGET_ABNORMAL"
        canonical_deadline = 20

        def as_dict(self):
            return {
                "case_id": self.id,
                "theta": 0,
                "switch_kind": "PRE_HI",
                "switch_lower": None,
                "switch_upper": None,
                "target_classification": "ABNORMAL",
                "canonical_deadline": 20,
            }

    spec = CarryTaskSpec("hp", "HI", 7, 2, 4)
    monkeypatch.setattr(pcssc, "_carry_task_specs", lambda *args: (spec,))
    monkeypatch.setattr(pcssc, "_target_cap", lambda *args: 5)
    monkeypatch.setattr(
        pcssc,
        "target_release_joint_phase_parameters",
        lambda *args: (0, 1, 2),
    )
    monkeypatch.setattr(
        pcssc,
        "target_release_joint_phases_at_q",
        lambda *args, q, **kwargs: (q,),
    )

    # q=0 closes at R=9, q=1 closes at R=12.  A single common q-independent
    # recurrence is intentionally not used; V10.14 aggregates max(9,12)=12.
    monkeypatch.setattr(
        pcssc,
        "fixed_phase_pre_hi_carry",
        lambda specs, phases, completion: (0, {"carry_in": 0}),
    )
    monkeypatch.setattr(
        pcssc,
        "fixed_phase_post_switch_future_work",
        lambda specs, phases, horizon: (4 if phases[0] == 0 else 7),
    )
    monkeypatch.setattr(
        pcssc,
        "_controller_prefix_coverage_receipt",
        lambda *args, **kwargs: {"obligation_id": "PREFIX", "status": "PASS"},
    )

    class HP:
        name = "hp"
        period = 7

    class Model:
        agent_period = 5

    cert, path, failure = pcssc._pre_hi_phase_consistent_postfix_search_v10_14(
        Model(), Target(), (HP(),), object(), {}, Case()
    )
    assert failure is None
    assert cert is not None
    assert cert["R"] == 12
    assert cert["phase_subcase_count"] == 2
    assert cert["phase_subcase_covered_count"] == 2
    assert cert["uniform_R_is_common_postfix"] is False
    assert cert["case_theorem_basis"] == "V10_14_PRE_HI_PHASE_CONSISTENT"
    assert len(cert["phase_subcase_digest_sha256"]) == 64
    assert path[-1]["postfixed"] is True


def test_v10_14_pre_hi_phase_consistent_fails_closed_if_any_q_misses(monkeypatch):
    class Target:
        name = "hi0"
        period = 10
        deadline = 10

    class Switch:
        kind = "PRE_HI"
        id = "PRE_HI"

    class Case:
        theta = 0
        switch = Switch()
        target_classification = "ABNORMAL"
        id = "C"
        canonical_deadline = 10

        def as_dict(self):
            return {"case_id": self.id}

    spec = CarryTaskSpec("hp", "HI", 7, 2, 4)
    monkeypatch.setattr(pcssc, "_carry_task_specs", lambda *args: (spec,))
    monkeypatch.setattr(pcssc, "_target_cap", lambda *args: 4)
    monkeypatch.setattr(pcssc, "target_release_joint_phase_parameters", lambda *args: (0, 1, 2))
    monkeypatch.setattr(
        pcssc, "target_release_joint_phases_at_q", lambda *args, q, **kwargs: (q,)
    )
    monkeypatch.setattr(
        pcssc,
        "fixed_phase_pre_hi_carry",
        lambda specs, phases, completion: (0, {"carry_in": 0}),
    )
    monkeypatch.setattr(
        pcssc,
        "fixed_phase_post_switch_future_work",
        lambda specs, phases, horizon: (5 if phases[0] == 0 else 20),
    )
    monkeypatch.setattr(
        pcssc,
        "_controller_prefix_coverage_receipt",
        lambda *args, **kwargs: {"obligation_id": "PREFIX", "status": "PASS"},
    )

    class HP:
        name = "hp"
        period = 7

    class Model:
        agent_period = 5

    cert, path, failure = pcssc._pre_hi_phase_consistent_postfix_search_v10_14(
        Model(), Target(), (HP(),), object(), {}, Case()
    )
    assert cert is None
    assert "POSTFIX_NOT_FOUND_BY_DEADLINE" in failure
    assert "q=1" in failure


def test_v10_14_identifiers_and_completion_export_are_distinct():
    assert FRAMEWORK_REVISION == "V10.14_PRE_HI_PHASE_CONSISTENT_POSTFIX"
    assert TARGET_PROVED_PCSSC_REFINED_CASES_V10_14.endswith("V10_14")
    assert PCSSC_REFINED_CASE_COMPLETION_THEOREM_V10_14.endswith("V10_14")
