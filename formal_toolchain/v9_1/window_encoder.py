"""Finite first-HI-bad-window encodings for the V9.1 route."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

import z3

from .environment_encoder import declare_environment, target_release_constraints
from .safe_prefix_invariant import SafePrefixInvariant
from .symbolic_state import BoundModel, SymbolicKernelState, declare_state
from .transition_encoder import encode_phase_step


ENCODER_VERSION = "V9_1_KERNEL_ENCODER_V2_TWO_SLOT_CARRY_IN"


def _implementation_readiness() -> tuple[bool, tuple[str, ...]]:
    from .readiness import BLOCKERS, proof_pipeline_ready
    return proof_pipeline_ready(), tuple(row.code for row in BLOCKERS)


@dataclass(frozen=True, slots=True)
class WindowEncoding:
    target_task: str
    deadline: int
    formula: z3.BoolRef
    states: tuple[SymbolicKernelState, ...]
    environment: Any
    source_obligations: tuple[str, ...]

    def smt2(self) -> str:
        solver = z3.Solver()
        solver.add(self.formula)
        return solver.sexpr()


def _pre_deadline_state(states: tuple[SymbolicKernelState, ...], deadline: int) -> SymbolicKernelState:
    return states[deadline * 8 + 2]


def _post_deadline_state(states: tuple[SymbolicKernelState, ...], deadline: int) -> SymbolicKernelState:
    return states[deadline * 8 + 3]


def required_new_release_slots(model: BoundModel, *, window_length: int) -> int:
    """Compatibility metric; recycling makes the result independent of slots."""

    if window_length < 0:
        raise ValueError("window_length must be non-negative")
    # One exact slot plus one LO aggregate slot is the complete finite model.
    return 2 if any(task.criticality == "LO" for task in model.tasks) else 1


def build_first_bad_window(
    model: BoundModel,
    invariant: SafePrefixInvariant,
    target_task: str,
) -> WindowEncoding:
    task = model.task_by_name.get(target_task)
    if task is None or task.criticality != "HI":
        raise ValueError("FIRST_BAD_WINDOW_TARGET_MUST_BE_HI")
    if any(row.deadline > row.period for row in model.tasks):
        raise ValueError("V9_1_TWO_SLOT_CARRY_IN_REQUIRES_D_LE_T")
    required_slots = required_new_release_slots(model, window_length=int(task.deadline))
    if model.max_jobs_per_task < required_slots:
        raise ValueError(
            f"WINDOW_TWO_SLOT_CAPACITY_INSUFFICIENT: need {required_slots}, "
            f"have {model.max_jobs_per_task}"
        )

    deadline = int(task.deadline)
    # The origin is a target release, so origin % T_target == 0.  A release of
    # task k can occur at relative tick r only if gcd(T_target,T_k) divides r.
    # Declare demand variables only on those potentially reachable ticks; P3's
    # coverage implication still prevents any due release from escaping the
    # universally quantified demand domain.
    allowed_ticks = {
        row.name: tuple(
            tick for tick in range(deadline + 1)
            if tick % gcd(task.period, row.period) == 0
        )
        for row in model.tasks
    }
    env = declare_environment(
        "window.env", model, release_count=deadline + 1,
        allowed_ticks_by_task=allowed_ticks,
    )
    # +4 includes the post-P2 (p=3) target state where the sticky miss ledger
    # has actually been updated.
    states = tuple(declare_state(f"window.z.{index}", model) for index in range(deadline * 8 + 4))
    clauses: list[z3.BoolRef] = []
    z0 = states[0]
    clauses.extend((
        *env.constraints,
        env.phase.origin_time == z0.t,
        invariant.formula(z0),
        z0.hi_miss_ledger == 0,
        z0.p == 0,
    ))
    clauses.extend(target_release_constraints(env, task))
    # The target is release-eligible at the arbitrary absolute origin.
    clauses.append(z0.eta[target_task] == task.period)

    controller_stride = gcd(model.agent_period, task.period)
    for index in range(len(states) - 1):
        phase = index % 8
        relative_tick = index // 8
        controller_may_fire = (relative_tick % controller_stride == 0)
        clauses.append(encode_phase_step(
            states[index], states[index + 1], model, env,
            phase=phase, controller_may_fire=controller_may_fire,
        ))

    # Check the ledger *after* every earlier P2 observation.  This rules out an
    # earlier simultaneous or single HI miss, rather than checking the pre-P2
    # state where the ledger has not yet been updated.
    for tick in range(deadline):
        clauses.append(states[tick * 8 + 3].hi_miss_ledger == 0)

    pre = _pre_deadline_state(states, deadline)
    post = _post_deadline_state(states, deadline)
    target_slots = [pre.jobs[(target_task, slot)] for slot in range(model.max_jobs_per_task)]
    target_at_deadline = z3.Or(*(
        job.present
        & (job.absolute_deadline == pre.t)
        & (job.executed_service < job.effective_demand)
        for job in target_slots
    ))
    clauses.extend((
        pre.t == env.phase.origin_time + deadline,
        pre.p == 2,
        pre.hi_miss_ledger == 0,
        target_at_deadline,
        post.t == pre.t,
        post.p == 3,
        post.hi_miss_ledger >= 1,
    ))
    return WindowEncoding(
        target_task=target_task,
        deadline=deadline,
        formula=z3.And(*clauses),
        states=states,
        environment=env,
        source_obligations=(
            "window_start_requires_arbitrary_psi_no_prior_miss_and_target_eligibility",
            "absolute_periodic_origin_consumed_by_release_and_controller",
            "two_slot_lo_aggregate_preserves_arbitrary_carry_in_work",
            "single_exact_hi_slot_sound_under_D_le_T_and_no_prior_miss",
            "no_earlier_hi_miss_checked_after_each_deadline_observation",
            "target_deadline_post_observe_ledger_increment_encoded",
            "sparse_but_complete_relative_release_demand_universal_domain",
            "known_phase_unroll_avoids_dead_transition_branches",
            "controller_formula_only_on_gcd-compatible_relative_ticks",
        ),
    )


def write_first_bad_window(encoding: WindowEncoding, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = encoding.smt2()
    path.write_text(text, encoding="utf-8")
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ENCODER_COMPLETE, ENCODER_READINESS_GAPS = _implementation_readiness()


__all__ = ["ENCODER_COMPLETE", "ENCODER_READINESS_GAPS", "ENCODER_VERSION", "WindowEncoding",
           "build_first_bad_window", "required_new_release_slots", "write_first_bad_window"]
