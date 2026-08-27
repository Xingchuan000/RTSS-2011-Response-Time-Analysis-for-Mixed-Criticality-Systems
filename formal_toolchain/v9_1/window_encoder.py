"""Finite first-HI-bad-window encodings for the V9.1 route."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import z3

from .environment_encoder import declare_environment, target_release_constraints
from .safe_prefix_invariant import SafePrefixInvariant
from .symbolic_state import BoundModel, SymbolicKernelState, declare_state
from .transition_encoder import encode_step


ENCODER_VERSION = "V9_1_KERNEL_ENCODER_V1"


def _implementation_readiness() -> tuple[bool, tuple[str, ...]]:
    """Derive the gate from implemented generators, never from a status flag.

    The current repository still lacks a source-level real-seed replay provider;
    that gap is intentionally part of the computed result.  Once that provider
    and its fresh replay are present, this function is the only place that may
    open the gate.
    """
    required = (build_first_bad_window, write_first_bad_window)
    gaps = [] if all(callable(item) for item in required) else ["WINDOW_GENERATOR_MISSING"]
    gaps.append("REAL_SEED_CONCRETE_REPLAY_PROVIDER_UNBOUND")
    return not gaps, tuple(gaps)


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


def _target_state(states: tuple[SymbolicKernelState, ...], deadline: int) -> SymbolicKernelState:
    return states[deadline * 8 + 2]


def build_first_bad_window(
    model: BoundModel,
    invariant: SafePrefixInvariant,
    target_task: str,
) -> WindowEncoding:
    task = model.task_by_name.get(target_task)
    if task is None or task.criticality != "HI":
        raise ValueError("FIRST_BAD_WINDOW_TARGET_MUST_BE_HI")
    deadline = int(task.deadline)
    env = declare_environment("window.env", model, release_count=max(model.max_jobs_per_task, deadline + 1))
    states = tuple(declare_state(f"window.z.{index}", model) for index in range(deadline * 8 + 3))
    clauses: list[z3.BoolRef] = []
    z0 = states[0]
    clauses.extend((invariant.formula(z0), z0.hi_miss_ledger == 0,
                    z0.eta[target_task] == task.period,
                    z0.t == 0, z0.p == 0))
    clauses.extend(env.constraints)
    clauses.extend(target_release_constraints(env, task))
    for index in range(len(states) - 1):
        clauses.append(encode_step(states[index], states[index + 1], model, env))

    # Strictly before the target deadline's P2 observation, no HI miss may have
    # entered the sticky ledger.  We do not constrain the target timestamp's
    # other HI jobs: simultaneous first misses are allowed by the theorem.
    for time in range(deadline):
        clauses.append(states[time * 8 + 2].hi_miss_ledger == 0)
    observed = _target_state(states, deadline)
    target_slots = [observed.jobs[(target_task, slot)] for slot in range(model.max_jobs_per_task)]
    target_at_deadline = z3.Or(*(job.present & (job.absolute_deadline == observed.t) &
                                 (job.executed_service < job.effective_demand)
                                 for job in target_slots))
    clauses.extend((observed.t == deadline, observed.p == 2, target_at_deadline,
                    observed.hi_miss_ledger >= 1))
    return WindowEncoding(
        target_task=target_task,
        deadline=deadline,
        formula=z3.And(*clauses),
        states=states,
        environment=env,
        source_obligations=(
            "window_start_requires_psi_no_prior_miss_and_target_eligibility",
            "no_earlier_hi_miss_strictly_before_target_timestamp",
            "target_deadline_observe_encoded",
            "per_release_actual_demand_universal_domain",
        ),
    )


def write_first_bad_window(encoding: WindowEncoding, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = encoding.smt2()
    path.write_text(text, encoding="utf-8")
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Task J gate: automatic and fail-closed until every source-level obligation is
# bound.  Do not replace this with a hand-maintained True constant.
ENCODER_COMPLETE, ENCODER_READINESS_GAPS = _implementation_readiness()


__all__ = ["ENCODER_COMPLETE", "ENCODER_READINESS_GAPS", "ENCODER_VERSION", "WindowEncoding",
           "build_first_bad_window", "write_first_bad_window"]
