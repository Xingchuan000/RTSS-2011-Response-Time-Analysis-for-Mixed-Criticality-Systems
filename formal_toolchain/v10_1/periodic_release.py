"""Pure arithmetic for V10.1 exact-periodic release profiles.

The frozen environment binds every task to phase-zero exact periodic releases.
For a response window translated to a target release, the absolute release
index is unknown; conditioning on the target-relative controller phase leaves a
finite modular orbit of compatible higher-priority-task release phases.  This
module computes that orbit without SMT.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd


@lru_cache(maxsize=32768)
def compatible_release_phases(
    target_period: int,
    task_period: int,
    controller_period: int,
    theta: int,
) -> tuple[int, ...]:
    """Return every hp-task phase compatible with one controller phase.

    Let the target release be ``t0 = n*T_i``.  In target-relative time,
    ``theta = (-t0) mod T_c`` and the hp task's first non-negative release is at
    ``phi = (-t0) mod T_j``.  Conditioning on theta determines ``n`` modulo
    ``T_c/gcd(T_i,T_c)``; one modular orbit then enumerates exactly all possible
    phi values.
    """

    ti = int(target_period)
    tj = int(task_period)
    tc = int(controller_period)
    th = int(theta)
    if ti <= 0 or tj <= 0 or tc <= 0:
        raise ValueError("EXACT_PERIODIC_PERIOD_NONPOSITIVE")

    g = gcd(ti, tc)
    if th < 0 or th >= tc or th % g != 0:
        raise ValueError(f"EXACT_PERIODIC_CONTROLLER_PHASE_INCOMPATIBLE:{th}")

    modulus = tc // g
    if modulus == 1:
        n0 = 0
    else:
        # gcd(ti/g, tc/g)=1, so the inverse always exists.
        a = (ti // g) % modulus
        rhs = (-th // g) % modulus
        n0 = (rhs * pow(a, -1, modulus)) % modulus

    step = (modulus * ti) % tj
    orbit = 1 if step == 0 else tj // gcd(tj, step)
    phases = {(-((n0 + q * modulus) * ti)) % tj for q in range(orbit)}
    return tuple(sorted(int(value) for value in phases))


def p7_eta_residue_counterexample(period: int) -> int | None:
    """Return a residue violating exact-periodic eta preservation, if any.

    At P7 the invariant represents ``eta`` as ``r=t mod T`` (including zero).
    P7 advances one tick and increments eta.  At successor P0, residue zero is
    represented as ``T`` so a just-due release can be consumed by P0--P3.
    Exhausting ``r in [0,T)`` is therefore a complete finite proof of the
    phase-change arithmetic, independent of every other kernel field.
    """

    t = int(period)
    if t <= 0:
        raise ValueError("EXACT_PERIODIC_PERIOD_NONPOSITIVE")
    for residue in range(t):
        eta_before = residue
        eta_after = eta_before + 1 if eta_before < t else t
        next_residue = (residue + 1) % t
        eta_expected = t if next_residue == 0 else next_residue
        if eta_after != eta_expected:
            return residue
    return None


__all__ = ["compatible_release_phases", "p7_eta_residue_counterexample"]
