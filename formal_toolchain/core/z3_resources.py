"""Small fail-closed Z3 resource helpers for the research verifier.

The V10.1 proof route intentionally has no wall-clock solver timeout.  A slow
proof obligation is allowed to finish so solver latency cannot be confused
with a mathematical UNRESOLVED result.  An optional process-wide memory cap is
still supported because memory exhaustion remains fail-closed and is not a
clock-based proof cutoff.
"""

from __future__ import annotations

import os
from typing import Any


def _positive_int_env(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def configure_z3(z3: Any) -> None:
    """Apply the optional process-wide memory cap; never set a time limit."""

    memory_mb = _positive_int_env("AMC_FORMAL_Z3_MEMORY_MB")
    if memory_mb:
        z3.set_param("memory_max_size", memory_mb)


def new_context(z3: Any) -> Any:
    configure_z3(z3)
    return z3.Context()


def new_solver(z3: Any, *, context: Any) -> Any:
    configure_z3(z3)
    return z3.Solver(ctx=context)
