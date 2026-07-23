"""Small fail-closed Z3 resource helpers.

The proof semantics are unchanged.  Solvers receive a fresh context so native
AST allocations can be released after each proof query.  Optional process-wide
limits are controlled only by environment variables; hitting a limit yields
``unknown``/an exception and therefore never creates a false PASS.
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


def configure_z3(z3: Any) -> int:
    """Apply optional limits and return the per-solver timeout in ms."""

    memory_mb = _positive_int_env("AMC_FORMAL_Z3_MEMORY_MB")
    if memory_mb:
        # Z3 documents memory_max_size in MB.  This is fail-closed: resource
        # exhaustion cannot be interpreted as a proof.
        z3.set_param("memory_max_size", memory_mb)
    return _positive_int_env("AMC_FORMAL_Z3_TIMEOUT_MS")


def new_context(z3: Any) -> Any:
    configure_z3(z3)
    return z3.Context()


def new_solver(z3: Any, *, context: Any) -> Any:
    timeout_ms = configure_z3(z3)
    solver = z3.Solver(ctx=context)
    if timeout_ms:
        solver.set(timeout=timeout_ms)
    return solver
