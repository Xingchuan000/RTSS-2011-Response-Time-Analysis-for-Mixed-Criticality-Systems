"""SMT2 query generators for the PP0 primitive transition obligations.

Uses the PP0 Transition IR and pp0_smt_encoder to generate code-bound SMT2
queries.  Each query includes domain, guard, state, frame, and time constraints
from the IR.  Free-variable queries (no transition equations) are classified
as SCHEMA_ONLY_NOT_CODE_BOUND and cannot prove PP0 conformance.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object

from .pp0_smt_encoder import (
    generate_code_bound_queries,
    is_trivial_query_source,
)


def generate_all_queries() -> dict[str, dict[str, Any]]:
    """Generate all PP0 SMT2 queries with IR binding.

    Each query is generated from the PP0TransitionIR via the SMT encoder.
    Queries without code-bound transition equations are marked accordingly
    and cannot produce PASS.

    The deprecated transition_equations_bound=False placeholder has been
    removed.  All non-trivial queries must be code-bound.
    """
    return generate_code_bound_queries()
