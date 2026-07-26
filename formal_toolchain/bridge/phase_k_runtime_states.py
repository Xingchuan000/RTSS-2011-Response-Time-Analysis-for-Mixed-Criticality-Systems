"""Phase-K PreClosed(0) states from the frozen C-AMC-sem contract.

The proof no longer executes the mutable experiment runtime.  q-AMC and other
runtime extensions are outside this semantic adapter and cannot perturb the
existing C-AMC-sem/P0 proof route.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.semantics.frozen_preclosed_state import (
    build_frozen_preclosed_bundle,
)


def build_preclosed_runtime_states(
    target: Any,
    reference_taskset: Mapping[str, Any],
):
    concrete, reference, _snapshot = build_frozen_preclosed_bundle(
        target,
        reference_taskset,
    )
    return concrete, reference


__all__ = ["build_preclosed_runtime_states"]
