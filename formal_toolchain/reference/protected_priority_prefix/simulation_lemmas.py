"""Named receipts for the protected-prefix macro-step proof."""

from .macro_step import (
    prove_arrival_batch_projection, prove_completion_removal_correspondence,
    prove_deadline_batch_correspondence, prove_final_dispatch_correspondence,
    prove_mode_tail_phase_join, prove_protected_macro_step_preservation,
    prove_protected_service_correspondence, prove_tail_service_exclusion,
)

__all__ = [name for name in globals() if name.startswith("prove_")]
