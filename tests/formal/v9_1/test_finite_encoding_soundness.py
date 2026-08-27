from formal_toolchain.v9_1.safe_prefix_invariant import SafePrefixInvariant
from formal_toolchain.v9_1.soundness_checker import build_finite_window_soundness_certificate
from formal_toolchain.v9_1.symbolic_state import BoundModel, TaskBound


def test_soundness_certificate_does_not_promote_missing_tree_to_true():
    model = BoundModel((TaskBound("hi", 0, 3, 3, "HI", 1, 2, 1, 1, 2),), 2)
    certificate = build_finite_window_soundness_certificate(model, SafePrefixInvariant(model))
    assert set(certificate["clauses"]) == set(certificate["required_clauses"])
    assert certificate["clauses"]["numeric_observation_tree_mask_firstvalid_noop_budget_update_exact"] is False
    assert certificate["all_pass"] is False
    assert certificate["formula_hash"] and certificate["fresh_recompute_hash"]
