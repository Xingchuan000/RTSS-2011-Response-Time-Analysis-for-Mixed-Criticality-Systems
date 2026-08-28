import pytest

from formal_toolchain.v9_2.policy import first_valid_explicit_noop


def test_first_valid_uses_rank_order_and_no_fallback():
    assert first_valid_explicit_noop((2, 0, 1), (False, True, True), action_dim=3, noop_id=1) == 2
    assert first_valid_explicit_noop((0, 2, 1), (False, True, False), action_dim=3, noop_id=1) == 1


def test_first_valid_rejects_non_total_noop_mask():
    with pytest.raises(ValueError, match="explicit noop"):
        first_valid_explicit_noop((0, 1), (True, False), action_dim=2, noop_id=1)
