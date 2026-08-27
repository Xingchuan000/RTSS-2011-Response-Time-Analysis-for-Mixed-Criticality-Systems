"""Exact V9.1 ranked FirstValid selection with a total explicit noop."""

from __future__ import annotations

from collections.abc import Sequence


def first_valid_explicit_noop(
    ranking: Sequence[int],
    valid_mask: Sequence[bool],
    *,
    action_dim: int,
    noop_id: int,
) -> int:
    if action_dim <= 0 or len(ranking) != action_dim or len(valid_mask) != action_dim:
        raise ValueError("V9.1 ranking/mask dimension mismatch")
    normalized = tuple(int(value) for value in ranking)
    if tuple(sorted(normalized)) != tuple(range(action_dim)):
        raise ValueError("V9.1 ranking must be a complete action permutation")
    if not (0 <= noop_id < action_dim):
        raise ValueError("V9.1 explicit noop id is outside the action alphabet")
    if not bool(valid_mask[noop_id]):
        raise ValueError("V9.1 explicit noop must be valid in every controller state")
    for action_id in normalized:
        if bool(valid_mask[action_id]):
            return action_id
    raise AssertionError("unreachable: explicit noop makes FirstValid total")
