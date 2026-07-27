from __future__ import annotations

from pathlib import Path

import pytest
import torch

from amc_py.dqn import (
    DqnBudgetAgent,
    DqnConfig,
    validate_observation_binding,
)
from amc_py.rl.observation_schema import observation_schema_fingerprint


MODE = "v14_qamc_full_12d"
NAMES = ("T00.L.budget_norm", "T00.L.qamc_target_demand_ratio")


def _agent(
    *,
    mode: str | None = MODE,
    names: tuple[str, ...] | None = NAMES,
    fingerprint: str | None = None,
) -> DqnBudgetAgent:
    resolved_fingerprint = fingerprint
    if resolved_fingerprint is None and mode is not None and names is not None:
        resolved_fingerprint = observation_schema_fingerprint(
            observation_mode=mode,
            feature_names=names,
        )
    return DqnBudgetAgent(
        observation_dim=2,
        action_dim=1,
        config=DqnConfig(hidden_layers=(4,)),
        observation_mode=mode,
        observation_feature_names=names,
        observation_schema_fingerprint=resolved_fingerprint,
    )


def test_checkpoint_round_trip_preserves_observation_schema(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    _agent().save(path)
    loaded = DqnBudgetAgent.load(path, device="cpu")
    validate_observation_binding(
        loaded,
        expected_mode=MODE,
        expected_feature_names=NAMES,
        require_schema_binding=True,
    )


def test_same_dimension_wrong_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="MODEL_OBSERVATION_MODE_MISMATCH"):
        validate_observation_binding(
            _agent(mode="v14_qamc_quality_11d"),
            expected_mode=MODE,
            expected_feature_names=NAMES,
            require_schema_binding=True,
        )


def test_swapped_feature_order_is_rejected() -> None:
    swapped = tuple(reversed(NAMES))
    with pytest.raises(
        ValueError,
        match="MODEL_OBSERVATION_FEATURE_ORDER_MISMATCH",
    ):
        validate_observation_binding(
            _agent(names=swapped),
            expected_mode=MODE,
            expected_feature_names=NAMES,
            require_schema_binding=True,
        )


def test_wrong_schema_fingerprint_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="MODEL_OBSERVATION_SCHEMA_FINGERPRINT_MISMATCH",
    ):
        validate_observation_binding(
            _agent(fingerprint="wrong"),
            expected_mode=MODE,
            expected_feature_names=NAMES,
            require_schema_binding=True,
        )


def test_legacy_checkpoint_loads_but_cannot_be_used_as_o2(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.pt"
    _agent().save(path)
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint.pop("observation_mode")
    checkpoint.pop("observation_feature_names")
    checkpoint.pop("observation_schema_fingerprint")
    torch.save(checkpoint, path)

    loaded = DqnBudgetAgent.load(path, device="cpu")
    assert loaded.observation_mode is None
    validate_observation_binding(
        loaded,
        expected_mode="v10_basic",
        expected_feature_names=NAMES,
        require_schema_binding=False,
    )
    with pytest.raises(ValueError, match="MODEL_OBSERVATION_MODE_MISMATCH"):
        validate_observation_binding(
            loaded,
            expected_mode=MODE,
            expected_feature_names=NAMES,
            require_schema_binding=True,
        )
