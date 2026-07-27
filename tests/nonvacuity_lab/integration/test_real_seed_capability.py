from __future__ import annotations

from pathlib import Path

import pytest

from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact


@pytest.mark.real_seed
@pytest.mark.slow
def test_s185_real_tree_artifact_is_available_for_lab():
    root = Path(__file__).resolve().parents[3]
    seed = root / "s185" / "best_overall"
    if not seed.is_dir():
        pytest.skip("s185 real seed artifact unavailable")
    inventory = inspect_tree_artifact(seed, expected_seed=185)
    assert inventory["state_dim"] == 128
    assert inventory["action_dim"] == 24
