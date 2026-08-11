"""Independent diagnostics tests for the stratified-dynamic workload family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.diagnose_mc_stratified_dynamic_structure import (
    MANIFEST_SCHEMA_VERSION,
    lag1_autocorrelation,
    normalized_hamming_distance,
    one_step_competition_probe,
    validate_manifest_rows,
)
from amc_py.models import Criticality, Task


def test_old_manifest_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        validate_manifest_rows([{"schema_version": "mc_fairgen_manifest_v1", "candidate_seed": "1"}])


def test_new_manifest_schema_is_accepted() -> None:
    validate_manifest_rows([{"schema_version": MANIFEST_SCHEMA_VERSION, "candidate_seed": "1"}])


def test_mask_turnover_is_normalized_hamming_distance() -> None:
    assert normalized_hamming_distance((True, False, True, False), (True, True, False, False)) == pytest.approx(0.5)
    assert normalized_hamming_distance((), ()) == 0.0


def test_lag1_autocorrelation_is_deterministic() -> None:
    assert lag1_autocorrelation((1.0, 2.0, 3.0, 4.0)) == pytest.approx(1.0)
    assert lag1_autocorrelation((1.0, 1.0, 1.0)) == 0.0


@dataclass(frozen=True)
class _Action:
    increase_task: str | None
    decrease_tasks: tuple[str, ...] = ()


def test_competition_probe_does_not_mutate_live_budget() -> None:
    tasks = {"lo": Task("lo", 10, 10, 2, 2, Criticality.LO)}
    actions = (_Action("lo"), _Action(None, ("lo",)))
    budgets = {"lo": 2}

    def mask_for_budget(candidate: dict[str, int]) -> tuple[bool, bool]:
        return (candidate["lo"] < 3, True)

    values = one_step_competition_probe(
        mask_before=(True, True),
        actions=actions,
        budget_before=budgets,
        mask_for_budget=mask_for_budget,
        tasks_by_name=tasks,
    )
    assert values == [pytest.approx(1.0)]
    assert budgets == {"lo": 2}


def test_new_scripts_do_not_import_legacy_workload_or_cli() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "scripts/diagnose_mc_stratified_dynamic_structure.py",
        "scripts/select_mc_stratified_dynamic_primary10.py",
    ):
        source = (root / name).read_text(encoding="utf-8")
        assert "from amc_py.workloads.mc_fairgen" not in source
        assert "common_mc_fairgen_cli" not in source

