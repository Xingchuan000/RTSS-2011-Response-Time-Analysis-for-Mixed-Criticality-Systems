"""Independent diagnostics tests for the stratified-dynamic workload family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.diagnose_mc_stratified_dynamic_structure import (
    MANIFEST_SCHEMA_VERSION,
    lag1_autocorrelation,
    normalized_hamming_distance,
    static_characterization,
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



def test_multistep_probe_detects_delayed_competition() -> None:
    from scripts.diagnose_mc_stratified_dynamic_structure import multistep_competition_frontier_probe

    tasks = {
        "a": Task("a", 100, 100, 10, 10, Criticality.LO),
        "b": Task("b", 100, 100, 10, 10, Criticality.LO),
    }
    actions = (_Action("a"), _Action("b"))
    budgets = {"a": 10, "b": 10}

    # Both actions are initially legal. Increasing a twice consumes the shared
    # synthetic frontier and removes b; one 2% step alone does not.
    def mask_for_budget(candidate: dict[str, int]) -> tuple[bool, bool]:
        total = candidate["a"] + candidate["b"]
        return (candidate["a"] < 20, total < 23)

    result = multistep_competition_frontier_probe(
        mask_before=(True, True),
        actions=actions,
        budget_before=budgets,
        mask_for_budget=mask_for_budget,
        tasks_by_name=tasks,
        max_depth=12,
    )
    assert max(result["competition_scores"]) > 0.0
    assert result["first_other_loss_steps"]
    assert budgets == {"a": 10, "b": 10}


def test_round_robin_probe_produces_frontier_mask_turnover() -> None:
    from scripts.diagnose_mc_stratified_dynamic_structure import (
        deterministic_round_robin_frontier_probe,
        normalized_hamming_distance,
    )

    tasks = {
        "a": Task("a", 100, 100, 10, 10, Criticality.LO),
        "b": Task("b", 100, 100, 10, 10, Criticality.LO),
    }
    actions = (_Action("a"), _Action("b"))
    budgets = {"a": 10, "b": 10}

    def mask_for_budget(candidate: dict[str, int]) -> tuple[bool, bool]:
        return (candidate["a"] < 13, candidate["b"] < 14)

    result = deterministic_round_robin_frontier_probe(
        mask_before=(True, True),
        actions=actions,
        budget_before=budgets,
        mask_for_budget=mask_for_budget,
        tasks_by_name=tasks,
        max_steps=20,
    )
    masks = result["mask_sequence"]
    turnover = [normalized_hamming_distance(a, b) for a, b in zip(masks, masks[1:])]
    assert result["unique_mask_count"] > 1
    assert max(turnover) > 0.0
    assert budgets == {"a": 10, "b": 10}


def test_static_characterization_uses_requested_c_amc_sem_opa_order() -> None:
    from amc_py.workloads.mc_stratified_dynamic import (
        MCStratifiedDynamicWorkloadConfig,
        MCStratifiedDynamicWorkloadProvider,
    )

    config = MCStratifiedDynamicWorkloadConfig(
        seed=4,
        require_schedulable=True,
        sched_method="c_amc_sem",
        priority_policy="opa",
        c_amc_sem_xf=0.5,
    )
    bundle = MCStratifiedDynamicWorkloadProvider(config).build(4)
    static = static_characterization(
        bundle,
        analysis_method="c_amc_sem",
        priority_policy="opa",
        c_amc_sem_xf=0.5,
    )
    assert static["analysis_method"] == "c_amc_sem"
    assert static["analysis_priority_policy"] == "opa"
    assert static["analysis_schedulable"] is True
    assert static["analysis_normalized_slack"] >= 0.0
    assert tuple(task.name for task in static["_ordered_tasks"])
