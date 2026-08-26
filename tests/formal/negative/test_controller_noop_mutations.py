from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from formal_toolchain.binding.controller_binding import bind_controller_runtime


ROOT = Path(__file__).parents[3]
WRAPPER = ROOT / "formal_toolchain/semantics/frozen_c_amc_sem_runtime_wrapper.py"
EVENT_RUNTIME = ROOT / "formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py"
ENV = ROOT / "amc_py/rl/env.py"


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for source in (WRAPPER, EVENT_RUNTIME, ENV):
        destination = root / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return root


def _binding_with_mutation(tmp_path: Path, *, wrapper_replace: tuple[str, str] | None = None,
                           env_replace: tuple[str, str] | None = None) -> dict:
    root = _source_root(tmp_path)
    if wrapper_replace is not None:
        path = root / WRAPPER.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        old, new = wrapper_replace
        assert old in text
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    if env_replace is not None:
        path = root / ENV.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        old, new = env_replace
        assert old in text
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return bind_controller_runtime(root)


@pytest.mark.parametrize(
    "replacement",
    [
        (
            "\"updates\": {},",
            "\"updates\": {\"MUTATED\": 1},",
        ),
        (
            "budget_snapshot = dict(engine.runtime_budgets.budgets)",
            "budget_snapshot = dict(engine.runtime_budgets.budgets)\n            engine.run_until(current_tick)",
        ),
        (
            '"budget_after": budget_snapshot,',
            '"budget_after": dict(budget_snapshot, MUTATED=1),',
        ),
    ],
)
def test_noop_wrapper_mutations_fail_closed(tmp_path: Path, replacement: tuple[str, str]) -> None:
    assert _binding_with_mutation(tmp_path, wrapper_replace=replacement)["status"] != "PASS"


@pytest.mark.parametrize(
    "replacement",
    [
        (
            "updates = {}\n                candidate_budgets = dict(budget_before)",
            "updates = {'MUTATED': 1}\n                candidate_budgets = dict(budget_before)",
        ),
        (
            "updates = {}\n                candidate_budgets = dict(budget_before)",
            "updates = {}\n                self._engine.state.mode = 'HI'\n                candidate_budgets = dict(budget_before)",
        ),
        (
            "updates = {}\n                candidate_budgets = dict(budget_before)",
            "updates = {}\n                self._engine.state.current_time += 1\n                candidate_budgets = dict(budget_before)",
        ),
    ],
)
def test_mutable_environment_noop_writes_fail_closed(tmp_path: Path, replacement: tuple[str, str]) -> None:
    assert _binding_with_mutation(tmp_path, env_replace=replacement)["status"] != "PASS"


def test_selected_action_queued_event_rebinding_fails_closed(tmp_path: Path) -> None:
    result = _binding_with_mutation(
        tmp_path,
        wrapper_replace=(
            "engine.apply_budget_updates(updates)",
            "engine._process_event(BUDGET_UPDATE)",
        ),
    )
    assert result["status"] != "PASS"


def test_noop_fallback_plant_progression_fails_closed(tmp_path: Path) -> None:
    result = _binding_with_mutation(
        tmp_path,
        env_replace=(
            "            if action.is_noop:\n",
            "            if action.is_noop:\n                self._engine.run_until(1)\n",
        ),
    )
    assert result["status"] != "PASS"
