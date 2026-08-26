from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from formal_toolchain.binding.controller_binding import _analyze_env_selected_action
from formal_toolchain.binding.controller_update_binding import bind_controller_budget_update
from formal_toolchain.semantics.frozen_runtime_contract import (
    frozen_budget_runtime_path,
    frozen_event_runtime_path,
)
from formal_toolchain.verifier.recompute import recompute_controller_transition_certificate
from formal_toolchain.core.hashing import sha256_object


ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = ROOT / "amc_py/rl/env.py"


def _normal_branch_mutation(insert: str) -> str:
    source = ENV_PATH.read_text(encoding="utf-8")
    anchor = "                if accepted:\n                    self._engine.apply_budget_updates(updates, source=self.budget_update_source)"
    index = source.rfind(anchor)
    assert index >= 0
    return source[:index] + insert + source[index:]


@pytest.mark.parametrize("insert", [
    "                self._engine.run_until(0)\n",
    "                self._engine.state.mode = self._engine.state.mode\n",
    "                self._engine.state.current_time = self._engine.current_time\n",
    "                self._engine.state.running_job = self._engine.state.running_job\n",
    "                job.runtime_budget_at_release = job.runtime_budget_at_release\n",
    "                job.executed_time += 1\n",
])
def test_selected_action_state_and_plant_mutations_fail_closed(insert: str) -> None:
    result = _analyze_env_selected_action(_normal_branch_mutation(insert))
    assert result["status"] == "FAIL"


def test_selected_action_requires_candidate_evaluation_before_commit() -> None:
    source = ENV_PATH.read_text(encoding="utf-8")
    start = source.rfind("                evaluation = self.evaluate_budget_candidate(")
    end = source.index("                accepted = evaluation.accepted", start)
    source = source[:start] + "                evaluation = None\n" + source[end:]
    assert _analyze_env_selected_action(source)["status"] == "FAIL"


def test_selected_action_rejected_branch_cannot_commit() -> None:
    source = _normal_branch_mutation("")
    anchor = "                if accepted:\n                    self._engine.apply_budget_updates(updates, source=self.budget_update_source)"
    index = source.rfind(anchor)
    source = source[:index] + source[index:].replace("if accepted:", "if True:", 1)
    assert _analyze_env_selected_action(source)["status"] == "FAIL"


def test_budget_updates_have_no_partial_mutation() -> None:
    task = Task("known", 10, 10, 2, 2, Criticality.LO)
    state = BudgetState.from_tasks([task])
    before = dict(state.budgets)
    with pytest.raises(KeyError):
        state.apply_updates({"known": 3, "missing": 1})
    assert state.budgets == before


def _binding_root(tmp_path: Path) -> Path:
    for relative in (
        "formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py",
        "formal_toolchain/semantics/frozen_c_amc_sem_runtime_wrapper.py",
        "formal_toolchain/semantics/frozen_c_amc_sem_action_runtime.py",
        "formal_toolchain/semantics/frozen_c_amc_sem_observation.py",
        "formal_toolchain/semantics/frozen_c_amc_sem_event_models.py",
        "formal_toolchain/semantics/frozen_c_amc_sem_budget_runtime.py",
        "formal_toolchain/semantics/frozen_runtime_contract.py",
        "amc_py/rl/env.py",
        "amc_py/viper/tree_policy.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


@pytest.mark.parametrize(("needle", "replacement"), [
    ("prio < best_priority", "prio > best_priority"),
    ("self.priority_map = {task.name: idx for idx, task in enumerate(self.ordered_tasks)}", "self.priority_map = {task.name: -idx for idx, task in enumerate(self.ordered_tasks)}"),
    ("time=now + job.remaining()", "time=now + job.remaining() + 1"),
    ("event_type=EventType.JOB_COMPLETION", "event_type=EventType.BUDGET_OVERRUN"),
])
def test_force_reschedule_and_frontier_mutations_fail_closed(
    tmp_path: Path, needle: str, replacement: str,
) -> None:
    root = _binding_root(tmp_path)
    path = root / "formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py"
    source = path.read_text(encoding="utf-8")
    assert needle in source
    path.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
    assert bind_controller_budget_update(root)["status"] == "FAIL"


def test_fresh_recompute_rejects_mutated_deployed_step(tmp_path: Path) -> None:
    root = _binding_root(tmp_path)
    path = root / "amc_py/rl/env.py"
    source = path.read_text(encoding="utf-8")
    anchor = "                if accepted:\n                    self._engine.apply_budget_updates(updates, source=self.budget_update_source)"
    index = source.rfind(anchor)
    assert index >= 0
    source = source[:index] + source[index:].replace(
        "if accepted:", "if True:", 1,
    )
    path.write_text(source, encoding="utf-8")
    certificate = recompute_controller_transition_certificate(
        source_root=root,
        verified_action_binding={
            "status": "PASS", "action_dim": 25, "explicit_noop": True,
            "action_space_type": "single",
        },
        verified_policy_binding={
            "status": "PASS", "artifact_hash": sha256_object({"policy": "fresh"}),
        },
        context_hash="0" * 64,
    )
    assert certificate["obligation_status"] != "PASS"


def test_extra_c_amc_queue_write_fails_controller_update_binding(tmp_path: Path) -> None:
    root = _binding_root(tmp_path)
    path = root / "formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py"
    source = path.read_text(encoding="utf-8")
    anchor = "    completion_token = state.next_token()\n"
    assert anchor in source
    injected = (
        "    queue.push(Event(time=now + 99, event_type=EventType.JOB_COMPLETION, "
        "task_name=job.task.name, release_index=job.release_index, token=999999))\n"
    )
    path.write_text(source.replace(anchor, injected + anchor, 1), encoding="utf-8")
    assert bind_controller_budget_update(root)["status"] == "FAIL"


def test_core_reschedule_active_set_mutation_fails_binding(tmp_path: Path) -> None:
    root = _binding_root(tmp_path)
    path = root / "formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py"
    source = path.read_text(encoding="utf-8")
    anchor = "    state.running_job = selected\n"
    assert anchor in source
    path.write_text(source.replace(anchor, "    state.active_jobs.clear()\n" + anchor, 1), encoding="utf-8")
    assert bind_controller_budget_update(root)["status"] == "FAIL"


def test_token_invalidation_mutation_fails_binding(tmp_path: Path) -> None:
    root = _binding_root(tmp_path)
    path = root / "formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py"
    source = path.read_text(encoding="utf-8")
    needle = "    state.valid_completion_tokens.pop(key, None)\n"
    assert needle in source
    path.write_text(source.replace(needle, "", 1), encoding="utf-8")
    assert bind_controller_budget_update(root)["status"] == "FAIL"
