from __future__ import annotations

from types import SimpleNamespace

from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.observation_metadata import build_action_definitions
from formal_toolchain.core.formal_checks import _build_formal_actions
from formal_toolchain.core.hashing import proof_safe_value, sha256_object
from formal_toolchain.semantics.frozen_c_amc_sem_action_runtime import canonical_action_schema
from formal_toolchain.verifier.checker_catalog import _actions


class _Adapter:
    def export_mask_contract(self):
        return {"explicit_noop": True, "explicit_noop_action_ids": [24],
                "explicit_noop_always_valid": True}


def _target():
    tasks = tuple(
        Task(name=f"T{index:02d}", period=100, deadline=100, c_lo=10, c_hi=20,
             criticality=Criticality.HI if index < 6 else Criticality.LO)
        for index in range(12)
    )
    actions = build_budget_action_space(
        tasks, action_space="single", budget_increase_ratio=0.02,
        budget_decrease_ratio=0.02, include_explicit_noop=True,
    )
    return SimpleNamespace(
        ordered_tasks=tasks,
        runtime_config=SimpleNamespace(action_space="single", budget_increase_ratio=0.02,
                                       budget_decrease_ratio=0.02),
        runtime_adapter=_Adapter(),
        action_definitions=tuple(build_action_definitions(actions)),
    )


def test_compiler_and_fresh_verifier_rebuild_identical_single25_schema() -> None:
    target = _target()
    compiler = _build_formal_actions(target)
    verifier = _actions({"target": target})
    compiler_rows = [canonical_action_schema(row) for row in compiler]
    verifier_rows = [canonical_action_schema(row) for row in verifier]
    assert compiler_rows == verifier_rows
    assert sha256_object(proof_safe_value(compiler_rows)) == sha256_object(
        proof_safe_value(verifier_rows)
    )
    assert compiler_rows[24]["is_noop"] is True
