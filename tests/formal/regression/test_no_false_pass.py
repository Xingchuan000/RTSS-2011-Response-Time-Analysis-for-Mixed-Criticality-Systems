from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from amc_py.models import Criticality
from formal_toolchain.adapters.batch_frozen_scenario import BatchFrozenExecutionScenario
from formal_toolchain.adapters.formal_scenario_factory import build_formal_scenario
from formal_toolchain.bridge.effective_event_frontier import effective_frontier
from formal_toolchain.adapters.formal_runtime_snapshot import build_formal_runtime_snapshot
from formal_toolchain.bridge.phase_k_runtime_states import build_preclosed_runtime_states
from formal_toolchain.bridge.state_relation import P0ConcreteState, P0Job
from formal_toolchain.verifier.structural_checks import verify_independent_bundle


@dataclass(frozen=True)
class _Task:
    name: str
    period: int
    deadline: int
    c_lo: int
    c_hi: int
    criticality: object
    offset: int = 0


@dataclass(frozen=True)
class _Mode:
    name: str


@dataclass
class _Job:
    task: _Task
    release_index: int = 0
    release_time: int = 0
    absolute_deadline: int = 10
    actual_cost: int = 7
    original_actual_cost: int = 11
    removed_demand: int = 3
    runtime_budget_at_release: int | None = 2
    executed_time: int = 4
    dropped: bool = False
    is_degraded: bool = True
    released_in_mode: _Mode = _Mode("LO")

    def finished(self) -> bool:
        return False


@dataclass
class _Engine:
    jobs_by_key: dict[tuple[str, int], _Job]
    state: object
    result: object
    queue: object
    runtime_budgets: object
    current_time: int = 0
    priority_map: dict[str, int] | None = None


class _SnapshotQueue:
    def __init__(self, events):
        self._events = tuple(events)

    def snapshot(self):
        return self._events


class _RuntimeSnapshot:
    def __init__(self, *, active_job_keys=()):
        self.completion_tokens = lambda key: 1 if key == ("job", 0) else None
        self.overrun_tokens = lambda key: None
        self.response_tokens = lambda key: None
        self.active_job_keys = tuple(active_job_keys)


@dataclass(frozen=True)
class _FrontierEvent:
    time: int
    event_type: str
    task_name: str | None
    release_index: int | None
    token: int | None = None


def test_batch_frozen_execution_scenario_exports_actual_cost_for():
    assert hasattr(BatchFrozenExecutionScenario, "actual_cost_for")

    class _Scenario:
        def actual_cost_for(self, task, release_index):
            return task.c_lo + release_index

    task = _Task("tau", 10, 10, 2, 4, Criticality.LO)
    adapter = build_formal_scenario(base_scenario=_Scenario(), ordered_tasks=[task])
    assert adapter.actual_cost_for(task, 0) == 2


def test_preclosed_builder_uses_frozen_semantics_not_runtime_engine(monkeypatch):
    reads = []

    class _RuntimeConfig:
        def __init__(self):
            self.semantics = "C_AMC_SEM"
            self.c_amc_sem_lo_degradation_ratio = 0.5
            self.c_amc_sem_primary_on_switch_time = True
            self.end_time = 17

    class _Target:
        def __init__(self):
            self.runtime_config = _RuntimeConfig()
            self.ordered_tasks = [_Task("tau", 10, 10, 2, 4, Criticality.LO)]
            self.scenario = type(
                "_Scenario",
                (),
                {"actual_cost_for": lambda self, task, release_index: (
                    reads.append((task.name, release_index)) or task.c_lo + release_index
                )},
            )()

    # A mutable runtime construction would be a regression: Phase K must use
    # the frozen formal adapter even when the implementation runtime changes.
    def forbidden_build(*_args, **_kwargs):
        raise AssertionError("mutable EventRuntimeEngine must not be executed")

    monkeypatch.setattr("amc_py.event_runtime.EventRuntimeEngine.build", forbidden_build)

    target = _Target()
    reference_taskset = {
        "tasks": [{
            "name": "tau",
            "priority_index": 0,
            "initial_runtime_budget": 2,
            "degraded_cost": 1,
        }]
    }
    concrete, reference = build_preclosed_runtime_states(target, reference_taskset)

    assert reads == [("tau", 0)]
    assert concrete.time == reference.time == 0
    assert concrete.running_job == reference.running_job == ("tau", 0)
    assert concrete.active_jobs[0].removal_demand == 2


def test_formal_runtime_snapshot_uses_original_actual_cost_and_removal_demand():
    task = _Task("tau", 10, 10, 2, 4, Criticality.LO)
    job = _Job(task=task)
    engine = _Engine(
        jobs_by_key={("tau", 0): job},
        state=type("State", (), {"active_jobs": (job,), "mode": _Mode("LO"), "running_job": None})(),
        result=type("Result", (), {"deadline_misses": ()})(),
        queue=_SnapshotQueue(()),
        runtime_budgets=type("Budgets", (), {"budgets": {"tau": 2}})(),
        priority_map={"tau": 0},
    )

    snapshot = build_formal_runtime_snapshot(engine)
    record = snapshot.released_ledger[0]

    assert record.raw_actual_cost == job.original_actual_cost
    assert record.release_class is not None
    assert record.removal_demand >= 0


def test_effective_frontier_uses_logical_events():
    events = (
        _FrontierEvent(0, "JOB_ARRIVAL", "job", 0),
        _FrontierEvent(0, "DEADLINE_CHECK", "job", 0),
        _FrontierEvent(0, "RECOVERY", None, None),
    )
    runtime_snapshot = _RuntimeSnapshot(active_job_keys=(("job", 0),))

    frontier = effective_frontier(events, runtime_snapshot)
    from formal_toolchain.bridge.logical_events import LogicalEventKind
    kinds = [event.kind.value for event in frontier]
    assert kinds == ["REC", "DDL", "ARR_BATCH"]


def test_common_certificate_schema_is_structural_not_self_ref():
    schema_path = Path(__file__).resolve().parents[3] / "formal_toolchain" / "specs" / "certificates" / "common.schema.json"
    schema = __import__("json").loads(schema_path.read_text(encoding="utf-8"))
    assert "$ref" not in schema
    assert "required" in schema
    assert "artifact_schema_version" in schema.get("properties", {})


def test_kernel_assurance_requires_executable_backend_not_text_hash():
    policy_path = Path(__file__).resolve().parents[3] / "formal_toolchain" / "theory" / "assurance_policy.json"
    policy = __import__("json").loads(policy_path.read_text(encoding="utf-8"))
    assert "MACHINE_FORMALIZED_KERNEL" not in policy.get("allowed_levels", [])


def test_checker_exception_is_structured_failure(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AttributeError("boom")

    monkeypatch.setattr(
        "formal_toolchain.verifier.structural_checks.verify_obligation_certificate",
        boom,
    )
    result = verify_independent_bundle(
        certificates={"A": {"obligation_id": "A", "artifact_hash": "a" * 64}},
        registry=[{"id": "A", "depends_on": []}],
    )
    assert result.status == "FAIL"
    assert result.route == "PROOF_BUNDLE_INVALID"
