"""Phase K02/K03：纯 P0 timing state IR 与状态关系。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.bridge.model_bounds import P0ModelBounds
from formal_toolchain.adapters.formal_runtime_snapshot import ReleasedJobRecord, TerminalRecord, MissRecord


N6_JOB_RELATION_SUFFIXES = (
    "present",
    "key",
    "criticality",
    "release",
    "deadline",
    "demand",
    "service",
    "hi_miss",
)

PARAMETERIZED_RELATION_COMPONENTS = (
    "time_equal", "mode_equal", "released_job_map_total",
    "released_job_map_injective", "released_metadata_equal",
    "active_domain_mapped", "active_service_equal",
    "remaining_to_removal_equal", "ready_order_mapped",
    "running_key_mapped", "terminal_ledger_mapped", "miss_ledger_mapped",
    "miss_ledger_exactly_mapped", "future_budget_ghost_equal",
    "effective_event_frontier_isomorphic",
)


def parameterized_state_relation_schema_hash() -> str:
    return sha256_object({
        "schema_version": "parameterized_closed_prefix_relation_v1",
        "components": PARAMETERIZED_RELATION_COMPONENTS,
    })


@dataclass(frozen=True, slots=True)
class JobRelationRow:
    concrete_key: tuple[str, int]
    reference_key: tuple[str, int]
    release_time_match: bool
    deadline_match: bool
    remaining_match: bool
    priority_match: bool


@dataclass(frozen=True, slots=True)
class FiniteJobMap:
    rows: tuple[JobRelationRow, ...]

    def mapping_is_total_on_released(self, released_keys: set) -> bool:
        concrete_mapped = {row.concrete_key for row in self.rows}
        reference_mapped = {row.reference_key for row in self.rows}
        if concrete_mapped != released_keys:
            return False
        if len(concrete_mapped) != len(self.rows):
            return False
        if len(reference_mapped) != len(self.rows):
            return False
        return True

    def validate(self, concrete_released: set, reference_released: set) -> None:
        c_keys = [row.concrete_key for row in self.rows]
        r_keys = [row.reference_key for row in self.rows]
        if len(c_keys) != len(set(c_keys)):
            raise ValueError("CONCRETE_JOB_MAP_NOT_INJECTIVE")
        if len(r_keys) != len(set(r_keys)):
            raise ValueError("REFERENCE_JOB_MAP_NOT_INJECTIVE")
        if set(c_keys) != set(concrete_released):
            raise ValueError("CONCRETE_RELEASED_DOMAIN_NOT_TOTAL")
        if set(r_keys) != set(reference_released):
            raise ValueError("REFERENCE_RELEASED_DOMAIN_NOT_TOTAL")
        for row in self.rows:
            if not (row.release_time_match and row.deadline_match
                    and row.remaining_match and row.priority_match):
                raise ValueError(f"JOB_RELATION_ROW_MISMATCH:{row.concrete_key}")


@dataclass(frozen=True, slots=True)
class RelationResult:
    pass_: bool
    checks: dict[str, bool]

    def __bool__(self) -> bool:
        return self.pass_


def frontiers_isomorphic(concrete_frontier, reference_frontier) -> bool:
    c_keys = tuple(e.logical_key() if hasattr(e, "logical_key") else e for e in concrete_frontier)
    r_keys = tuple(e.logical_key() if hasattr(e, "logical_key") else e for e in reference_frontier)
    return c_keys == r_keys


def _mapped(job_map: Mapping, key):
    return job_map.get(key)


def normalize_terminal_kind(kind: str) -> str:
    if kind in {"PRIMARY_LO_BUDGET_CANCELLATION", "ACTIVE_LO_DROPPED_ON_MODE_SWITCH", "LO_RELEASE_DROPPED_IN_HI_MODE", "COMPLETED"}:
        return "LOGICAL_REMOVAL"
    raise ValueError(f"UNKNOWN_TERMINAL_KIND:{kind}")


def _released_records(state):
    records = tuple(getattr(state, "released_ledger", ()))
    if records:
        return {record.job_key: record for record in records}
    # Compatibility for old synthetic P0 states; authoritative runtime states
    # always carry the formal snapshot ledger.
    return {
        job.job_key: job for job in getattr(state, "active_jobs", ())
    }


def relation_holds(
    concrete: "P0ConcreteState",
    reference: "P0ReferenceState",
    job_map: Mapping[tuple[str, int], tuple[str, int]] | None = None,
) -> RelationResult:
    """Authoritative finite-map relation for arbitrary released-job domains.

    ``job_map`` is the proof witness.  The optional compatibility default is
    deliberately derived from equal keys and is not used by N5 builders.
    """
    c_released = _released_records(concrete)
    r_released = _released_records(reference)
    if job_map is None:
        job_map = {key: key for key in c_released}
    mapped_values = tuple(job_map.values())
    c_active = {job.job_key: job for job in getattr(concrete, "active_jobs", ())
                if getattr(job, "state", "active") not in {"finished", "dropped"}}
    r_active = {job.job_key: job for job in getattr(reference, "active_jobs", ())
                if getattr(job, "state", "active") not in {"finished", "dropped"}}
    r_active_ref = getattr(reference, "jobs", {})
    if not r_active and r_active_ref:
        r_active = dict(r_active_ref)

    def rec_value(record, name, fallback=None):
        aliases = {"absolute_deadline": "deadline", "release_class": "release_category"}
        return getattr(record, name, getattr(record, aliases.get(name, ""), fallback))

    released_metadata = True
    for ckey, crecord in c_released.items():
        rkey = job_map.get(ckey)
        rrecord = r_released.get(rkey)
        if rrecord is None:
            released_metadata = False
            break
        fields = ("release_time", "absolute_deadline", "criticality",
                  "priority_index", "release_class", "release_budget",
                  "removal_demand")
        released_metadata &= all(rec_value(crecord, f) == rec_value(rrecord, f) for f in fields)

    active_domain = set(job_map.get(k) for k in c_active) == set(r_active)
    active_service = all(
        job_map.get(k) in r_active and cjob.service == r_active[job_map[k]].service
        for k, cjob in c_active.items()
    )
    remaining = all(
        job_map.get(k) in r_active and remaining_remove(cjob) == remaining_remove(r_active[job_map[k]])
        for k, cjob in c_active.items()
    )
    c_terminal = {r.job_key: r for r in getattr(concrete, "terminal_ledger", ())}
    r_terminal = {r.job_key: r for r in getattr(reference, "terminal_ledger", ())}
    if not r_terminal:
        r_terminal = dict(getattr(reference, "terminal", {}))
    mapped_terminal_keys = {job_map.get(k) for k in c_terminal}
    terminal_ok = None not in mapped_terminal_keys and mapped_terminal_keys == set(r_terminal)
    if terminal_ok:
        terminal_ok = all(
            normalize_terminal_kind(c_terminal[ck].terminal_kind) == normalize_terminal_kind(r_terminal[rk].terminal_kind)
            and c_terminal[ck].terminal_time == r_terminal[rk].terminal_time
            and c_terminal[ck].executed_service == r_terminal[rk].executed_service
            for ck, rk in job_map.items() if ck in c_terminal and rk in r_terminal
        )
    c_misses = {(r.job_key, r.miss_time, r.absolute_deadline, r.executed_at_miss)
                for r in getattr(concrete, "miss_ledger", ())}
    c_records = tuple(getattr(concrete, "miss_ledger", ()))
    r_records = tuple(getattr(reference, "miss_ledger", ())) or tuple(getattr(reference, "misses", ()))
    mapped_concrete_misses = {(job_map.get(r.job_key), r.miss_time, r.absolute_deadline, r.executed_at_miss, r.criticality, r.release_time, r.priority_index) for r in c_records}
    reference_misses = {(r.job_key, r.miss_time, r.absolute_deadline, r.executed_at_miss, r.criticality, r.release_time, r.priority_index) for r in r_records}
    miss_ok = None not in {item[0] for item in mapped_concrete_misses} and mapped_concrete_misses == reference_misses
    checks = {
        "time_equal": concrete.time == reference.time,
        "mode_equal": concrete.mode == reference.mode,
        "released_job_map_total": set(job_map) == set(c_released) and set(mapped_values) == set(r_released),
        "released_job_map_injective": len(mapped_values) == len(set(mapped_values)) and len(job_map) == len(c_released),
        "released_metadata_equal": released_metadata,
        "active_domain_mapped": active_domain,
        "active_service_equal": active_service,
        "remaining_to_removal_equal": remaining,
        "ready_order_mapped": tuple(job_map.get(k) for k in getattr(concrete, "ready_jobs", ())) == tuple(getattr(reference, "ready_jobs", getattr(reference, "ready_order", ()))),
        "running_key_mapped": job_map.get(getattr(concrete, "running_job", None)) == getattr(reference, "running_job", getattr(reference, "running", None)),
        "terminal_ledger_mapped": terminal_ok,
        "miss_ledger_mapped": miss_ok,
        "miss_ledger_exactly_mapped": miss_ok,
        "future_budget_ghost_equal": concrete.global_future_budgets == getattr(reference, "global_future_budgets", concrete.global_future_budgets),
        "effective_event_frontier_isomorphic": frontiers_isomorphic(getattr(concrete, "effective_event_frontier", ()), getattr(reference, "effective_event_frontier", getattr(reference, "frontier", ()))),
    }
    return RelationResult(pass_=all(checks.values()), checks=checks)


def state_relation(concrete, reference, mapping: FiniteJobMap) -> RelationResult:
    c_time = getattr(concrete, "time", None)
    r_time = getattr(reference, "time", None)
    c_mode = getattr(concrete, "mode", None)
    r_mode = getattr(reference, "mode", None)

    c_released = {r.job_key for r in getattr(concrete, "released_ledger", ())}
    r_released = set(getattr(reference, "released", {}).keys()) if hasattr(reference, "released") else set()
    c_terminal_records = {r.job_key for r in getattr(concrete, "terminal_ledger", ())}
    r_terminal = set(getattr(reference, "terminal", {}).keys()) if hasattr(reference, "terminal") else set()
    c_misses = {(m.job_key, m.miss_time) for m in getattr(concrete, "miss_ledger", ())}
    r_misses = set(getattr(reference, "misses", ()))

    def map_key_if_exists(key, mapping_rows):
        for row in mapping_rows:
            if row.concrete_key == key:
                return row.reference_key
        return None

    checks = {
        "time": c_time == r_time,
        "mode": c_mode == r_mode,
        "frontier": frontiers_isomorphic(
            getattr(concrete, "effective_event_frontier", ()),
            getattr(reference, "frontier", ()),
        ),
        "job_map_bijection": mapping.mapping_is_total_on_released(c_released),
        "released_domain": c_released == r_released,
        "terminal_domain": c_terminal_records <= r_released,
        "miss_domain": c_misses <= {(m, 0) for m in c_released},
        "ready_order": tuple(getattr(concrete, "ready_jobs", ())) == tuple(getattr(reference, "ready_order", ())),
        "running": getattr(concrete, "running_job", None) == getattr(reference, "running", None) if False else (
            map_key_if_exists(getattr(concrete, "running_job", None), mapping.rows)
            == getattr(reference, "running", None)
        ),
    }
    return RelationResult(pass_=all(checks.values()), checks=checks)


@dataclass(frozen=True, slots=True)
class P0Job:
    """P0 job：只保留 timing-relevant 字段。"""
    job_key: tuple[str, int]
    priority_index: int
    release_time: int
    deadline: int
    release_category: str
    release_budget: int | None
    demand: int
    service: int = 0
    state: str = "active"
    mode: str = "LO"
    hi_completed: bool = False
    hi_deadline_miss: bool = False
    criticality: str = "LO"
    released_mode: str = "LO"
    is_degraded: bool = False
    # raw_actual_cost 保留 runtime 的原始执行需求；removal_demand 是按
    # release-fixed 规则计算、供 concrete/reference remaining 关系使用的需求。
    raw_actual_cost: int | None = None
    removal_demand: int | None = None

    @property
    def remaining(self) -> int:
        demand = self.demand if self.removal_demand is None else self.removal_demand
        return max(0, demand - self.service)


@dataclass(frozen=True, slots=True)
class P0ConcreteState:
    time: int
    mode: str
    active_jobs: tuple[P0Job, ...] = ()
    ready_jobs: tuple[tuple[str, int], ...] = ()
    running_job: tuple[str, int] | None = None
    global_future_budgets: tuple[tuple[str, int], ...] = ()
    miss_flags: tuple[tuple[str, int], ...] = ()
    queue_projection: tuple[tuple[Any, ...], ...] = ()
    next_controller_boundary: int | None = None
    next_timing_boundary: int | None = None
    released_ledger: tuple[ReleasedJobRecord, ...] = ()
    terminal_ledger: tuple[TerminalRecord, ...] = ()
    miss_ledger: tuple[MissRecord, ...] = ()
    effective_event_frontier: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class P0ReferenceState:
    time: int
    mode: str
    active_jobs: tuple[P0Job, ...] = ()
    ready_jobs: tuple[tuple[str, int], ...] = ()
    running_job: tuple[str, int] | None = None
    global_future_budgets: tuple[tuple[str, int], ...] = ()
    miss_flags: tuple[tuple[str, int], ...] = ()
    queue_projection: tuple[tuple[Any, ...], ...] = ()
    next_controller_boundary: int | None = None
    next_timing_boundary: int | None = None
    released_ledger: tuple[ReleasedJobRecord, ...] = ()
    terminal_ledger: tuple[TerminalRecord, ...] = ()
    miss_ledger: tuple[MissRecord, ...] = ()
    effective_event_frontier: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class P0Event:
    time: int
    kind: str
    job_key: tuple[str, int] | None = None
    payload: tuple[tuple[str, Any], ...] = ()


def p0_state_relation_schema() -> tuple[str, ...]:
    """状态关系实际比较的字段清单，供证明对象做版本绑定。"""
    return (
        "time", "mode", "active_jobs.job_key", "active_jobs.active", "active_jobs.ready",
        "active_jobs.running", "active_jobs.priority_index", "active_jobs.release_time",
        "active_jobs.deadline", "active_jobs.release_category", "active_jobs.release_budget",
        "active_jobs.demand", "active_jobs.service", "active_jobs.state", "active_jobs.mode",
        "active_jobs.hi_completed", "active_jobs.hi_deadline_miss", "ready_jobs",
        "running_job", "global_future_budgets", "miss_flags",
        "affected_job_key", "affected_job_active", "affected_job_ready",
        "affected_job_running", "affected_job_priority", "affected_job_release",
        "affected_job_deadline", "affected_job_category", "affected_job_budget",
        "affected_job_demand", "affected_job_service", "affected_job_hi_complete",
        "affected_job_hi_miss", "affected_task_budget", "frame.other_jobs",
        "frame.other_task_budgets", "queue_projection", "next_controller_boundary",
        "event_job_key", "running_job_key", "selected_job_key",
        "job_slots.pointwise", "task_budget_slots.pointwise", "queue_slots.pointwise",
        "queue_slots.minimum_future_time", "queue_slots.event_identity", "next_timing_boundary",
    )


def p0_smt_relation_fields(bounds: P0ModelBounds) -> tuple[str, ...]:
    """按 bounds 生成每条 transition query 实际声明的字段。"""
    scalar_fields = (
        "time", "service", "remaining", "budget", "miss", "mode", "active",
        "ready", "running", "priority", "release", "deadline", "category",
        "job_key", "hi_complete", "future_budget", "affected_job_key",
        "affected_job_active", "affected_job_ready", "affected_job_running",
        "affected_job_priority", "affected_job_release", "affected_job_deadline",
        "affected_job_category", "affected_job_budget", "affected_job_demand",
        "affected_job_service", "affected_job_hi_complete", "affected_job_hi_miss",
        "affected_task_budget",
        "next_controller_boundary", "event_job_key", "running_job_key",
        "selected_job_key",
        "ready_empty",
        # queue 只保留决定下一步 timing transition 的摘要，不展开 heap slot。
        "queue_min_time", "queue_min_kind", "queue_min_job_key", "queue_min_token",
        "queue_next_release_time", "queue_next_deadline_time", "queue_event_count",
        "queue_token_epoch",
    )
    job_fields = tuple(
        f"job_{slot}_{field}" for slot in range(bounds.job_slots)
        for field in ("present", "key", "active", "ready", "running", "priority",
                      "release", "deadline", "category", "criticality", "mode",
                      "released_mode", "is_degraded", "budget", "demand", "service",
                      "completion_token", "overrun_token", "hi_complete", "hi_miss")
    )
    task_fields = tuple(
        f"task_{slot}_{field}" for slot in range(bounds.task_slots)
        for field in ("present", "key", "criticality", "future_budget")
    )
    return scalar_fields + job_fields + task_fields


def n6_relation_projection_fields(
    bounds: P0ModelBounds,
) -> tuple[str, ...]:
    fields = ["time", "miss"]

    for slot in range(bounds.job_slots):
        fields.extend(
            f"job_{slot}_{suffix}"
            for suffix in N6_JOB_RELATION_SUFFIXES
        )

    return tuple(fields)


def p0_state_relation_schema_hash(bounds: P0ModelBounds) -> str:
    # This list is also consumed by case_templates when it builds the actual SMT
    # declarations.  Keeping the hash here prevents a descriptive Python-only
    # schema from being mistaken for the proved relation.
    return sha256_object({"schema": "p0_state_relation_v5_dynamic",
                          "model_bounds": bounds.to_dict(),
                           "smt_fields": p0_smt_relation_fields(bounds),
                           "python_relation_fields": p0_state_relation_schema()})


def bounded_smt_relation_fields(bounds: P0ModelBounds) -> tuple[str, ...]:
    return p0_smt_relation_fields(bounds)


def bounded_smt_relation_schema_hash(bounds: P0ModelBounds) -> str:
    return p0_state_relation_schema_hash(bounds)


def build_n6_relation_interface(*_bounded_diagnostic_args: Any) -> dict[str, Any]:
    return {
        "schema_version": "n6_closed_prefix_relation_interface_v2",
        "scope": "EVERY_REACHABLE_CLOSED_PREFIX",
        "map_domain": "ALL_RELEASED_JOB_KEYS_IN_PREFIX",
        "required_quantities": ["mapped_job_key", "criticality", "release_time", "absolute_deadline", "removal_demand", "service_at_deadline", "terminal_status", "miss_time", "miss_ledger_membership"],
        "parameterized_relation_schema_hash": parameterized_state_relation_schema_hash(),
    }


def validate_n6_relation_interface(interface: Mapping[str, Any]) -> None:
    if interface.get("schema_version") != "n6_closed_prefix_relation_interface_v2":
        raise ValueError("N6_RELATION_INTERFACE_SCHEMA_INVALID")
    if interface.get("scope") != "EVERY_REACHABLE_CLOSED_PREFIX":
        raise ValueError("N6_RELATION_INTERFACE_SCOPE_INVALID")
    if "job_slots" in interface or any(str(k).startswith("job_") for k in interface.get("required_fields", ())):
        raise ValueError("N6_RELATION_INTERFACE_LEGACY_SLOT_BASED")
    if interface.get("parameterized_relation_schema_hash") != parameterized_state_relation_schema_hash():
        raise ValueError("N6_RELATION_INTERFACE_RELATION_SCHEMA_INVALID")


def p0_state_from_runtime_engine(engine: Any) -> P0ConcreteState:
    """从真实 ``EventRuntimeEngine`` 当前状态构造 PreClosed P0 state。

    该函数只读 engine，不创建或补写任何 synthetic job；调用方必须先把
    engine 推进到明确的 time-0 closure 边界。
    """
    jobs = []
    for job in tuple(engine.state.active_jobs):
        key = (str(job.task.name), int(job.release_index))
        raw_actual_cost = int(getattr(job, "original_actual_cost", job.actual_cost))
        if getattr(job, "removal_demand", None) is not None:
            removal_demand = int(job.removal_demand)
        elif bool(job.is_degraded):
            removal_demand = raw_actual_cost
        elif str(getattr(job.task.criticality, "value", job.task.criticality)) == "LO" and job.runtime_budget_at_release is not None:
            removal_demand = min(raw_actual_cost, int(job.runtime_budget_at_release) + 1)
        else:
            removal_demand = raw_actual_cost
        jobs.append(P0Job(
            job_key=key, priority_index=int(engine.priority_map[job.task.name]),
            release_time=int(job.release_time), deadline=int(job.absolute_deadline),
            release_category=("degraded" if bool(job.is_degraded) else "normal"),
            release_budget=None if job.runtime_budget_at_release is None else int(job.runtime_budget_at_release),
            demand=raw_actual_cost, service=int(job.executed_time),
            state="dropped" if bool(job.dropped) else ("finished" if job.finished() else "active"),
            mode=engine.state.mode.name, hi_completed=bool(job.task.criticality.value == "HI" and job.finished()),
            hi_deadline_miss=any(m.task == job.task.name and m.release_index == job.release_index
                                 for m in engine.result.deadline_misses),
            criticality=str(getattr(job.task.criticality, "value", job.task.criticality)),
            released_mode=str(getattr(job.released_in_mode, "name", job.released_in_mode)),
            is_degraded=bool(job.is_degraded), raw_actual_cost=raw_actual_cost,
            removal_demand=removal_demand))
    active_keys = tuple(job.job_key for job in jobs if job.state not in {"dropped", "finished"})
    running = engine.state.running_job
    running_key = None if running is None else (str(running.task.name), int(running.release_index))
    queue_projection = []
    queue_snapshot = getattr(engine.queue, "snapshot", None)
    if callable(queue_snapshot):
        for item in queue_snapshot():
            queue_projection.append((int(item.time), str(item.event_type), item.task_name,
                                     item.release_index, item.token))
    else:
        for item in getattr(engine.queue, "_heap", ()):
            event = item[3]
            queue_projection.append((int(event.time), str(event.event_type.value), event.task_name,
                                     event.release_index, event.token))
    budgets = tuple(sorted((str(name), int(value)) for name, value in engine.runtime_budgets.budgets.items()))
    queue_projection = tuple(sorted(queue_projection))
    next_boundary = min((int(item[0]) for item in queue_projection
                         if int(item[0]) >= int(engine.current_time)), default=None)
    from formal_toolchain.adapters.formal_runtime_snapshot import build_formal_runtime_snapshot
    snapshot = build_formal_runtime_snapshot(engine, engine.priority_map)
    return P0ConcreteState(time=int(engine.current_time), mode=str(engine.state.mode.name),
                           active_jobs=tuple(jobs), ready_jobs=active_keys,
                           running_job=running_key, global_future_budgets=budgets,
                           miss_flags=tuple((str(m.task), int(m.release_index)) for m in engine.result.deadline_misses),
                           queue_projection=queue_projection,
                           next_controller_boundary=None,
                           next_timing_boundary=next_boundary,
                           released_ledger=snapshot.released_ledger,
                           terminal_ledger=snapshot.terminal_ledger,
                           miss_ledger=snapshot.miss_ledger,
                           effective_event_frontier=snapshot.effective_event_frontier)


def remaining_remove(job: P0Job) -> int:
    """按 release-fixed demand 计算关系中的 concrete remaining。"""
    demand = job.removal_demand if job.removal_demand is not None else job.demand
    return max(0, demand - job.service)
