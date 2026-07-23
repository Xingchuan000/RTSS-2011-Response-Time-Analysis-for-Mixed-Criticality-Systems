from formal_toolchain.bridge.state_relation import P0ConcreteState, P0Job, P0ReferenceState, relation_holds
from formal_toolchain.adapters.formal_runtime_snapshot import ReleasedJobRecord


def test_relation_supports_arbitrary_finite_map_and_ignores_raw_queue():
    concrete_keys = [("task", i) for i in range(20)]
    reference_keys = [("reference", i) for i in range(20)]
    record = lambda key: ReleasedJobRecord(key, 0, 10, "LO", "LO", "LO_PRIMARY_NORMAL", 3, 4, 4, 1, "test")
    jobs = lambda keys: tuple(P0Job(k, 1, 0, 10, "normal", 3, 4, 0, raw_actual_cost=4, removal_demand=4) for k in keys)
    concrete = P0ConcreteState(0, "LO", jobs(concrete_keys), concrete_keys, concrete_keys[0], released_ledger=tuple(record(k) for k in concrete_keys), effective_event_frontier=(("effective",),), queue_projection=(("stale",),))
    reference = P0ReferenceState(0, "LO", jobs(reference_keys), reference_keys, reference_keys[0], released_ledger=tuple(record(k) for k in reference_keys), effective_event_frontier=(("effective",),), queue_projection=())
    result = relation_holds(concrete, reference, dict(zip(concrete_keys, reference_keys)))
    assert result.pass_
    assert result.checks["effective_event_frontier_isomorphic"]
