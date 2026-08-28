from formal_toolchain.v9_2.conformance import check_projection_conformance
from formal_toolchain.v9_2.timestamp_trace import TimestampSemanticRecord


def _record(**overrides):
    values = dict(
        timestamp=0,
        completion_batch=(),
        deadline_batch=(),
        release_batch=(),
        controller_enabled=False,
        controller_action=None,
        budget_before_controller=(),
        budget_after_controller=None,
        final_dispatch_job=None,
        service_job=None,
        service_amount=0,
        hi_miss_count=0,
    )
    values.update(overrides)
    return TimestampSemanticRecord(**values)


def test_finite_projection_is_diagnostic_only_and_accepts_consistent_record():
    result = check_projection_conformance((_record(),))
    assert result.status == "PASS"


def test_projection_diagnostic_rejects_budget_change_without_controller_action():
    result = check_projection_conformance((
        _record(budget_after_controller=(("t", 1),)),
    ))
    assert result.status == "FAIL"
