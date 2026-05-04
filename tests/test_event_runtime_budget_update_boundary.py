"""预算更新边界语义回归测试。"""

from __future__ import annotations

from amc_py.budget_runtime import BudgetState
from amc_py.event_runtime import EventRuntimeEngine
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def test_budget_update_does_not_retroactively_change_released_job_budget() -> None:
    """已释放 job 应继续使用 release 时预算，不受后续预算更新影响。"""

    task = Task("lo", period=20, deadline=20, c_lo=3, c_hi=3, criticality=Criticality.LO)
    engine = EventRuntimeEngine.build(
        ordered_tasks=[task],
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_PLUS),
        budget_state=BudgetState.from_tasks([task]),
    )
    engine.run_until(1, include_boundary=True)
    engine.apply_budget_updates({"lo": 1})
    engine.run_until(10, include_boundary=True)
    result = engine.finish()

    assert len(result.job_cancellations) == 0
    assert any(job.task.name == "lo" and job.release_index == 0 and job.completion_time is not None for job in result.jobs)
