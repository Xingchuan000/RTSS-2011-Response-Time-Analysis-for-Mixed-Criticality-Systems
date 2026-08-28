from types import SimpleNamespace

import pytest

from amc_py.models import Criticality, Task
from amc_py.rl.observation import TaskNormalizationBound
from formal_toolchain.v9_2.bindings import _freeze_normalization_bounds


def _task(name: str = "t") -> Task:
    return Task(name=name, period=10, deadline=10, criticality=Criticality.HI, c_lo=2, c_hi=5)


def test_freezes_custom_integer_normalization_bounds() -> None:
    task = _task()
    env = SimpleNamespace(normalization_bounds={task.name: TaskNormalizationBound(1.0, 7.0)})
    assert _freeze_normalization_bounds(env, (task,)) == {task.name: {"min_cost": 1, "max_cost": 7}}


def test_rejects_nonintegral_normalization_bounds() -> None:
    task = _task()
    env = SimpleNamespace(normalization_bounds={task.name: TaskNormalizationBound(0.5, 7.0)})
    with pytest.raises(ValueError, match="NONINTEGRAL_NORMALIZATION_BOUND"):
        _freeze_normalization_bounds(env, (task,))
