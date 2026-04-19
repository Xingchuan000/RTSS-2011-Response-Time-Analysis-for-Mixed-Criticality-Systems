from pathlib import Path

from amc_py.generator import (
    generate_task,
    generate_taskset,
    load_generation_config,
    make_generation_config,
    sample_period_log_uniform,
    taskset_total_util,
    uunifast,
)
from amc_py.models import Criticality


def test_uunifast_sum_close() -> None:
    # 验证 UUniFast 生成向量总和接近目标利用率。
    utils = uunifast(num_tasks=6, total_util=0.8)
    assert len(utils) == 6
    assert abs(sum(utils) - 0.8) < 1e-8
    assert all(value > 0 for value in utils)


def test_period_sampling_range() -> None:
    # 多次采样都应落在指定范围内。
    for _ in range(100):
        period = sample_period_log_uniform(10, 100)
        assert 10 <= period <= 100


def test_generate_taskset_basic_properties_fixed_count() -> None:
    # fixed_count 下，HI 任务个数应稳定为 round(n*cp)。
    taskset = generate_taskset(
        num_tasks=8,
        total_util=0.9,
        min_period=10,
        max_period=100,
        time_scale=10,
        cf=2.0,
        cp=0.5,
        seed=7,
        deadline_mode="implicit",
        criticality_assignment="fixed_count",
        lo_hi_budget_policy="scaled_by_cf",
    )

    assert len(taskset) == 8
    hi_count = sum(1 for task in taskset if task.criticality is Criticality.HI)
    assert hi_count == 4

    util_lo = taskset_total_util(taskset)
    util_hi = taskset_total_util(taskset, mode=Criticality.HI)
    assert util_lo > 0
    assert util_hi >= 0


def test_implicit_deadline_mode_sets_d_equal_t() -> None:
    taskset = generate_taskset(
        num_tasks=5,
        total_util=0.4,
        min_period=10,
        max_period=100,
        time_scale=10,
        cf=2.0,
        cp=0.5,
        seed=1,
        deadline_mode="implicit",
        criticality_assignment="bernoulli",
        lo_hi_budget_policy="scaled_by_cf",
    )
    assert all(task.deadline == task.period for task in taskset)


def test_generate_task_allows_hi_c_hi_greater_than_deadline_in_implicit_mode() -> None:
    task = generate_task(
        task_id=1,
        util=0.3,
        period=100,
        criticality=Criticality.HI,
        cf=5.0,
        deadline_mode="implicit",
    )

    assert task.deadline == task.period
    assert task.c_hi > task.deadline


def test_high_cf_implicit_taskset_generation_does_not_fail() -> None:
    taskset = generate_taskset(
        num_tasks=20,
        total_util=0.8,
        min_period=10,
        max_period=1000,
        time_scale=100,
        cf=5.0,
        cp=0.5,
        seed=2026,
        deadline_mode="implicit",
        criticality_assignment="bernoulli",
        lo_hi_budget_policy="scaled_by_cf",
    )
    assert len(taskset) == 20


def test_arbitrary_paper_deadline_respects_paper_bounds() -> None:
    taskset = generate_taskset(
        num_tasks=10,
        total_util=0.6,
        min_period=10,
        max_period=100,
        time_scale=10,
        cf=2.0,
        cp=0.5,
        deadline_mode="arbitrary_paper",
        deadline_ratio_min=0.5,
        criticality_assignment="bernoulli",
        lo_hi_budget_policy="scaled_by_cf",
        seed=2,
    )
    for task in taskset:
        if task.criticality is Criticality.HI:
            assert task.c_hi <= task.deadline <= task.period
        else:
            assert task.c_lo <= task.deadline <= task.period


def test_actual_util_close_to_target_under_time_scale() -> None:
    taskset = generate_taskset(
        num_tasks=20,
        total_util=0.025,
        min_period=10,
        max_period=1000,
        time_scale=100,
        cf=2.0,
        cp=0.5,
        seed=1,
        deadline_mode="implicit",
        criticality_assignment="bernoulli",
        lo_hi_budget_policy="scaled_by_cf",
    )
    actual = taskset_total_util(taskset)
    assert abs(actual - 0.025) < 0.02


def test_lo_task_can_have_separate_chi_in_scaled_policy() -> None:
    taskset = generate_taskset(
        num_tasks=12,
        total_util=0.5,
        min_period=10,
        max_period=200,
        time_scale=10,
        cf=2.0,
        cp=0.5,
        seed=1234,
        deadline_mode="implicit",
        criticality_assignment="bernoulli",
        lo_hi_budget_policy="scaled_by_cf",
    )
    lo_tasks = [task for task in taskset if task.criticality is Criticality.LO]
    assert lo_tasks
    assert any(task.c_hi > task.c_lo for task in lo_tasks)


def test_make_generation_config_modes() -> None:
    fast = make_generation_config("fast")
    paper = make_generation_config("paper")

    assert fast.num_tasks > 0
    assert paper.num_tasks > 0
    assert fast.deadline_mode == "implicit"
    assert paper.deadline_mode == "implicit"
    assert fast.time_scale == 10
    assert paper.time_scale == 100
    assert fast.lo_hi_budget_policy == "scaled_by_cf"
    assert paper.lo_hi_budget_policy == "scaled_by_cf"
    assert fast.max_period != paper.max_period


def test_paper_mode_uses_implicit_deadline() -> None:
    cfg = make_generation_config("paper")
    taskset = generate_taskset(
        num_tasks=cfg.num_tasks,
        total_util=cfg.total_util,
        min_period=cfg.min_period,
        max_period=cfg.max_period,
        time_scale=cfg.time_scale,
        cf=cfg.cf,
        cp=cfg.cp,
        seed=2026,
        deadline_mode=cfg.deadline_mode,
        deadline_ratio_min=cfg.deadline_ratio_min,
        criticality_assignment=cfg.criticality_assignment,
        lo_hi_budget_policy=cfg.lo_hi_budget_policy,
    )

    assert all(task.deadline == task.period for task in taskset)


def test_load_generation_config_from_yaml() -> None:
    cfg = load_generation_config(Path("configs/generator_paper.yaml"))
    assert cfg.time_scale == 100
    assert cfg.deadline_mode == "implicit"
    assert cfg.criticality_assignment == "bernoulli"
    assert cfg.lo_hi_budget_policy == "scaled_by_cf"


def test_load_generation_config_compatible_with_legacy_bool(tmp_path: Path) -> None:
    yaml_path = tmp_path / "legacy_gen.yaml"
    yaml_path.write_text(
        """
num_tasks: 6
total_util: 0.7
min_period: 10
max_period: 100
time_scale: 10
cf: 2.0
cp: 0.5
deadline_equals_period: false
deadline_ratio_min: 0.6
criticality_assignment: bernoulli
lo_hi_budget_policy: scaled_by_cf
""".strip(),
        encoding="utf-8",
    )

    cfg = load_generation_config(yaml_path)
    assert cfg.num_tasks == 6
    assert cfg.deadline_mode == "ratio_uniform"
    assert cfg.criticality_assignment == "bernoulli"
