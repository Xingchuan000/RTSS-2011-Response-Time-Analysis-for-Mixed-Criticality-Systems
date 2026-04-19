from pathlib import Path

from amc_py.experiments import (
    compute_weighted_schedulability,
    plot_schedulable_percentage,
    plot_weighted_schedulability,
    run_cf_sweep,
    run_cp_sweep,
    run_taskset_size_sweep,
    run_utilization_sweep,
)


def test_run_utilization_sweep_and_weighted_metrics() -> None:
    # 使用小样本运行 utilization sweep，验证结果结构与统计函数。
    results = run_utilization_sweep(
        util_values=[0.4, 0.6],
        num_tasksets=4,
        num_tasks=5,
        method="smc",
        priority_policy="dm",
        seed=11,
    )

    assert not results.empty
    assert {"sweep_type", "sweep_value", "schedulable", "actual_total_util_lo"}.issubset(results.columns)

    weighted = compute_weighted_schedulability(results)
    assert not weighted.empty
    assert {"weighted_schedulability", "schedulable_ratio"}.issubset(weighted.columns)


def test_run_other_sweeps() -> None:
    # 覆盖 cf/cp/n 三类 sweep 接口，确保能返回非空表格。
    cf_results = run_cf_sweep(
        cf_values=[1.5, 2.0],
        num_tasksets=3,
        num_tasks=5,
        total_util=0.6,
        method="amc_rtb",
        priority_policy="dm",
        seed=3,
    )
    cp_results = run_cp_sweep(
        cp_values=[0.25, 0.75],
        num_tasksets=3,
        num_tasks=5,
        total_util=0.6,
        method="amc_rtb",
        priority_policy="dm",
        seed=5,
    )
    n_results = run_taskset_size_sweep(
        taskset_sizes=[4, 6],
        num_tasksets=3,
        total_util=0.6,
        method="amc_rtb",
        priority_policy="dm",
        seed=7,
    )

    assert not cf_results.empty
    assert not cp_results.empty
    assert not n_results.empty


def test_plot_functions(tmp_path: Path) -> None:
    # 验证两种绘图函数都能落盘图片文件。
    results = run_utilization_sweep(
        util_values=[0.3, 0.5],
        num_tasksets=3,
        num_tasks=4,
        method="smc",
        priority_policy="dm",
        seed=13,
    )
    weighted = compute_weighted_schedulability(results)

    fig1 = tmp_path / "sched_ratio.png"
    fig2 = tmp_path / "weighted_ratio.png"

    out1 = plot_schedulable_percentage(results, output_path=fig1)
    out2 = plot_weighted_schedulability(weighted, output_path=fig2)

    assert out1 is not None and out1.exists()
    assert out2 is not None and out2.exists()
