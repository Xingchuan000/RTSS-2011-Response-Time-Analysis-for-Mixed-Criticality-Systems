"""MC-FairGen generator/manifest 回归测试。"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(cwd)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def test_generate_mc_fairgen_uses_distinct_eval_scenarios(tmp_path: Path) -> None:
    """同一 candidate_seed 下，eval seed 应只改变 scenario 而不改变 taskset。"""

    # 使用测试文件位置推导仓库根目录，避免硬编码绝对路径导致跨机器失败。
    repo = Path(__file__).resolve().parents[1]
    from amc_py.dqn.experiment import build_mc_fairgen_experiment_config, resolve_experiment_bundle

    cfg = build_mc_fairgen_experiment_config(
        mode="paper_learnable_headroom",
        num_tasks=8,
        fixed_taskset_seed=0,
    )
    bundles = [resolve_experiment_bundle(cfg, seed) for seed in (100, 101, 102)]

    task_fp = [
        tuple((t.name, t.period, t.deadline, t.c_lo, t.c_hi, t.criticality.value) for t in b.ordered_tasks)
        for b in bundles
    ]
    assert task_fp[0] == task_fp[1] == task_fp[2]

    scenario_fp = []
    for b in bundles:
        triples = []
        for task in b.ordered_tasks[:3]:
            for release in range(5):
                triples.append((task.name, release, b.scenario.actual_cost_for(task, release)))
        scenario_fp.append(tuple(triples))
    assert len(set(scenario_fp)) > 1

    # 额外跑一次最小 generator，保证当前 CLI 路径可执行。
    out_manifest = tmp_path / "manifest.csv"
    out_reject = tmp_path / "reject.csv"
    _run(
        [
            sys.executable,
            "scripts/generate_learnable_tasksets.py",
            "--workload",
            "mc_fairgen",
            "--mc-fairgen-num-tasks",
            "8",
            "--num-tasksets",
            "1",
            "--learnable-max-attempts",
            "5",
            "--learnable-fast-eval-seeds",
            "2",
            "--learnable-fast-end-time",
            "20000",
            "--learnable-fast-event-min",
            "0",
            "--learnable-fast-event-max",
            "1000",
            "--learnable-fast-min-lo-cancellations",
            "0",
            "--learnable-fast-min-lo-cancellation-ratio",
            "0",
            "--learnable-fast-min-valid-decrease",
            "0",
            "--enable-constraint-guided-pair-diagnostic",
            "--constraint-guided-pair-min-valid-count",
            "0",
            "--learnable-selection-target",
            "constraint_guided_pair",
            "--output-manifest",
            str(out_manifest),
            "--output-rejections",
            str(out_reject),
        ],
        repo,
    )
    assert out_manifest.exists()


def test_mc_fairgen_generator_manifest_has_required_fields(tmp_path: Path) -> None:
    """manifest 应包含 MC-FairGen 关键复现字段。"""

    # 使用测试文件位置推导仓库根目录，避免硬编码绝对路径导致跨机器失败。
    repo = Path(__file__).resolve().parents[1]
    out_manifest = tmp_path / "manifest.csv"
    out_reject = tmp_path / "reject.csv"
    _run(
        [
            sys.executable,
            "scripts/generate_learnable_tasksets.py",
            "--workload",
            "mc_fairgen",
            "--mc-fairgen-num-tasks",
            "8",
            "--num-tasksets",
            "1",
            "--learnable-max-attempts",
            "10",
            "--learnable-fast-eval-seeds",
            "1",
            "--learnable-fast-end-time",
            "20000",
            "--learnable-fast-event-min",
            "0",
            "--learnable-fast-event-max",
            "1000",
            "--learnable-fast-min-lo-cancellations",
            "0",
            "--learnable-fast-min-lo-cancellation-ratio",
            "0",
            "--learnable-fast-min-valid-decrease",
            "0",
            "--enable-constraint-guided-pair-diagnostic",
            "--constraint-guided-pair-min-valid-count",
            "0",
            "--learnable-selection-target",
            "constraint_guided_pair",
            "--output-manifest",
            str(out_manifest),
            "--output-rejections",
            str(out_reject),
        ],
        repo,
    )

    with out_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    assert rows
    required = [
        "workload",
        "candidate_seed",
        "mc_fairgen_num_tasks",
        "mc_fairgen_hi_ratio",
        "mc_fairgen_u_hi_lo_min",
        "mc_fairgen_u_hi_lo_max",
        "mc_fairgen_u_hi_hi_min",
        "mc_fairgen_u_hi_hi_max",
        "mc_fairgen_u_lo_lo_min",
        "mc_fairgen_u_lo_lo_max",
        "mc_fairgen_hi_budget_rho_min",
        "mc_fairgen_hi_budget_rho_max",
        "mc_fairgen_lo_budget_rho_min",
        "mc_fairgen_lo_budget_rho_max",
        "fast_baseline_lo_cancellation_ratio_total",
    ]
    missing = [c for c in required if c not in fields]
    assert not missing
