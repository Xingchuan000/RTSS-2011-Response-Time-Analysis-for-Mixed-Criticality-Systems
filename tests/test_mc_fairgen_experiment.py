"""MC-FairGen experiment config 接入测试（Step5）。"""

from __future__ import annotations

from amc_py.dqn.experiment import (
    build_mc_fairgen_experiment_config,
    resolve_experiment_bundle,
)


def test_build_mc_fairgen_experiment_config_resolves_bundle() -> None:
    """mc_fairgen builder 应可解析出完整 bundle。"""

    config = build_mc_fairgen_experiment_config(fixed_taskset_seed=0, num_tasks=16)
    bundle = resolve_experiment_bundle(config, seed=0)

    assert len(bundle.ordered_tasks) == 16
    assert bundle.scenario is not None
    assert bundle.normalization_bounds is not None
    assert bundle.metadata is not None
    assert bundle.metadata["workload_family"] == "mc_fairgen"
