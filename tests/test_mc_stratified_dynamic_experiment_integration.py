"""mc_stratified_dynamic experiment/train/evaluate/VIPER 集成 smoke。"""

from __future__ import annotations

from amc_py.dqn import (
    build_experiment_config,
    build_mc_fairgen_experiment_config,
    build_mc_stratified_dynamic_experiment_config,
    resolve_experiment_bundle,
)
from scripts.collect_viper_teacher_data import build_parser as build_collect_parser
from scripts.evaluate_dqn_amc import build_parser as build_evaluate_parser
from scripts.train_dqn_amc import build_parser as build_train_parser
from scripts.train_viper_tree import build_parser as build_tree_parser


def test_experiment_factory_and_legacy_builder_are_separate() -> None:
    new_config = build_experiment_config(
        "mc_stratified_dynamic",
        num_tasks=6,
        fixed_taskset_seed=17,
        require_schedulable=False,
    )
    assert new_config.workload_provider is not None
    assert new_config.workload_provider.name == "mc_stratified_dynamic"
    assert build_mc_fairgen_experiment_config(num_tasks=8).name.startswith("mc_fairgen_")


def test_same_fixed_taskset_and_scenario_seed_match() -> None:
    kwargs = dict(num_tasks=6, fixed_taskset_seed=17, scenario_seed_offset=100000)
    train_bundle = resolve_experiment_bundle(build_mc_stratified_dynamic_experiment_config(**kwargs), 23)
    eval_bundle = resolve_experiment_bundle(build_mc_stratified_dynamic_experiment_config(**kwargs), 23)
    assert train_bundle.taskset_fingerprint == eval_bundle.taskset_fingerprint
    assert train_bundle.taskset_seed == eval_bundle.taskset_seed == 17
    assert train_bundle.scenario_seed == eval_bundle.scenario_seed == 100023
    assert train_bundle.scenario.actual_cost_for(train_bundle.ordered_tasks[0], 0) == eval_bundle.scenario.actual_cost_for(
        eval_bundle.ordered_tasks[0], 0
    )


def test_all_runtime_cli_parsers_accept_new_workload() -> None:
    train_args = build_train_parser().parse_args(["--workload", "mc_stratified_dynamic"])
    eval_args = build_evaluate_parser().parse_args(
        ["--workload", "mc_stratified_dynamic", "--model", "model.pt"]
    )
    collect_args = build_collect_parser().parse_args(
        [
            "--workload",
            "mc_stratified_dynamic",
            "--model",
            "model.pt",
            "--teacher-id",
            "teacher",
            "--output-dir",
            "dataset",
        ]
    )
    tree_args = build_tree_parser().parse_args(
        [
            "--workload",
            "mc_stratified_dynamic",
            "--method",
            "bc",
            "--teacher-model",
            "model.pt",
            "--teacher-id",
            "teacher",
            "--output-dir",
            "tree",
        ]
    )
    assert all(
        args.workload == "mc_stratified_dynamic"
        for args in (train_args, eval_args, collect_args, tree_args)
    )
