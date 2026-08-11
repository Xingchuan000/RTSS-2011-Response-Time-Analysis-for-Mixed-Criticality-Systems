"""独立 mc_stratified_dynamic CLI helper 测试。"""

from __future__ import annotations

import argparse
import json

import pytest

from scripts.common_mc_stratified_dynamic_cli import (
    add_mc_stratified_dynamic_args,
    assert_mc_stratified_dynamic_args_match_dataset,
    build_mc_stratified_dynamic_config_from_args,
    build_mc_stratified_dynamic_workload_cli_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-schedulable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scenario-seed-offset", type=int, default=100000)
    parser.add_argument("--fixed-taskset-seed", type=int, default=None)
    add_mc_stratified_dynamic_args(parser)
    return parser


def test_new_cli_uses_only_mc_stratified_dynamic_prefix() -> None:
    args = _parser().parse_args(
        [
            "--mc-strat-dyn-num-tasks",
            "8",
            "--mc-strat-dyn-period-family",
            "semi_harmonic",
            "--mc-strat-dyn-stratum",
            "S3",
        ]
    )
    config = build_mc_stratified_dynamic_workload_cli_config(args)
    assert config["workload"] == "mc_stratified_dynamic"
    assert config["stratum"] == "S3"
    assert config["mc_stratified_dynamic"]["num_tasks"] == 8
    assert len(str(config["generator_config_hash"])) == 64
    assert not hasattr(args, "mc_fairgen_mode")


def test_new_cli_builds_independent_provider() -> None:
    args = _parser().parse_args(["--mc-strat-dyn-num-tasks", "6"])
    config = build_mc_stratified_dynamic_config_from_args(args)
    assert config.workload_provider is not None
    assert config.workload_provider.name == "mc_stratified_dynamic"


def test_dataset_config_mismatch_is_detected(tmp_path) -> None:
    args = _parser().parse_args(["--mc-strat-dyn-num-tasks", "6"])
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = build_mc_stratified_dynamic_workload_cli_config(args)
    (dataset / "manifest.json").write_text(json.dumps({"workload_cli_config": manifest}), encoding="utf-8")
    assert_mc_stratified_dynamic_args_match_dataset(args, dataset)

    changed = _parser().parse_args(["--mc-strat-dyn-num-tasks", "8"])
    with pytest.raises(ValueError, match="mc_stratified_dynamic"):
        assert_mc_stratified_dynamic_args_match_dataset(changed, dataset)
