"""独立的 mc_stratified_dynamic CLI / manifest 参数 helper。"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Any

from amc_py.dqn import build_mc_stratified_dynamic_experiment_config
from amc_py.workloads.mc_stratified_dynamic import MCStratifiedDynamicWorkloadConfig


_CONFIG_FIELDS = tuple(
    field.name
    for field in fields(MCStratifiedDynamicWorkloadConfig)
    if field.name not in {"seed", "require_schedulable"}
)
_INTEGER_FIELDS = {
    "num_tasks",
    "period_scale",
    "tick_ns",
    "log_uniform_period_min_ms",
    "log_uniform_period_max_ms",
    "max_attempts",
}
_STRING_FIELDS = {"period_family", "sched_method", "priority_policy"}


def _option_name(field_name: str) -> str:
    return "--mc-strat-dyn-" + field_name.replace("_", "-")


def add_mc_stratified_dynamic_args(parser: argparse.ArgumentParser) -> None:
    """向 parser 注入新 workload 的完整、独立参数集合。"""

    defaults = MCStratifiedDynamicWorkloadConfig()
    for field_name in _CONFIG_FIELDS:
        option = _option_name(field_name)
        default = getattr(defaults, field_name)
        if field_name == "period_family":
            parser.add_argument(
                option,
                dest=f"mc_strat_dyn_{field_name}",
                choices=["semi_harmonic", "log_uniform", "seed_paired"],
                default=default,
            )
        elif field_name in _INTEGER_FIELDS:
            parser.add_argument(option, dest=f"mc_strat_dyn_{field_name}", type=int, default=default)
        elif field_name in _STRING_FIELDS:
            parser.add_argument(option, dest=f"mc_strat_dyn_{field_name}", type=str, default=default)
        elif isinstance(default, bool):
            parser.add_argument(
                option,
                dest=f"mc_strat_dyn_{field_name}",
                action=argparse.BooleanOptionalAction,
                default=default,
            )
        else:
            parser.add_argument(option, dest=f"mc_strat_dyn_{field_name}", type=float, default=default)
    # Stratum is a manifest/selection label, not a workload-generator input.
    parser.add_argument("--mc-strat-dyn-stratum", default="unassigned")


def mc_stratified_dynamic_config_dict_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """返回可写入 manifest/config 的完整 generator config dict。"""

    config = {
        field_name: getattr(args, f"mc_strat_dyn_{field_name}")
        for field_name in _CONFIG_FIELDS
    }
    config.update(
        {
            "require_schedulable": bool(args.require_schedulable),
            "scenario_seed_offset": int(args.scenario_seed_offset),
            "fixed_taskset_seed": args.fixed_taskset_seed,
        }
    )
    return config


def mc_stratified_dynamic_config_hash(config: dict[str, Any]) -> str:
    """Compute the same generator-config hash used by candidate manifests.

    Provider/run identity fields such as ``fixed_taskset_seed`` and
    ``scenario_seed_offset`` must not change the workload-generator hash.
    Keeping this payload aligned with ``MCStratifiedDynamicWorkloadConfig``
    makes candidate CSV, training config and HOUT rows directly comparable.
    """

    allowed = {
        field.name
        for field in fields(MCStratifiedDynamicWorkloadConfig)
        if field.name != "seed"
    }
    payload = {key: config[key] for key in sorted(allowed) if key in config}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_mc_stratified_dynamic_config_from_args(args: argparse.Namespace):
    """从新 workload CLI 参数直接构造独立 experiment config。"""

    values = {
        field_name: getattr(args, f"mc_strat_dyn_{field_name}")
        for field_name in _CONFIG_FIELDS
    }
    return build_mc_stratified_dynamic_experiment_config(
        **values,
        require_schedulable=args.require_schedulable,
        scenario_seed_offset=args.scenario_seed_offset,
        fixed_taskset_seed=args.fixed_taskset_seed,
    )


def build_mc_stratified_dynamic_workload_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    """构造新 workload 的带 family/schema 审计信息的 CLI config。"""

    config = mc_stratified_dynamic_config_dict_from_args(args)
    return {
        "workload": "mc_stratified_dynamic",
        "workload_schema_version": "mc_stratified_dynamic_workload_v1",
        "stratum": str(args.mc_strat_dyn_stratum),
        "generator_config_hash": mc_stratified_dynamic_config_hash(config),
        "mc_stratified_dynamic": config,
    }


def assert_mc_stratified_dynamic_args_match_dataset(
    args: argparse.Namespace,
    dataset_dir: Path,
) -> None:
    """ fail-closed 校验 tree CLI 与 teacher dataset 的 workload config 一致。"""

    manifest_path = dataset_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    dataset_config = manifest.get("workload_cli_config")
    if not isinstance(dataset_config, dict):
        raise ValueError(
            "initial_dataset 的 manifest.json 缺少 workload_cli_config，无法校验 mc_stratified_dynamic 参数一致性"
        )
    current_config = build_mc_stratified_dynamic_workload_cli_config(args)
    if dataset_config != current_config:
        raise ValueError(
            "mc_stratified_dynamic workload 参数与 initial_dataset 不一致: "
            f"dataset={dataset_config!r}, current={current_config!r}"
        )


__all__ = [
    "add_mc_stratified_dynamic_args",
    "assert_mc_stratified_dynamic_args_match_dataset",
    "build_mc_stratified_dynamic_config_from_args",
    "build_mc_stratified_dynamic_workload_cli_config",
    "mc_stratified_dynamic_config_dict_from_args",
    "mc_stratified_dynamic_config_hash",
]
