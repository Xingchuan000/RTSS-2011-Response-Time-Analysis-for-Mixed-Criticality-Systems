"""VIPER seed split 校验工具。"""

from __future__ import annotations


def parse_seed_spec(raw_value: str) -> set[int]:
    """解析与现有脚本一致的 seed 表达式。"""

    result: set[int] = set()
    for part in (item.strip() for item in raw_value.split(",")):
        if not part:
            continue
        if ":" in part:
            begin_text, end_text = (token.strip() for token in part.split(":", maxsplit=1))
            begin = int(begin_text)
            end = int(end_text)
            if end < begin:
                raise ValueError(f"非法 seed 区间: {part}")
            result.update(range(begin, end + 1))
        else:
            result.add(int(part))
    return result


def assert_disjoint_splits(splits: dict[str, set[int]]) -> None:
    """断言多个 split 之间互不重叠。"""

    names = list(splits.keys())
    for idx, name in enumerate(names):
        for other_name in names[idx + 1:]:
            overlap = splits[name] & splits[other_name]
            if overlap:
                preview = ",".join(str(item) for item in sorted(overlap)[:10])
                raise ValueError(f"{name} 与 {other_name} 存在重叠 seeds: {preview}")


def validate_viper_split_config(config: dict[str, object]) -> dict[str, object]:
    """把原始 JSON 配置解析并做 strict 检查。"""

    parsed = {key: parse_seed_spec(str(value)) for key, value in config.items()}
    assert_disjoint_splits(
        {
            "viper_train_seeds": parsed.get("viper_train_seeds", set()),
            "viper_validation_seeds": parsed.get("viper_validation_seeds", set()),
            "strict_final_hout_seeds": parsed.get("strict_final_hout_seeds", set()),
        }
    )
    dqn_training = parsed.get("dqn_training_episode_seeds", set())
    strict_targets = {
        "viper_train_seeds": parsed.get("viper_train_seeds", set()),
        "viper_validation_seeds": parsed.get("viper_validation_seeds", set()),
        "strict_final_hout_seeds": parsed.get("strict_final_hout_seeds", set()),
    }
    for name, seeds in strict_targets.items():
        overlap = seeds & dqn_training
        if overlap:
            preview = ",".join(str(item) for item in sorted(overlap)[:10])
            raise ValueError(f"{name} 与 dqn_training_episode_seeds 重叠: {preview}")
    warnings: list[str] = []
    old_validation = parsed.get("old_validation_seeds", set())
    legacy_hout = parsed.get("legacy_hout_seeds", set())
    if old_validation & legacy_hout:
        warnings.append("legacy_only: old_validation_seeds 与 legacy_hout_seeds 有重叠")
    if old_validation & dqn_training:
        warnings.append("legacy_only: old_validation_seeds 与 dqn_training_episode_seeds 有重叠")
    if legacy_hout & dqn_training:
        warnings.append("legacy_only: legacy_hout_seeds 与 dqn_training_episode_seeds 有重叠")
    return {
        "parsed_splits": {key: sorted(value) for key, value in parsed.items()},
        "warnings": warnings,
    }
