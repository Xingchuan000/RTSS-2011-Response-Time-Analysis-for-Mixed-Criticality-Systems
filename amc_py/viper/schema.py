"""VIPER 数据集、artifact 与部署语义版本常量。

这些常量用于阻止新代码把不同编码或不同运行时语义的文件混在一起。
"""

VIPER_DATASET_SCHEMA_VERSION = "viper_fixed_v1"
VIPER_ARTIFACT_SCHEMA_VERSION = "viper_integer_artifact_v1"
INTEGER_TREE_SCHEMA_VERSION = "integer_tree_v1"
DEPLOYMENT_SEMANTICS_VERSION = "formal_deployment_v1"


def resolve_deployment_semantics_version(
    *,
    tree_state_encoding: str,
    tree_fallback_mode: str,
    action_validation_mode: str,
    strict_candidate_deploy_cap: bool,
    carry_over_aware_safety: bool,
    lo_budget_overrun_guard_units: int,
) -> str:
    """统一判定部署语义版本字符串。

    只有以下条件全部成立时才返回 "formal_deployment_v1"：
    - tree_state_encoding == "fixed_point_int"
    - tree_fallback_mode == "top1_or_noop"
    - action_validation_mode == "formal_v1"
    - strict_candidate_deploy_cap == True
    - carry_over_aware_safety == True
    - lo_budget_overrun_guard_units == 1

    其他组合返回明确的 legacy/mixed 版本，阻止误判为正式语义。
    """
    if (
        tree_state_encoding == "fixed_point_int"
        and tree_fallback_mode == "top1_or_noop"
        and action_validation_mode == "formal_v1"
        and strict_candidate_deploy_cap is True
        and carry_over_aware_safety is True
        and lo_budget_overrun_guard_units == 1
    ):
        return "formal_deployment_v1"
    if action_validation_mode == "formal_v1" or strict_candidate_deploy_cap or carry_over_aware_safety:
        return "legacy_mixed_semantics_v1"
    return "legacy_baseline_v1"

