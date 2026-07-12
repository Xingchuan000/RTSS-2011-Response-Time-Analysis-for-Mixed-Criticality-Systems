"""VIPER 策略提取相关模块导出。

这里对 `sklearn/joblib` 相关导入做了轻量延迟兼容：
- 在尚未安装新增依赖时，`import amc_py.viper` 仍应成功；
- 只有真正访问 tree artifact / trainer / sklearn 评估能力时，才要求这些依赖存在。
"""

from .dataset import ViperSample, read_viper_dataset, samples_to_xyw, write_viper_dataset
from .fixed_point import FixedPointConfig, quantize_state_vector
from .integer_tree import IntegerTreeModel, compile_sklearn_tree_to_integer, load_integer_tree_json
from .registry import build_teacher_registry_row
from .selection import SelectionConfig, select_best_tree
from .splits import assert_disjoint_splits, parse_seed_spec, validate_viper_split_config
from .teacher import collect_teacher_labeled_rollouts
from .tree_policy import IntegerTreeBudgetPolicy, TreeBudgetPolicy, TreePolicyProtocol

try:
    from .artifacts import export_tree_rules_text, load_tree_policy_artifact, save_tree_policy_artifact
    from .metrics import (
        compute_offline_tree_metrics,
        evaluate_tree_policy_once,
        retention_higher_is_better,
        retention_lower_is_better,
    )
    from .training import TreeHyperParams, run_viper_iterations, train_cart_tree
except ModuleNotFoundError:
    export_tree_rules_text = None
    load_tree_policy_artifact = None
    save_tree_policy_artifact = None
    compute_offline_tree_metrics = None
    evaluate_tree_policy_once = None
    retention_higher_is_better = None
    retention_lower_is_better = None
    TreeHyperParams = None
    run_viper_iterations = None
    train_cart_tree = None

__all__ = [
    "SelectionConfig",
    "TreeBudgetPolicy",
    "IntegerTreeBudgetPolicy",
    "TreePolicyProtocol",
    "FixedPointConfig",
    "IntegerTreeModel",
    "compile_sklearn_tree_to_integer",
    "load_integer_tree_json",
    "quantize_state_vector",
    "TreeHyperParams",
    "ViperSample",
    "assert_disjoint_splits",
    "build_teacher_registry_row",
    "collect_teacher_labeled_rollouts",
    "compute_offline_tree_metrics",
    "evaluate_tree_policy_once",
    "export_tree_rules_text",
    "load_tree_policy_artifact",
    "parse_seed_spec",
    "read_viper_dataset",
    "retention_higher_is_better",
    "retention_lower_is_better",
    "run_viper_iterations",
    "samples_to_xyw",
    "save_tree_policy_artifact",
    "select_best_tree",
    "train_cart_tree",
    "validate_viper_split_config",
    "write_viper_dataset",
]
