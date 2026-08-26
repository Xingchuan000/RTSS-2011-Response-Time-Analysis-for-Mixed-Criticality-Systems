"""决策树策略封装。

新增 leaf-level execution audit 能力：
- _predict_full_proba(): 复用概率计算逻辑，供 predict_action_ranking 和 trace_decision_path 共享。
- trace_decision_path(): 返回当前 state 在 sklearn decision tree 中的 leaf/path/probability 诊断信息。
- select_action_id(): 新增 include_decision_trace 参数，可选将 trace 信息合并到 info 字典中。
- action_definition(): 按 action_id 返回动作语义描述，供 audit log 使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from amc_py.viper.fixed_point import (
    FixedPointConfig,
    fixed_point_config_hash,
    quantize_state_vector,
)
from amc_py.viper.integer_tree import IntegerTreeModel, _validate_integer_tree_model, evaluate_integer_tree


def _explicit_noop_action_id(action_definitions: list[dict[str, object]]) -> int | None:
    """Return the unique explicit-noop action id encoded by the artifact.

    V7 treats ``a_noop`` as a real member of the deployed action alphabet.  The
    normal path still selects it through ranked-first-valid.  This helper is
    used only by the defensive all-invalid fallback required by A8; legacy
    artifacts without an explicit noop keep the historical ``None`` fallback.
    """

    noop_ids: list[int] = []
    for index, definition in enumerate(action_definitions):
        if not bool(definition.get("is_noop", False)):
            continue
        action_id = int(definition.get("action_id", index))
        if action_id != index:
            raise ValueError("explicit noop action_id 必须与 action_definitions 下标一致")
        noop_ids.append(action_id)
    if len(noop_ids) > 1:
        raise ValueError("action_definitions 最多只能包含一个 explicit noop")
    return noop_ids[0] if noop_ids else None


class TreePolicyProtocol(Protocol):
    """legacy sklearn policy 与整数部署 policy 的共同调用协议。"""

    metadata: dict[str, object]
    feature_names: tuple[str, ...]
    action_definitions: list[dict[str, object]]

    def select_action_id(self, state_vector: tuple[float, ...], valid_action_mask: tuple[bool, ...] | None, *, include_decision_trace: bool = False) -> tuple[int | None, dict[str, object]]: ...
    def trace_decision_path(self, state_vector: tuple[float, ...]) -> dict[str, object]: ...
    def action_definition(self, action_id: int | None) -> dict[str, object] | None: ...


@dataclass(slots=True)
class TreeBudgetPolicy:
    """对 sklearn 决策树做一层 mask-aware runtime 封装。"""

    classifier: Any
    metadata: dict[str, object]
    feature_names: tuple[str, ...]
    action_definitions: list[dict[str, object]]

    def _predict_full_proba(self, state_vector: tuple[float, ...]) -> np.ndarray:
        """为所有 action 生成完整概率向量。

        说明：
        - sklearn 的 predict_proba 只覆盖训练时见过的 classes_；
        - 这里先给所有 action 初始化为 0 概率，再回填 seen classes；
        - dtype 使用 np.float64，保证精度一致。
        """

        state_dim = int(self.metadata.get("state_dim", len(self.feature_names)))
        if len(state_vector) != state_dim:
            raise ValueError(
                f"state_vector 维度不匹配: got={len(state_vector)} expected={state_dim}"
            )
        action_dim = int(self.metadata.get("action_dim", len(self.action_definitions)))
        x = np.asarray([state_vector], dtype=np.float32)
        proba = np.zeros(action_dim, dtype=np.float64)
        predicted = self.classifier.predict_proba(x)[0]
        for class_index, class_id in enumerate(self.classifier.classes_):
            proba[int(class_id)] = float(predicted[class_index])
        return proba

    def predict_action_ranking(self, state_vector: tuple[float, ...]) -> tuple[int, ...]:
        """返回完整动作编号排序。

        说明：
        - 复用 _predict_full_proba 获取概率；
        - 同概率时按 action_id 升序，保证 artifact 在重复加载后仍完全确定。
        """

        proba = self._predict_full_proba(state_vector)
        action_dim = int(self.metadata.get("action_dim", len(self.action_definitions)))
        ranking = sorted(range(action_dim), key=lambda action_id: (-proba[action_id], action_id))
        return tuple(int(action_id) for action_id in ranking)

    def trace_decision_path(self, state_vector: tuple[float, ...]) -> dict[str, object]:
        """返回当前 state 在 sklearn decision tree 中的 leaf/path/probability 诊断信息。

        返回字段包括：
        - tree_leaf_id: 命中的叶子节点编号。
        - tree_path_node_ids: 从根到叶子经过的所有节点编号。
        - tree_path_depth: 路径深度（节点数 - 1）。
        - tree_path_predicates: 每个内部节点的分裂条件与 state 实际取值。
        - tree_leaf_impurity: 叶子节点纯度。
        - tree_leaf_n_node_samples: 叶子节点训练样本数。
        - tree_leaf_weighted_n_node_samples: 叶子节点加权训练样本数。
        - tree_leaf_value: 叶子节点类别分布（flatten 为普通 Python list）。
        - tree_leaf_predicted_class_id: 叶子节点预测的动作类别编号。
        - tree_action_proba: 完整动作概率向量。
        - tree_action_ranking: 完整动作排名。
        """

        action_dim = int(self.metadata.get("action_dim", len(self.action_definitions)))
        x = np.asarray([state_vector], dtype=np.float32)

        # 使用 sklearn API 获取叶子编号与决策路径
        leaf_id = int(self.classifier.apply(x)[0])
        node_indicator = self.classifier.decision_path(x)
        node_ids = node_indicator.indices[
            node_indicator.indptr[0] : node_indicator.indptr[1]
        ]

        # 读取底层 CART tree 结构
        tree = self.classifier.tree_
        path_predicates: list[dict[str, object]] = []
        for node_id in node_ids:
            feature_idx = int(tree.feature[node_id])
            # 叶子节点的 feature 为 -2（TREE_UNDEFINED），跳过
            if feature_idx < 0:
                continue
            threshold = float(tree.threshold[node_id])
            value = float(state_vector[feature_idx])
            predicate: dict[str, object] = {
                "node_id": int(node_id),
                "feature_index": feature_idx,
                "feature_name": self.feature_names[feature_idx],
                "threshold": threshold,
                "value": value,
                "operator": "<=" if value <= threshold else ">",
                "decision": "left" if value <= threshold else "right",
            }
            path_predicates.append(predicate)

        # 叶子节点的训练统计信息
        leaf_value_flat = tree.value[leaf_id].ravel().tolist()
        leaf_predicted_class_raw = int(tree.value[leaf_id].ravel().argmax())
        # 将 tree 内部的 argmax 映射回 classifier.classes_ 中的真实 action_id
        leaf_predicted_action_id = int(self.classifier.classes_[leaf_predicted_class_raw])

        # 完整概率与排名
        proba = self._predict_full_proba(state_vector)
        ranking = sorted(range(action_dim), key=lambda aid: (-proba[aid], aid))

        return {
            "tree_leaf_id": leaf_id,
            "tree_path_node_ids": tuple(int(nid) for nid in node_ids),
            "tree_path_depth": len(node_ids) - 1,
            "tree_path_predicates": path_predicates,
            "tree_leaf_impurity": float(tree.impurity[leaf_id]),
            "tree_leaf_n_node_samples": int(tree.n_node_samples[leaf_id]),
            "tree_leaf_weighted_n_node_samples": float(tree.weighted_n_node_samples[leaf_id]),
            "tree_leaf_value": leaf_value_flat,
            "tree_leaf_predicted_class_id": leaf_predicted_action_id,
            "tree_action_proba": proba.tolist(),
            "tree_action_ranking": tuple(int(aid) for aid in ranking),
        }

    def select_action_id(
        self,
        state_vector: tuple[float, ...],
        valid_action_mask: tuple[bool, ...] | None,
        *,
        include_decision_trace: bool = False,
    ) -> tuple[int | None, dict[str, object]]:
        """在 tree 原始排序上叠加当前 runtime mask。

        新增参数：
        - include_decision_trace: 若为 True，则将 trace_decision_path() 的返回字段
          合并到返回的 info 字典中，用于 leaf-level audit。
        """

        # 如果开启了 decision trace，使用 trace 中的 ranking 避免重复 predict_proba
        if include_decision_trace:
            trace = self.trace_decision_path(state_vector)
            ranking = trace["tree_action_ranking"]
        else:
            ranking = self.predict_action_ranking(state_vector)

        raw_top1 = ranking[0] if ranking else None
        if valid_action_mask is None:
            info: dict[str, object] = {
                "tree_raw_top1_action_id": raw_top1,
                "tree_raw_top1_invalid": False,
                "tree_fallback_used": False,
                "tree_no_valid_action": raw_top1 is None,
                "tree_selected_action_id": raw_top1,
            }
            if include_decision_trace:
                info.update(trace)
            return raw_top1, info
        if len(valid_action_mask) != len(self.action_definitions):
            raise ValueError("valid_action_mask 长度必须与 action_dim 一致")
        raw_invalid = raw_top1 is not None and not bool(valid_action_mask[raw_top1])
        for candidate in ranking:
            if bool(valid_action_mask[candidate]):
                info = {
                    "tree_raw_top1_action_id": raw_top1,
                    "tree_raw_top1_invalid": bool(raw_invalid),
                    "tree_fallback_used": bool(raw_invalid and candidate != raw_top1),
                    "tree_no_valid_action": False,
                    "tree_selected_action_id": int(candidate),
                }
                if include_decision_trace:
                    info.update(trace)
                return int(candidate), info
        explicit_noop_action_id = _explicit_noop_action_id(self.action_definitions)
        if explicit_noop_action_id is not None:
            info = {
                "tree_raw_top1_action_id": raw_top1,
                "tree_raw_top1_invalid": bool(raw_invalid),
                "tree_fallback_used": True,
                "tree_no_valid_action": True,
                "tree_selected_action_id": explicit_noop_action_id,
                "tree_defensive_fallback_to_explicit_noop": True,
            }
            if include_decision_trace:
                info.update(trace)
            return explicit_noop_action_id, info
        info = {
            "tree_raw_top1_action_id": raw_top1,
            "tree_raw_top1_invalid": bool(raw_invalid),
            "tree_fallback_used": False,
            "tree_no_valid_action": True,
            "tree_selected_action_id": None,
        }
        if include_decision_trace:
            info.update(trace)
        return None, info

    def action_definition(self, action_id: int | None) -> dict[str, object] | None:
        """按 action_id 返回动作语义描述，供 audit log 写出可读动作信息。

        参数：
        - action_id: 动作编号；若为 None 则返回 None。
        返回：
        - 对应的 action_definitions 条目副本，或 None（当 action_id 无效时）。
        """

        if action_id is None:
            return None
        if action_id < 0 or action_id >= len(self.action_definitions):
            return None
        return dict(self.action_definitions[action_id])


@dataclass(slots=True)
class IntegerTreeBudgetPolicy:
    """接收原始 float observation、内部量化并执行整数 tree 的策略。"""

    model: IntegerTreeModel
    metadata: dict[str, object]
    feature_names: tuple[str, ...]
    action_definitions: list[dict[str, object]]
    fixed_point_config: FixedPointConfig

    def __post_init__(self) -> None:
        """对象构造时即做 fail-closed 校验，避免绕过 loader 直接注入非法配置。"""

        if len(self.feature_names) != self.model.state_dim:
            raise ValueError("feature_names 与 model.state_dim 不一致")
        if len(self.action_definitions) != self.model.action_dim:
            raise ValueError("action_definitions 与 model.action_dim 不一致")
        _ = self.metadata.get("state_dim")
        if int(self.metadata.get("state_dim", self.model.state_dim)) != self.model.state_dim:
            raise ValueError("metadata.state_dim 与 model.state_dim 不一致")
        if int(self.metadata.get("action_dim", self.model.action_dim)) != self.model.action_dim:
            raise ValueError("metadata.action_dim 与 model.action_dim 不一致")
        _validate_integer_tree_model(self.model)
        config_hash = fixed_point_config_hash(self.fixed_point_config)
        if self.model.fixed_point_config_hash != config_hash:
            raise ValueError("model.fixed_point_config_hash 与 fixed_point_config 不一致")
        metadata_config = self.metadata.get("fixed_point_config")
        if metadata_config is not None and not isinstance(metadata_config, dict):
            raise ValueError("metadata.fixed_point_config 必须是 dict 或 None")
        if metadata_config is not None:
            from amc_py.viper.fixed_point import fixed_point_config_to_dict

            if metadata_config != fixed_point_config_to_dict(self.fixed_point_config):
                raise ValueError("metadata.fixed_point_config 与 fixed_point_config 不一致")
        metadata_hash = self.metadata.get("fixed_point_config_hash")
        if metadata_hash is not None and str(metadata_hash) != config_hash:
            raise ValueError("metadata.fixed_point_config_hash 与 fixed_point_config 不一致")
        tree_metadata_hash = self.metadata.get("tree_fixed_point_config_hash")
        if tree_metadata_hash is not None and str(tree_metadata_hash) != config_hash:
            raise ValueError("metadata.tree_fixed_point_config_hash 与 fixed_point_config 不一致")

    def _evaluate(self, state_vector: tuple[float, ...]):
        if len(state_vector) != self.model.state_dim:
            raise ValueError("state_vector 维度不匹配")
        state_int = quantize_state_vector(state_vector, self.fixed_point_config)
        return state_int, evaluate_integer_tree(self.model, state_int)

    def predict_action_ranking(self, state_vector: tuple[float, ...]) -> tuple[int, ...]:
        return self._evaluate(state_vector)[1].action_ranking

    def trace_decision_path(self, state_vector: tuple[float, ...]) -> dict[str, object]:
        state_int, evaluation = self._evaluate(state_vector)
        leaf = next(leaf for leaf in self.model.leaves if leaf.node_id == evaluation.leaf_id)
        total_count = float(sum(float(value) for value in leaf.full_action_counts))
        if total_count > 0.0:
            action_proba = [float(value) / total_count for value in leaf.full_action_counts]
        else:
            action_proba = [0.0 for _ in leaf.full_action_counts]
        return {
            "tree_leaf_id": evaluation.leaf_id,
            "tree_path_node_ids": evaluation.path_node_ids,
            "tree_path_depth": len(evaluation.path_node_ids) - 1,
            "tree_path_predicates": list(evaluation.path_predicates),
            "tree_leaf_impurity": leaf.impurity,
            "tree_leaf_n_node_samples": leaf.n_node_samples,
            "tree_leaf_weighted_n_node_samples": leaf.weighted_n_node_samples,
            "tree_leaf_value": list(leaf.full_action_counts),
            "tree_leaf_predicted_class_id": leaf.raw_action_id,
            "tree_action_proba": action_proba,
            "tree_action_ranking": evaluation.action_ranking,
            "student_state_vector_int": state_int,
            "tree_runtime_policy_type": "integer_tree_ranked_valid_or_none",
        }

    def select_action_id(
        self,
        state_vector: tuple[float, ...],
        valid_action_mask: tuple[bool, ...] | None,
        *,
        include_decision_trace: bool = False,
    ) -> tuple[int | None, dict[str, object]]:
        state_int, evaluation = self._evaluate(state_vector)
        ranking = evaluation.action_ranking
        raw_top1 = ranking[0]
        trace = self.trace_decision_path(state_vector) if include_decision_trace else None
        base = {
            "tree_raw_top1_action_id": raw_top1,
            "tree_raw_top1_invalid": False,
            "tree_fallback_used": False,
            "tree_no_valid_action": False,
            "tree_selected_action_id": raw_top1,
            "tree_selected_rank": 0,
            "student_state_vector_int": state_int,
            "tree_runtime_policy_type": "integer_tree_ranked_valid_or_none",
        }
        if valid_action_mask is None:
            if trace is not None:
                base.update(trace)
            return raw_top1, base
        if len(valid_action_mask) != len(self.action_definitions):
            raise ValueError("valid_action_mask 长度必须与 action_dim 一致")
        raw_invalid = not bool(valid_action_mask[raw_top1])
        base["tree_raw_top1_invalid"] = raw_invalid
        for rank, candidate in enumerate(ranking):
            if bool(valid_action_mask[candidate]):
                base.update({
                    "tree_fallback_used": bool(raw_invalid and rank > 0),
                    "tree_selected_action_id": candidate,
                    "tree_selected_rank": rank,
                })
                if trace is not None:
                    base.update(trace)
                return candidate, base
        explicit_noop_action_id = _explicit_noop_action_id(self.action_definitions)
        if explicit_noop_action_id is not None:
            base.update({
                "tree_fallback_used": True,
                "tree_no_valid_action": True,
                "tree_selected_action_id": explicit_noop_action_id,
                "tree_selected_rank": None,
                "tree_defensive_fallback_to_explicit_noop": True,
            })
            if trace is not None:
                base.update(trace)
            return explicit_noop_action_id, base
        base.update({"tree_no_valid_action": True, "tree_selected_action_id": None, "tree_selected_rank": None})
        if trace is not None:
            base.update(trace)
        return None, base

    def action_definition(self, action_id: int | None) -> dict[str, object] | None:
        if action_id is None or action_id < 0 or action_id >= len(self.action_definitions):
            return None
        return dict(self.action_definitions[action_id])
