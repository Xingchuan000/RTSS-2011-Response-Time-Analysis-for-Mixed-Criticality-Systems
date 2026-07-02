"""决策树策略封装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class TreeBudgetPolicy:
    """对 sklearn 决策树做一层 mask-aware runtime 封装。"""

    classifier: Any
    metadata: dict[str, object]
    feature_names: tuple[str, ...]
    action_definitions: list[dict[str, object]]

    def predict_action_ranking(self, state_vector: tuple[float, ...]) -> tuple[int, ...]:
        """返回完整动作编号排序。

        说明：
        - sklearn 的 `predict_proba` 只覆盖训练时见过的 classes_；
        - 因此这里先给所有 action 初始化为 0 概率，再回填 seen classes；
        - 同概率时按 action_id 升序，保证 artifact 在重复加载后仍完全确定。
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
        ranking = sorted(range(action_dim), key=lambda action_id: (-proba[action_id], action_id))
        return tuple(int(action_id) for action_id in ranking)

    def select_action_id(
        self,
        state_vector: tuple[float, ...],
        valid_action_mask: tuple[bool, ...] | None,
    ) -> tuple[int | None, dict[str, object]]:
        """在 tree 原始排序上叠加当前 runtime mask。"""

        ranking = self.predict_action_ranking(state_vector)
        raw_top1 = ranking[0] if ranking else None
        if valid_action_mask is None:
            return raw_top1, {
                "tree_raw_top1_action_id": raw_top1,
                "tree_raw_top1_invalid": False,
                "tree_fallback_used": False,
                "tree_no_valid_action": raw_top1 is None,
                "tree_selected_action_id": raw_top1,
            }
        if len(valid_action_mask) != len(self.action_definitions):
            raise ValueError("valid_action_mask 长度必须与 action_dim 一致")
        raw_invalid = raw_top1 is not None and not bool(valid_action_mask[raw_top1])
        for candidate in ranking:
            if bool(valid_action_mask[candidate]):
                return int(candidate), {
                    "tree_raw_top1_action_id": raw_top1,
                    "tree_raw_top1_invalid": bool(raw_invalid),
                    "tree_fallback_used": bool(raw_invalid and candidate != raw_top1),
                    "tree_no_valid_action": False,
                    "tree_selected_action_id": int(candidate),
                }
        return None, {
            "tree_raw_top1_action_id": raw_top1,
            "tree_raw_top1_invalid": bool(raw_invalid),
            "tree_fallback_used": False,
            "tree_no_valid_action": True,
            "tree_selected_action_id": None,
        }
