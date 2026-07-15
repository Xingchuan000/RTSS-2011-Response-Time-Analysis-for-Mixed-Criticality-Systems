"""测试夹具 target factory；实际定义与 examples fixture 共用同一明确对象。"""

from dataclasses import replace
from examples.formal.synthetic_p0.target import build_target as _build_target

def build_target(**kwargs):
    target = _build_target(**kwargs)
    # fixture artifact 的 action schema 只保留计划要求的稳定字段；target 与
    # artifact 的比较因此是跨来源的精确比较，而不是 target 自比较。
    actions = tuple({key: value for key, value in action.items() if key != "is_noop"}
                    for action in target.action_definitions)
    return replace(target, action_definitions=actions)


__all__ = ["build_target"]
