"""s185 target recipe 占位。

计划要求的权威 taskset/priority/effective config 尚未随当前 workspace 提供；
此 factory 明确失败闭合，禁止把当前默认重建结果冒充训练时 target。
"""


def build_target(**_kwargs):
    raise RuntimeError("AUTHORITATIVE_TARGET_MISSING: s185 target recipe 尚未提供")
