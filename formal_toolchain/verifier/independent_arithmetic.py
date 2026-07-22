"""verifier 的独立整数算术入口。

该模块不调用 ``formal_checks`` 或 compiler。production 证书由 candidate
提供，verifier 只从 reference taskset 重新计算 replay 并逐字段比对。
"""

from typing import Any, Mapping

from formal_toolchain.reference.rta_replay import replay_all_task_rta


def replay_protected_hi_rta(reference_taskset: Any,
                            production: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """独立重放 Protected-HI RTA。

    ``production`` 只作为 replay 所需的参考 witness，不被用于生成新的
    recurrence 结果；replay 算法本身来自 reference/rta_replay。
    """

    if production is None:
        production = {}
    return replay_all_task_rta(reference_taskset, production)

__all__ = ["replay_protected_hi_rta"]
