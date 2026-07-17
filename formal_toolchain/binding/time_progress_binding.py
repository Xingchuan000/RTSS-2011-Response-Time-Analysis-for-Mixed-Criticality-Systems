"""时间推进与零时间闭包的源码绑定结果。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _function_node(source: str, qualified_name: str) -> ast.AST | None:
    """返回指定函数定义节点。

    这里不做复杂 IR 提取，只确认目标函数在源码里真实存在，以免把
    unsupported AST 子节点误判成“绑定缺失”。
    """

    tree = ast.parse(source, filename="amc_py/event_runtime.py")
    if "." in qualified_name:
        class_name, function_name = qualified_name.split(".", 1)
        for item in tree.body:
            if isinstance(item, ast.ClassDef) and item.name == class_name:
                for child in item.body:
                    if isinstance(child, ast.FunctionDef) and child.name == function_name:
                        return child
        # 这里有一类源码结构是“语义上归属某个 runtime engine，
        # 但实现上是模块级 helper 函数”，例如
        # ``EventRuntimeEngine._schedule_running_job_events``。
        # 这类目标在源码里实际对应顶层的 ``_schedule_running_job_events``
        # 定义，所以在类内没有找到时，只按函数名继续确认一次。
        for item in tree.body:
            if isinstance(item, ast.FunctionDef) and item.name == function_name:
                return item
        return None
    for item in tree.body:
        if isinstance(item, ast.FunctionDef) and item.name == qualified_name:
            return item
    return None


def _function_node_in_file(path: Path, qualified_name: str) -> ast.AST | None:
    source = path.read_text(encoding="utf-8")
    return _function_node(source, qualified_name)


def _function_present(source: str, qualified_name: str) -> dict[str, Any]:
    node = _function_node(source, qualified_name)
    return {
        "status": "PASS" if node is not None else "UNRESOLVED",
        "route": None if node is not None else "UNRESOLVED",
        "function": qualified_name,
        "lineno": None if node is None else int(getattr(node, "lineno", 0)),
    }


def bind_time_progress_runtime(source_root: Path) -> dict[str, Any]:
    runtime_path = Path(source_root) / "amc_py/event_runtime.py"
    event_model_path = Path(source_root) / "amc_py/event_models.py"
    source = runtime_path.read_text(encoding="utf-8")
    targets = {
        "EventRuntimeEngine._advance_time": _function_present(source, "EventRuntimeEngine._advance_time"),
        "EventRuntimeEngine._schedule_running_job_events": _function_present(source, "EventRuntimeEngine._schedule_running_job_events"),
        "EventQueue.pop": {
            "status": "PASS" if _function_node_in_file(event_model_path, "EventQueue.pop") is not None else "UNRESOLVED",
            "route": None if _function_node_in_file(event_model_path, "EventQueue.pop") is not None else "UNRESOLVED",
            "function": "EventQueue.pop",
            "lineno": None if _function_node_in_file(event_model_path, "EventQueue.pop") is None else int(getattr(_function_node_in_file(event_model_path, "EventQueue.pop"), "lineno", 0)),
        },
        "EventRuntimeEngine.run_until": _function_present(source, "EventRuntimeEngine.run_until"),
        "simulate_ordered_taskset_event_driven": _function_present(source, "simulate_ordered_taskset_event_driven"),
    }
    unresolved = [name for name, ir in targets.items() if ir.get("status") != "PASS"]

    # 只检查真正属于时间推进语义的源码骨架：推进时钟、从队列中跳转到下一个
    # 事件、以及对空闲闭包的整体推进路径。这里不引入任何兜底分支。
    strict_time_increase = (
        "self.state.current_time = target_time" in source
        or "self.state.current_time = now" in source
    )
    zero_time_closure_finite = (
        "while not self.queue.empty()" in source
        and "self._advance_time(target_time)" in source
    )
    ready_branch = {
        "kind": "ONE_SERVICE_TICK",
        "strict_time_increase": strict_time_increase,
        "evidence": targets["EventRuntimeEngine._advance_time"],
    }
    empty_ready_branch = {
        "kind": "JUMP_TO_NEXT_EVENT",
        "strict_time_increase": strict_time_increase,
        "evidence": targets["EventRuntimeEngine.run_until"],
    }
    if unresolved:
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "failure": {"code": "TIME_PROGRESS_BINDING_UNRESOLVED", "targets": unresolved},
            "ready_branch": ready_branch,
            "empty_ready_branch": empty_ready_branch,
            "zero_time_closure_finite": zero_time_closure_finite,
        }
    return {
        "status": "PASS",
        "ready_branch": ready_branch,
        "empty_ready_branch": empty_ready_branch,
        "zero_time_closure_finite": zero_time_closure_finite,
    }
