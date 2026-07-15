"""idle recovery 的单独绑定入口，避免 recovery 语义被 removal checker 遗漏。"""

from __future__ import annotations

from pathlib import Path

from .python_ast_ir import function_to_ir


def bind_recovery_runtime(source_root: Path) -> dict[str, object]:
    source = (Path(source_root) / "amc_py/event_runtime.py").read_text(encoding="utf-8")
    ir = function_to_ir(source, "_maybe_recover_to_lo")
    semantic_ok = "state.mode is SystemMode.HI and not state.active_jobs and state.running_job is None" in source
    status = "PASS" if ir.get("status") == "PASS" and semantic_ok else ("FAIL" if ir.get("status") == "PASS" else "UNRESOLVED")
    return {"status": status,
            "target": ir, "contract": "recovery iff active/running are empty"}
