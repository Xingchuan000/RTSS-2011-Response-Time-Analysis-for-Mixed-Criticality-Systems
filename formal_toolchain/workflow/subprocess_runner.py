"""顶层工作流的受控 Python 子进程执行器。"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# 证明子进程只继承运行所需的最小环境。Windows 下保留创建进程、临时目录和
# conda/Python 运行常用的基础变量；变量名比较使用 upper()，兼容 ``Path``。
_ALLOWED_ENV_KEYS = {
    "PATH",
    "PYTHONPATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "APPDATA",
    "LOCALAPPDATA",
    "LANG",
    "LC_ALL",
    "TZ",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
}


def _controlled_environment() -> dict[str, str]:
    """返回用于 fresh-process verifier 的稳定、最小环境。"""

    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _ALLOWED_ENV_KEYS
    }
    # Python 子进程自身输出统一使用 UTF-8；父进程仍按原始 bytes 捕获，因而即使
    # 第三方原生库写出本地代码页字节，也不会再触发 TextIOWrapper 解码异常。
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _as_bytes(value: bytes | str | None) -> bytes:
    """把 subprocess 输出规范为 bytes，并对异常返回的 None fail-closed。"""

    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="backslashreplace")


def run_cli(module: str, args: list[str], *, cwd: Path, log_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", module, *args]
    started = datetime.now(timezone.utc).isoformat()

    # 不使用 text=True/encoding=...。Windows/conda 环境中，子进程可能同时输出
    # UTF-8、系统代码页或原生库字节；文本模式会在线程读取阶段抛
    # UnicodeDecodeError，并使 CompletedProcess.stdout/stderr 变成 None。
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=_controlled_environment(),
    )
    ended = datetime.now(timezone.utc).isoformat()

    stdout = _as_bytes(result.stdout)
    stderr = _as_bytes(result.stderr)

    def digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    log_dir.mkdir(parents=True, exist_ok=True)
    name = module.rsplit(".", 1)[-1]
    (log_dir / f"{name}.stdout.log").write_bytes(stdout)
    (log_dir / f"{name}.stderr.log").write_bytes(stderr)
    return {
        "module": module,
        "argv": command,
        "cwd": ".",
        "started_at": started,
        "ended_at": ended,
        "returncode": result.returncode,
        "stdout_sha256": digest(stdout),
        "stderr_sha256": digest(stderr),
    }
