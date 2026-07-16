"""顶层工作流的受控 Python 子进程执行器。"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_cli(module: str, args: list[str], *, cwd: Path, log_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", module, *args]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False,
                            env={key: value for key, value in os.environ.items()
                                 if key in {"PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL", "TZ"}})
    ended = datetime.now(timezone.utc).isoformat()
    stdout = result.stdout.encode("utf-8")
    stderr = result.stderr.encode("utf-8")
    digest = lambda value: hashlib.sha256(value).hexdigest()
    log_dir.mkdir(parents=True, exist_ok=True)
    name = module.rsplit(".", 1)[-1]
    (log_dir / f"{name}.stdout.log").write_bytes(stdout)
    (log_dir / f"{name}.stderr.log").write_bytes(stderr)
    return {"module": module, "argv": command, "cwd": ".", "started_at": started,
            "ended_at": ended, "returncode": result.returncode,
            "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr)}
