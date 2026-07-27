"""Auditable subprocess execution for ordinary proof/HOUT commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_dir: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, object]:
    if not command:
        raise ValueError("command 不得为空")
    log_dir.mkdir(parents=True, exist_ok=True)
    resolved_env = os.environ.copy()
    resolved_env.update({str(key): str(value) for key, value in (env or {}).items()})
    index = len(list(log_dir.glob("command_*.json")))
    stdout_path = log_dir / f"command_{index:03d}.stdout.log"
    stderr_path = log_dir / f"command_{index:03d}.stderr.log"
    timed_out = False
    try:
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=str(Path(cwd).resolve()),
            env=resolved_env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
        returncode = None
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    traceback_marker = "Traceback (most recent call last)" in stderr
    receipt = {
        "command": [str(item) for item in command],
        "cwd": str(Path(cwd).resolve()),
        "returncode": returncode,
        "execution_status": "TIMED_OUT" if timed_out else "COMPLETED",
        "timed_out": timed_out,
        "traceback_marker": traceback_marker,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "environment_overrides": dict(env or {}),
    }
    (log_dir / f"command_{index:03d}.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def ordinary_prove_command(
    *,
    seed_dir: Path,
    tree_variant: str,
    source_root: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "formal_toolchain.cli.prove_seed",
        "--seed-dir",
        str(Path(seed_dir).resolve()),
        "--tree-variant",
        tree_variant,
        "--code-root",
        str(Path(source_root).resolve()),
        "--proof-route",
        "protected_prefix",
        "--out",
        str(Path(output_dir).resolve()),
        "--overwrite",
    ]


def ordinary_verify_command(
    *,
    request: Path,
    bundle: Path,
    output_dir: Path,
    source_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "formal_toolchain.cli.verify_bundle",
        "--request",
        str(Path(request).resolve()),
        "--bundle",
        str(Path(bundle).resolve()),
        "--out",
        str(Path(output_dir).resolve()),
        "--source-root",
        str(Path(source_root).resolve()),
    ]
