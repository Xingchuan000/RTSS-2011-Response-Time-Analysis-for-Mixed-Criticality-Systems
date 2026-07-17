from __future__ import annotations

import hashlib
from pathlib import Path

from formal_toolchain.workflow.subprocess_runner import _as_bytes, run_cli


def test_run_cli_preserves_non_utf8_output_as_raw_bytes(tmp_path: Path) -> None:
    module = tmp_path / "emit_mixed_bytes.py"
    stdout = b"valid-utf8:\xe4\xb8\xad\xe6\x96\x87\ninvalid:\xce\xff\n"
    stderr = b"stderr-local-codepage:\xce\xc4\xbc\xfe\n"
    module.write_text(
        "from __future__ import annotations\n"
        "import os\n"
        "import sys\n"
        f"os.write(sys.stdout.fileno(), {stdout!r})\n"
        f"os.write(sys.stderr.fileno(), {stderr!r})\n",
        encoding="utf-8",
    )

    log_dir = tmp_path / "logs"
    result = run_cli("emit_mixed_bytes", [], cwd=tmp_path, log_dir=log_dir)

    assert result["returncode"] == 0
    assert (log_dir / "emit_mixed_bytes.stdout.log").read_bytes() == stdout
    assert (log_dir / "emit_mixed_bytes.stderr.log").read_bytes() == stderr
    assert result["stdout_sha256"] == hashlib.sha256(stdout).hexdigest()
    assert result["stderr_sha256"] == hashlib.sha256(stderr).hexdigest()


def test_as_bytes_handles_none_and_text_defensively() -> None:
    assert _as_bytes(None) == b""
    assert _as_bytes(b"raw") == b"raw"
    assert _as_bytes("中文") == "中文".encode("utf-8")
