from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


FORBIDDEN_CLI_TOKENS = {"--nonvacuity-profile", "--nonvacuity", "--mutation-id", "--expected-failure"}


def capture_help(module: str, source_root: Path) -> str:
    completed = subprocess.run([sys.executable, "-m", module, "--help"], cwd=source_root, env={**os.environ, "PYTHONPATH": str(source_root)}, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    return completed.stdout + completed.stderr


def check_ordinary_cli_surface(config: dict):
    from .schema import DoctorCheck
    source_root = Path(config["source_binding"]["clean_source_root"])
    checks = []
    for module in config.get("ordinary_cli_modules", ("formal_toolchain.cli.prove_seed", "formal_toolchain.cli.verify_bundle")):
        try:
            text = capture_help(module, source_root)
            forbidden = sorted(token for token in FORBIDDEN_CLI_TOKENS if token in text)
            checks.append(DoctorCheck.fail("ordinary_cli_clean", "forbidden experiment flags", module=module, tokens=forbidden) if forbidden else DoctorCheck.pass_("ordinary_cli_clean", "ordinary CLI surface is clean", module=module))
        except Exception as exc:
            checks.append(DoctorCheck.fail("ordinary_cli_surface", "ordinary CLI unavailable", module=module, error=repr(exc)))
    return checks
