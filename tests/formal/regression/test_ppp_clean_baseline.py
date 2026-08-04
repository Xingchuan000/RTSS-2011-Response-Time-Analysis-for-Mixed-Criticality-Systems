from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from formal_toolchain.workflow.prove_seed import prove_seed
from tests.formal.regression.ppp_baseline_helpers import extract_ppp_baseline

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "formal" / "fixtures" / "synthetic_p0"
EXPECTED = ROOT / "tests" / "formal" / "fixtures" / "ppp_clean_baseline" / "synthetic_expected.json"
CLI_SURFACE = EXPECTED.with_name("cli_surface.json")


def test_ppp_cli_surface_contract():
    contract = json.loads(CLI_SURFACE.read_text(encoding="utf-8"))
    completed = subprocess.run(
        [sys.executable, "-m", "formal_toolchain.cli.prove_seed", "--help"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    for option in contract["required_after_phase1"]:
        assert option in completed.stdout
    for option in contract["forbidden_after_phase1"]:
        assert option not in completed.stdout


def test_ppp_baseline_fixture_contract():
    snapshot = json.loads(EXPECTED.read_text(encoding="utf-8"))
    allowed = json.loads(CLI_SURFACE.with_name("request_allowed_keys.json").read_text(encoding="utf-8"))
    assert snapshot["schema_version"] == "ppp_clean_baseline_v1"
    assert set(snapshot["request_keys"]) <= set(allowed["allowed_keys"])
    assert not any(key.startswith(tuple(allowed["forbidden_prefixes"])) for key in snapshot["request_keys"])


@pytest.mark.real_data
def test_optional_real_ppp_baseline(tmp_path: Path):
    raw = os.environ.get("PPP_REAL_BASELINE_SEED_DIR")
    if not raw:
        pytest.skip("PPP_REAL_BASELINE_SEED_DIR 未配置")
    code, result = prove_seed(
        seed_dir=Path(raw), tree_variant=os.environ.get("PPP_REAL_BASELINE_TREE_VARIANT", "best_overall"),
        code_root=ROOT, out=tmp_path / "real_bundle", overwrite=False,
        refresh_phase_k_map=True, proof_route="protected_prefix",
    )
    assert code in {0, 10, 11, 12, 13, 14, 20, 30}
    assert result["primary_claim"] == "DEPLOYED_HI_SAFETY"
