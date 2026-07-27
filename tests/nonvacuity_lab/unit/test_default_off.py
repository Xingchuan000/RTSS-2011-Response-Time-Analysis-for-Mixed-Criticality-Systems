from __future__ import annotations

import json
from pathlib import Path

from nonvacuity_lab.manifest import load_campaign
from nonvacuity_lab.runners.campaign import run_campaign


def test_missing_enabled_defaults_false_and_creates_nothing(tmp_path: Path):
    source = tmp_path / "repo"
    source.mkdir()
    config_path = tmp_path / "campaign.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "ppp_nonvacuity_campaign_v1",
                "campaign_id": "disabled",
                "proof_route": "protected_prefix",
                "output_root": str(tmp_path / "out"),
                "source_root": str(source),
                "mutations": [],
            }
        ),
        encoding="utf-8",
    )
    config = load_campaign(config_path)
    result = run_campaign(config, enabled_by_cli=True)
    assert config.enabled is False
    assert result["status"] == "EXPERIMENT_DISABLED"
    assert not (tmp_path / "out").exists()


def test_cli_enable_is_a_second_gate(tmp_path: Path):
    source = tmp_path / "repo"
    source.mkdir()
    config_path = tmp_path / "campaign.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "ppp_nonvacuity_campaign_v1",
                "enabled": True,
                "campaign_id": "disabled_by_cli",
                "proof_route": "protected_prefix",
                "output_root": str(tmp_path / "out"),
                "source_root": str(source),
                "mutations": [],
            }
        ),
        encoding="utf-8",
    )
    result = run_campaign(load_campaign(config_path), enabled_by_cli=False)
    assert result["status"] == "EXPERIMENT_DISABLED"
    assert not (tmp_path / "out").exists()
