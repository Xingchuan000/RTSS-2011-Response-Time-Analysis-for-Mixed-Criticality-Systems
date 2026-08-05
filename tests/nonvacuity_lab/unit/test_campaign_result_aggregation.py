from __future__ import annotations

from pathlib import Path

from nonvacuity_lab.runners import campaign as campaign_runner
from nonvacuity_lab.schema import CampaignConfig


def test_campaign_aggregates_baseline_result_with_null_semantic_section(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    seed = tmp_path / "seed"
    source.mkdir()
    seed.mkdir()
    config = CampaignConfig.from_mapping(
        {
            "schema_version": "ppp_nonvacuity_campaign_v1",
            "enabled": True,
            "campaign_id": "baseline_only_smoke",
            "proof_route": "protected_prefix",
            "source_root": str(source),
            "output_root": str(tmp_path / "out"),
            "run_baselines": True,
            "run_semantic_recompile": False,
            "run_integrity_reuse": False,
            "run_hout": False,
            "mutations": [
                {
                    "schema_version": "nonvacuity_mutation_v1",
                    "enabled": True,
                    "mutation_id": "P0",
                    "mutation_class": "BASELINE",
                    "seed_dir": str(seed),
                    "tree_variant": "best_overall",
                    "mutator": {"kind": "identity", "parameters": {}},
                    "activation": {"mode": "none"},
                }
            ],
        },
        base_dir=tmp_path,
    )
    monkeypatch.setattr(
        campaign_runner,
        "audit_campaign",
        lambda _config: {"status": "PASS", "issues": []},
    )
    monkeypatch.setattr(
        campaign_runner,
        "run_one",
        lambda *args, **kwargs: {
            "schema_version": "nonvacuity_experiment_result_v1",
            "mutation_id": "P0",
            "status": "PASS_EXPECTED",
            "baseline": {"status": "DEPLOYED_TREE_PROVED"},
            "semantic_recompile": None,
            "integrity_reuse": None,
        },
    )

    result = campaign_runner.run_campaign(config, enabled_by_cli=True)

    assert result["status"] == "COMPLETED"
    assert result["mutation_results"][0]["mutation_id"] == "P0"
    assert (config.output_root / config.campaign_id / "campaign_result.json").is_file()
