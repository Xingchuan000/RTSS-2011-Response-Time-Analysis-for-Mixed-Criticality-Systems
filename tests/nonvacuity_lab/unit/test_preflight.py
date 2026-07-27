import json
from pathlib import Path

from nonvacuity_lab.manifest import load_campaign
from nonvacuity_lab.preflight import audit_campaign
from nonvacuity_lab.runners.campaign import run_campaign


def _write_campaign(
    tmp_path: Path,
    *,
    mutation_class: str,
    mutator: dict,
    seed_dir: Path | None = None,
) -> Path:
    config = {
        "schema_version": "ppp_nonvacuity_campaign_v1",
        "enabled": True,
        "campaign_id": "preflight_test",
        "proof_route": "protected_prefix",
        "output_root": str(tmp_path / "out"),
        "source_root": str(tmp_path / "source"),
        "mutations": [
            {
                "schema_version": "nonvacuity_mutation_v1",
                "enabled": True,
                "mutation_id": "M1",
                "mutation_class": mutation_class,
                "seed_dir": str(seed_dir or tmp_path / "seed"),
                "tree_variant": "best_overall",
                "mutator": mutator,
                "activation": {"mode": "symbolic"},
            }
        ],
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_preflight_rejects_empty_leaf_and_action_targets_without_workspace(tmp_path: Path):
    (tmp_path / "source").mkdir()
    (tmp_path / "seed").mkdir()
    path = _write_campaign(
        tmp_path,
        mutation_class="DANGEROUS_TOP1",
        mutator={
            "kind": "dangerous_top1",
            "parameters": {"leaf_candidates": [], "dangerous_actions": []},
        },
    )
    config = load_campaign(path)
    audit = audit_campaign(config)
    assert audit["status"] == "CAMPAIGN_PREFLIGHT_FAILED"
    assert {item["code"] for item in audit["issues"]} >= {
        "LEAF_TARGET_UNRESOLVED",
        "ACTION_TARGET_UNRESOLVED",
    }
    result = run_campaign(config, enabled_by_cli=True)
    assert result["status"] == "CAMPAIGN_PREFLIGHT_FAILED"
    assert not config.output_root.exists()


def test_preflight_rejects_nonblocking_and_refreshed_semantic_targets(tmp_path: Path):
    source = tmp_path / "source"
    seed = tmp_path / "seed"
    (source / "amc_py" / "rl").mkdir(parents=True)
    seed.mkdir()
    (source / "amc_py" / "rl" / "env.py").write_text(
        "class AmcBudgetEnv:\n    guard = True\n",
        encoding="utf-8",
    )
    path = _write_campaign(
        tmp_path,
        mutation_class="RUNTIME_SOURCE",
        seed_dir=seed,
        mutator={
            "kind": "source_overlay",
            "parameters": {
                "patches": [
                    {
                        "target_file": "amc_py/rl/env.py",
                        "target_symbol": "AmcBudgetEnv",
                        "before_snippet": "guard = True",
                        "after_snippet": "guard = False",
                    }
                ]
            },
        },
    )
    audit = audit_campaign(load_campaign(path))
    assert "NONBLOCKING_SEMANTIC_TARGET" in {
        item["code"] for item in audit["issues"]
    }


def test_campaign_json_schema_is_enforced(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ppp_nonvacuity_campaign_v1",
                "campaign_id": "bad",
                "proof_route": "protected_prefix",
                "output_root": "out",
                "source_root": ".",
                "mutations": [{"schema_version": "nonvacuity_mutation_v1"}],
            }
        ),
        encoding="utf-8",
    )
    try:
        load_campaign(path)
    except ValueError as exc:
        assert "JSON Schema" in str(exc)
    else:
        raise AssertionError("invalid campaign unexpectedly loaded")
