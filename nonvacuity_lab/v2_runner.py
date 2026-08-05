from __future__ import annotations

import json
from pathlib import Path

from .config_io import validate_config_kind, verify_config_hash
from .schema import experiment_envelope
from .schema import CampaignConfig
from .runners.campaign import run_campaign


def run_v2_campaign(
    config_path: Path,
    *,
    cli_enable: bool,
    doctor_receipt: Path | None = None,
    timeout_seconds: int | None = None,
    overwrite_existing: bool = False,
) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_config_kind(config)
    verify_config_hash(config)
    decisions = []
    enabled = []
    for mutation in config.get("mutations", []):
        if config.get("config_kind") != "RESOLVED":
            allowed, reason = False, "CONFIG_NOT_RESOLVED"
        elif not config.get("enabled", False):
            allowed, reason = False, "CAMPAIGN_DISABLED"
        elif not cli_enable:
            allowed, reason = False, "CLI_ENABLE_MISSING"
        elif not mutation.get("enabled", False):
            allowed, reason = False, "MUTATION_DISABLED"
        else:
            allowed, reason = True, "ENABLED"
        decisions.append({"mutation_id": mutation.get("mutation_id"), "allowed": allowed, "reason": reason})
        if allowed:
            enabled.append(mutation)
    if not enabled:
        return {"schema_version": "nonvacuity_campaign_run_v2", "status": "DISABLED", "decisions": decisions}
    if doctor_receipt is None:
        return {"schema_version": "nonvacuity_campaign_run_v2", "status": "SETUP_INVALID", "reason": "doctor receipt missing", "decisions": decisions}
    receipt = json.loads(Path(doctor_receipt).read_text(encoding="utf-8"))
    if receipt.get("overall_status") != "PASS" or receipt.get("config_sha256") != config.get("config_sha256"):
        return {"schema_version": "nonvacuity_campaign_run_v2", "status": "SETUP_INVALID", "reason": "doctor receipt does not bind PASS config", "decisions": decisions}
    # Translate only the outer v2 envelope to the already isolated v1
    # execution model.  The ordinary prove/verify commands still receive
    # their normal request and never see these experiment fields.
    raw_v1 = {
        "schema_version": "ppp_nonvacuity_campaign_v1",
        "enabled": True,
        "campaign_id": config["campaign_id"],
        "proof_route": "protected_prefix",
        "output_root": config.get("output_roots", {}).get("nonvacuity_lab", config.get("output_root", "outputs/nonvacuity_lab")),
        "source_root": config["source_binding"]["clean_source_root"],
        "preserve_workspaces": True,
        "run_baselines": True,
        "run_semantic_recompile": True,
        "run_integrity_reuse": True,
        "run_hout": True,
        "fail_on_not_activated": True,
        "mutations": [
            _v2_mutation_to_v1(
                item,
                config=config,
                base_dir=Path(config_path).resolve().parent,
            )
            for item in enabled
        ],
    }
    campaign = CampaignConfig.from_mapping(raw_v1, base_dir=Path(config_path).resolve().parent)
    result = run_campaign(
        campaign,
        enabled_by_cli=True,
        timeout_seconds=timeout_seconds,
        overwrite_existing=overwrite_existing,
    )
    result["v2_decisions"] = decisions
    return result


def _v2_mutation_to_v1(
    mutation: dict,
    *,
    config: dict | None = None,
    base_dir: Path | None = None,
) -> dict:
    target = dict(mutation.get("resolved_target", {}))
    mutator = dict(mutation.get("mutator", {}))
    parameters = dict(mutator.get("parameters", {}))
    parameters.update(
        {
            key: target[key]
            for key in ("leaf_id", "action_id", "tree_sha256", "tree_path")
            if key in target
        }
    )
    mutation_class = str(mutation.get("mutation_class", ""))
    if mutation_class == "ENVELOPE_GRADIENT":
        mutation_class = "ENVELOPE"
    if mutation_class.startswith("BUNDLE_") or mutation_class == "SOURCE_BINDING_TAMPER":
        mutation_class = "BUNDLE_INTEGRITY"

    expected = dict(mutation.get("expected", {}))
    allowed_statuses = tuple(str(item) for item in expected.get("allowed_result_statuses", ()))
    legacy_status = expected.get("result_status")
    if not allowed_statuses and legacy_status:
        allowed_statuses = (str(legacy_status),)
    allowed_obligations = tuple(
        str(item)
        for item in expected.get(
            "allowed_first_failing_obligations",
            expected.get("first_failing_obligations", ()),
        )
    )
    expected_v1 = {
        "allowed_result_statuses": list(allowed_statuses),
        "allowed_first_failing_obligations": list(allowed_obligations),
        "allowed_failure_routes": list(expected.get("allowed_failure_routes", ())),
        "require_failure": bool(expected.get("require_failure", False)),
        "require_proved": bool(expected.get("require_proved", False)),
        "require_activation": bool(expected.get("require_activation", True)),
        "integrity_result_status": expected.get(
            "integrity_result_status", "PROOF_BUNDLE_INVALID"
        ),
    }
    # Keep the legacy scalar only when it is unambiguous.  The v1 execution
    # model now understands allowed_result_statuses directly.
    if len(allowed_statuses) == 1:
        expected_v1["result_status"] = allowed_statuses[0]
    if allowed_obligations:
        expected_v1["first_failing_obligations"] = list(allowed_obligations)

    metadata = dict(mutation.get("metadata", {}))
    if target:
        # Phase-6 activation consumes the exact resolved tree/leaf/action
        # binding from metadata.  Keep one immutable copy instead of relying
        # on mutator parameters, which have a different semantic purpose.
        metadata["resolved_target"] = dict(target)
    profile_id = mutation.get("hout_profile_id") or metadata.get("hout_profile_id")
    if profile_id is not None:
        if not isinstance(config, dict):
            raise ValueError("HOUT profile resolution requires the v2 campaign config")
        profiles = config.get("hout_profiles", {})
        profile = profiles.get(str(profile_id)) if isinstance(profiles, dict) else None
        if not isinstance(profile, dict):
            raise ValueError(f"HOUT_PROFILE_NOT_FOUND:{profile_id}")
        metadata["hout_profile_id"] = str(profile_id)
        metadata["hout"] = _resolve_hout_profile_paths(
            {"profile_id": str(profile_id), **profile},
            base_dir=base_dir or Path.cwd(),
        )

    activation_v1 = dict(mutation.get("activation", {}))
    activation_mode = str(activation_v1.get("mode", "")).lower()
    if "hout" in activation_mode:
        if target.get("leaf_id") is not None:
            activation_v1.setdefault("required_leaf_id", int(target["leaf_id"]))
        if target.get("action_id") is not None:
            activation_v1.setdefault("required_action_id", int(target["action_id"]))
        original_class = str(mutation.get("mutation_class", ""))
        if original_class == "DANGEROUS_TOP1":
            activation_v1.setdefault("require_mutated_reject", True)
        elif original_class == "MASK_BYPASS":
            activation_v1.setdefault("require_baseline_reject", True)
            activation_v1.setdefault("require_selected_after_mutation", True)
        elif original_class == "ACTION_SEMANTICS":
            activation_v1.setdefault("require_any_budget_difference", True)
        elif original_class == "RETROACTIVE_RELEASE_BUDGET":
            activation_v1.setdefault("require_active_release_budget_difference", True)

    seed_dir = mutation.get("seed_dir") or target.get("seed_dir")
    return {
        "schema_version": "nonvacuity_mutation_v1",
        "enabled": True,
        "mutation_id": str(mutation["mutation_id"]),
        "mutation_class": mutation_class,
        "seed_dir": seed_dir,
        "base_seed": mutation.get("seed"),
        "tree_variant": mutation.get("tree_variant", "best_overall"),
        "paired_with": mutation.get("pair_with"),
        "single_semantic_change": True,
        "mutator": {
            "kind": mutator.get("kind", mutation_class.lower()),
            "parameters": parameters,
        },
        "activation": activation_v1,
        "expected": expected_v1,
        "reuse_source_bundle": mutation.get("reuse_source_bundle"),
        "metadata": metadata,
    }


def _resolve_hout_profile_paths(profile: dict, *, base_dir: Path) -> dict:
    result = dict(profile)
    for key in ("taskset_path", "runtime_config_path", "demand_trace_path"):
        value = result.get(key)
        if value in (None, ""):
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        result[key] = str(path)
    return result
