from __future__ import annotations

from formal_toolchain.core.artifact import obligation_certificate


def build_early_stop_configuration_gate(
    *, runtime_config, context_hash, closure_completion_certificate=None,
):
    enabled = bool(getattr(runtime_config, "stop_at_first_miss", False))
    if not enabled:
        return obligation_certificate(
            obligation_id="EARLY_STOP_CONFIGURATION_GATE",
            status="PASS",
            context_hash=context_hash,
            inputs={"stop_at_first_miss": False},
            witness={
                "mode": "DISABLED_IN_P0",
                "closure_completion_required": False,
            },
            direct_predecessor_hashes={},
            checker_id=__name__,
            checker_version="early-stop-gate-v1",
        )

    if closure_completion_certificate is None or closure_completion_certificate.get("obligation_status") != "PASS":
        return obligation_certificate(
            obligation_id="EARLY_STOP_CONFIGURATION_GATE",
            status="UNRESOLVED",
            context_hash=context_hash,
            inputs={"stop_at_first_miss": True},
            witness={
                "mode": "ENABLED_REQUIRES_CLOSURE_COMPLETION",
                "closure_completion_required": True,
            },
            direct_predecessor_hashes={},
            checker_id=__name__,
            checker_version="early-stop-gate-v1",
            failure={
                "route": "MODEL_CONFORMANCE_FAILED",
                "code": "EARLY_STOP_CLOSURE_COMPLETION_MISSING",
            },
        )

    return obligation_certificate(
        obligation_id="EARLY_STOP_CONFIGURATION_GATE",
        status="PASS",
        context_hash=context_hash,
        inputs={"stop_at_first_miss": True},
        witness={
            "mode": "ENABLED",
            "closure_completion_required": True,
        },
        direct_predecessor_hashes={
            "EARLY_STOP_CLOSURE_COMPLETION": closure_completion_certificate["artifact_hash"]
        },
        checker_id=__name__,
        checker_version="early-stop-gate-v1",
    )
