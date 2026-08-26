from formal_toolchain.binding.action_binding import bind_action_runtime
from formal_toolchain.binding.controller_binding import bind_controller_runtime
from formal_toolchain.bridge.closure_cases import build_controller_postclosure_certificate
from formal_toolchain.bridge.controller_transition import build_controller_transition_certificate
from formal_toolchain.core.hashing import sha256_object


def _controller_certificate() -> dict:
    return build_controller_transition_certificate(
        controller_binding=bind_controller_runtime("."),
        action_binding=bind_action_runtime(".", action_dim=25, explicit_noop=True),
        deployed_policy_binding={"status": "PASS", "binding_hash": sha256_object({"p": "deployed"})},
        controller_postclosure_certificate={
            "obligation_status": "PASS",
            "artifact_hash": sha256_object({"controller_postclosure": "verified"}),
        },
        context_hash="0" * 64,
    )


def test_postclosure_consumes_controller_transition_certificate() -> None:
    result = build_controller_postclosure_certificate(
        context_hash="0" * 64,
        controller_transition_certificate=_controller_certificate(),
    )
    assert result["obligation_status"] == "PASS"
    assert result["witness"]["case_ids"] == [
        "CONTROLLER_NO_ACTION",
        "CONTROLLER_SELECTED_ACTION",
    ]


def test_postclosure_without_controller_certificate_is_unresolved() -> None:
    result = build_controller_postclosure_certificate(context_hash="0" * 64)
    assert result["obligation_status"] == "UNRESOLVED"
    assert result["failure"]["code"] == "CONTROLLER_TRANSITION_CERTIFICATE_REQUIRED"
