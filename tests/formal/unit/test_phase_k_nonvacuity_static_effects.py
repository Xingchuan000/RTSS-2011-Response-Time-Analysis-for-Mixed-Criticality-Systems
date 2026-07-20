from types import SimpleNamespace

import pytest

from formal_toolchain.bridge.effect_compiler import (
    build_phase_k_static_effect_bindings,
    compile_effect_ir,
)
from formal_toolchain.bridge.model_bounds import _legacy_test_bounds


_HELPER_SOURCE = (
    "_apply_retroactive_release_budget_mutation(state=self.state, "
    "updates=update_payload, cfg=self.config)"
)
_HELPER_HASH = "b" * 64


def _effect():
    return [{
        "kind": "CALL",
        "source": _HELPER_SOURCE,
        "ast_hash": _HELPER_HASH,
    }]


def test_off_profile_retroactive_helper_is_statically_elided():
    cfg = SimpleNamespace(nonvacuity_profile="off")
    bindings = build_phase_k_static_effect_bindings(cfg)
    compiled = compile_effect_ir(
        _effect(),
        bounds=_legacy_test_bounds(),
        static_effect_bindings=bindings,
    )

    assert bindings == {"retroactive_release_budget_mutation": False}
    assert compiled.consumed_effect_hashes == (_HELPER_HASH,)
    assert _HELPER_HASH in compiled.non_state_effect_hashes
    assert all("runtime_budget_at_release" not in equation
               for equation in compiled.equations)
    # Static no-op still produces a complete framed post-state.
    assert any("c_affected_job_budget_post" in equation
               for equation in compiled.equations)


def test_exported_effective_config_shape_is_supported():
    cfg = {
        "schema_version": "effective_runtime_config_v1",
        "fields": {
            "nonvacuity_profile": {"value": "off"},
        },
    }
    assert build_phase_k_static_effect_bindings(cfg) == {
        "retroactive_release_budget_mutation": False,
    }


def test_c3_profile_fails_closed_if_phase_k_is_reached():
    cfg = SimpleNamespace(
        nonvacuity_profile="c3_retroactive_release_budget"
    )
    bindings = build_phase_k_static_effect_bindings(cfg)
    assert bindings == {"retroactive_release_budget_mutation": True}

    with pytest.raises(
        ValueError,
        match=(
            "NONVACUITY_C3_RETROACTIVE_EFFECT_REQUIRES_"
            "POLICY_CONTRACT_REJECTION"
        ),
    ):
        compile_effect_ir(
            _effect(),
            bounds=_legacy_test_bounds(),
            static_effect_bindings=bindings,
        )


def test_legacy_direct_calls_default_to_safe_off():
    compiled = compile_effect_ir(_effect(), bounds=_legacy_test_bounds())
    assert compiled.consumed_effect_hashes == (_HELPER_HASH,)
    assert _HELPER_HASH in compiled.non_state_effect_hashes
