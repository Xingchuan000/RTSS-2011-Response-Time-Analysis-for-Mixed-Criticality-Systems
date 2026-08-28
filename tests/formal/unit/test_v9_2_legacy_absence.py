from pathlib import Path

from formal_toolchain.adapters.source_manifest import FORMAL_TARGET_FILES

ROOT = Path(__file__).parents[3]


def test_retired_route_stacks_are_physically_absent():
    retired = (
        "formal_toolchain/routes",
        "formal_toolchain/compiler",
        "formal_toolchain/verifier",
        "formal_toolchain/reference/protected_priority_prefix",
        "formal_toolchain/workflow/prove_seed_v8.py",
        "formal_toolchain/workflow/seed_workspace.py",
        "configs/nonvacuity",
        "nonvacuity_lab",
    )
    for relative in retired:
        assert not (ROOT / relative).exists(), relative


def test_retired_route_registries_and_ppp_scripts_are_absent():
    retired = (
        "formal_toolchain/specs/routes/protected_prefix_registry.json",
        "formal_toolchain/specs/routes/raw_protected_prefix_registry.json",
        "formal_toolchain/specs/routes/strict_full_registry.json",
        "formal_toolchain/specs/certificates/protected_prefix_taskset.schema.json",
        "formal_toolchain/specs/certificates/protected_prefix_partition.schema.json",
        "scripts/run_nonvacuity_hout.py",
        "scripts/prepare_ppp_nonvacuity_hout.py",
    )
    for relative in retired:
        assert not (ROOT / relative).exists(), relative


def test_active_source_manifest_points_at_v9_2_terminal_components():
    required = {
        "formal_toolchain/v9_2/compiler.py",
        "formal_toolchain/v9_2/verifier.py",
        "formal_toolchain/v9_2/event_kernel.py",
        "formal_toolchain/v9_2/event_window_encoder.py",
        "formal_toolchain/v9_2/event_refinement.py",
    }
    assert required <= set(FORMAL_TARGET_FILES)
    assert all("/v9_2/" in row or "/binding/" in row or "/bridge/" in row for row in FORMAL_TARGET_FILES)


def test_packaging_surface_contains_only_active_formal_package():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["amc_py*", "formal_toolchain*"]' in text
    assert "amc-nonvacuity-lab" not in text
