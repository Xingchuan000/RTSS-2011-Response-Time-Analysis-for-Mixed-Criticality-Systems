from formal_toolchain.bridge.p0_case_manifest import (
    CaseSourceKind,
    controller_p0_cases,
    plant_p0_cases,
)
from formal_toolchain.bridge.runtime_branch_map import PATH_SPECS
from formal_toolchain.bridge.transition_cases import (
    REQUIRED_CONTROLLER_CASE_IDS,
    REQUIRED_PLANT_P0_CASE_IDS,
)


def test_controller_cases_not_in_plant_runtime_case_set() -> None:
    assert "CONTROLLER_NO_ACTION" not in REQUIRED_PLANT_P0_CASE_IDS
    assert "CONTROLLER_SELECTED_ACTION" not in REQUIRED_PLANT_P0_CASE_IDS
    assert {row["case_id"] for row in plant_p0_cases()} == set(REQUIRED_PLANT_P0_CASE_IDS)


def test_controller_case_set_complete() -> None:
    assert set(REQUIRED_CONTROLLER_CASE_IDS) == {
        "CONTROLLER_NO_ACTION",
        "CONTROLLER_SELECTED_ACTION",
    }
    assert {row["case_id"] for row in controller_p0_cases()} == set(REQUIRED_CONTROLLER_CASE_IDS)
    assert all(row["source_kind"] == CaseSourceKind.CONTROLLER_SYNCHRONOUS for row in controller_p0_cases())


def test_controller_cases_are_not_runtime_branch_paths() -> None:
    path_cases = {spec[2] for spec in PATH_SPECS}
    assert path_cases.isdisjoint(REQUIRED_CONTROLLER_CASE_IDS)
    assert all(
        row["source_kind"] == CaseSourceKind.PLANT_RUNTIME_BRANCH
        for row in plant_p0_cases()
    )
