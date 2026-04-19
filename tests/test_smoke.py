import amc_py


def test_import_smoke() -> None:
    assert hasattr(amc_py, "Task")
    assert hasattr(amc_py, "Criticality")
