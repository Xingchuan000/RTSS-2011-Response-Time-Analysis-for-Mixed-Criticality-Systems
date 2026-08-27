from __future__ import annotations

from pathlib import Path


def test_v9_1_binding_freezes_c_amc_sem_switch_boundary_assumptions() -> None:
    source = Path("formal_toolchain/v9_1/bindings.py").read_text(encoding="utf-8")
    assert 'V9_1_REQUIRES_C_AMC_SEM_RUNTIME_SEMANTICS' in source
    assert 'V9_1_REQUIRES_PRIMARY_ON_SWITCH_TIME_TRUE' in source
    assert 'V9_1_REQUIRES_C_AMC_SEM_ACTIVE_LO_CONTINUATION' in source
    assert '"primary_on_switch_time": True' in source
    assert '"drop_lo_jobs_on_hi_switch": False' in source
