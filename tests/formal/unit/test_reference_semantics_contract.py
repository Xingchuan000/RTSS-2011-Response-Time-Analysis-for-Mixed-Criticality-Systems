from formal_toolchain.reference.semantics_contract import evaluate_reference_semantics_contract


def test_reference_semantics_contract_passes_finite_taskset():
    result = evaluate_reference_semantics_contract({
        "fingerprint": "a" * 64,
        "tasks": [
            {"name": "hi", "criticality": "HI", "priority_index": 0, "c_lo": 2, "c_hi": 5},
            {"name": "lo", "criticality": "LO", "priority_index": 1, "c_lo": 3, "c_hi": 1},
        ],
    })
    assert result["status"] == "PASS", result


def test_reference_semantics_contract_checks_unique_switch():
    result = evaluate_reference_semantics_contract({
        "tasks": [{"name": "hi", "criticality": "HI", "priority_index": 0, "c_lo": 1, "c_hi": 2}],
    })
    assert result["checks"]["unique_switch_trigger"] is True
