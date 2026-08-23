import copy
import json
from pathlib import Path

import pytest

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.contexts import OBLIGATION_CONTEXT_LAYERS
from formal_toolchain.core.registry import load_registry, topological_order
from formal_toolchain.reference import model_conformance as mc
from formal_toolchain.reference.model_conformance import (
    build_reference_model_conformance_certificate,
    load_reference_model_conformance_contract,
)


ROOT = Path(__file__).resolve().parents[3]


def _inputs():
    contract = load_reference_model_conformance_contract()
    semantic = "1" * 64
    reference = "2" * 64
    bridge = "3" * 64
    contexts = {
        "semantic_context": {"hash": semantic},
        "reference_context": {"hash": reference},
        "bridge_context": {"hash": bridge},
    }
    predecessor_ids = sorted({
        predecessor_id
        for condition in contract["conditions"]
        for predecessor_id in condition["predecessor_obligation_ids"]
    })
    predecessors = {}
    for index, oid in enumerate(predecessor_ids):
        layer = OBLIGATION_CONTEXT_LAYERS[oid]
        context_hash = contexts[layer]["hash"]
        if oid == "REFERENCE_TASKSET":
            witness = {"reference_taskset": {"fingerprint": "b" * 64}}
        elif oid == "REFERENCE_TRANSITION_SYSTEM_IDENTITY":
            witness = {
                "transition_system_id": "FIXED_EXECUTABLE_REFERENCE_P0_V3",
                "contract": {"status": "PASS"},
                "identity_scope": {"event_frontier": "EFFECTIVE_EVENT_FRONTIER_RELATION"},
            }
        else:
            witness = {"source": oid}
        predecessors[oid] = obligation_certificate(
            obligation_id=oid, status="PASS", context_hash=context_hash,
            inputs={"fixture": True}, witness=witness,
            checker_id="test.predecessor", checker_version="v1",
        )
    taskset = {
        "schema_version": "reference_taskset_v2",
        "periodic_language_is_sporadic_sub_language": True,
        "fingerprint": "b" * 64,
        "tasks": [{"period": 10, "deadline": 10, "c_lo": 1, "c_hi": 1, "criticality": "HI"}],
    }
    theorem = json.loads((ROOT / "formal_toolchain/theory/statements/C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY.json").read_text())
    return contract, predecessors, taskset, theorem, contexts, reference


def _build(predecessors=None, contract=None, contexts=None):
    original_contract, original_predecessors, taskset, theorem, original_contexts, reference_hash = _inputs()
    return build_reference_model_conformance_certificate(
        reference_taskset=taskset, context_hash=reference_hash,
        verified_predecessors=predecessors or original_predecessors,
        contexts=contexts or original_contexts,
        conformance_contract=contract or original_contract,
        imported_theorem=theorem,
    )


def test_contract_has_exactly_fifteen_unique_conditions_and_dag_is_acyclic():
    contract = load_reference_model_conformance_contract()
    assert len(contract["conditions"]) == 15
    assert {row["condition_id"] for row in contract["conditions"]} == set(mc.EXPECTED_CONFORMANCE_CONDITIONS)
    entries = load_registry(ROOT / "formal_toolchain/specs/obligation_registry.json")
    order = topological_order(entries)
    assert order.index("REFERENCE_PREFIX_EXTENSION") < order.index("REFERENCE_MODEL_CONFORMANCE")


def test_real_cross_context_predecessors_pass():
    certificate = _build()
    assert certificate["obligation_status"] == "PASS"
    assert {row["predecessor_context_layers"][oid] for row in certificate["witness"]["condition_results"] for oid in row["predecessor_ids"]} == {"semantic_context", "reference_context", "bridge_context"}


@pytest.mark.parametrize("obligation_id", ["SCHEDULER_MODEL", "REFERENCE_TASKSET", "REFERENCE_PREFIX_EXTENSION"])
def test_context_mutation_is_fail_closed(obligation_id):
    _, predecessors, _, _, contexts, _ = _inputs()
    mutated = copy.deepcopy(predecessors)
    mutated[obligation_id]["certificate_context_hash"] = "f" * 64
    assert _build(predecessors=mutated, contexts=contexts)["obligation_status"] != "PASS"


@pytest.mark.parametrize("mutation", ["condition_id", "source_kind", "missing_predecessor", "extra_predecessor", "duplicate_predecessor", "unknown_predecessor"])
def test_contract_mutations_are_rejected(monkeypatch, tmp_path, mutation):
    contract, _, _, _, _, _ = _inputs()
    mutated = copy.deepcopy(contract)
    row = mutated["conditions"][0]
    if mutation == "condition_id":
        row["condition_id"] = "NOT_A_SPEC_CONDITION"
    elif mutation == "source_kind":
        row["source_kind"] = "VERIFIED_PREDECESSORS"
    elif mutation == "missing_predecessor":
        row["predecessor_obligation_ids"] = []
    elif mutation == "extra_predecessor":
        row["predecessor_obligation_ids"].append("TIME_PROGRESS")
    elif mutation == "duplicate_predecessor":
        row["predecessor_obligation_ids"].append("REFERENCE_TASKSET")
    else:
        row["predecessor_obligation_ids"] = ["UNKNOWN_OBLIGATION"]
    from formal_toolchain.core.hashing import sha256_object
    mutated["contract_hash"] = sha256_object({k: v for k, v in mutated.items() if k != "contract_hash"})
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    monkeypatch.setattr(mc, "CONTRACT_PATH", path)
    with pytest.raises(ValueError):
        load_reference_model_conformance_contract()


def test_missing_or_extra_predecessor_fails():
    _, predecessors, _, _, _, _ = _inputs()
    missing = copy.deepcopy(predecessors)
    missing.pop("SCHEDULER_MODEL")
    with pytest.raises(ValueError, match="EXACT_PREDECESSOR_SET"):
        _build(predecessors=missing)
    extra = copy.deepcopy(predecessors)
    extra["UNRELATED"] = copy.deepcopy(next(iter(predecessors.values())))
    with pytest.raises(ValueError, match="EXACT_PREDECESSOR_SET"):
        _build(predecessors=extra)


def test_reference_taskset_certificate_fingerprint_is_bound():
    _, predecessors, taskset, theorem, contexts, reference_hash = _inputs()
    predecessors["REFERENCE_TASKSET"]["witness"]["reference_taskset"]["fingerprint"] = "c" * 64
    with pytest.raises(ValueError):
        _build(predecessors=predecessors)
