from pathlib import Path

import pytest

from formal_toolchain.bridge.prefix_extension import build_parameterized_prefix_extension_certificate
from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.contexts import OBLIGATION_CONTEXT_LAYERS
from formal_toolchain.theory.loader import TCB_BACKENDS, load_verified_theory_statement


ROOT = Path(__file__).resolve().parents[3]


def _fixture():
    taskset = {"fingerprint": "b" * 64, "tasks": [{
        "name": "t", "period": 10, "deadline": 10, "offset": 0,
        "priority_index": 0, "c_lo": 1, "c_hi": 1, "criticality": "LO",
    }]}
    contexts = {
        "semantic_context": {"hash": "1" * 64},
        "reference_context": {"hash": "2" * 64},
        "bridge_context": {"hash": "3" * 64},
    }
    predecessors = {}
    for oid in ("REFERENCE_TASKSET", "TIME_PROGRESS", "EFFECTIVE_EVENT_ORDER"):
        witness = {"reference_taskset": taskset} if oid == "REFERENCE_TASKSET" else {}
        predecessors[oid] = obligation_certificate(
            obligation_id=oid, status="PASS",
            context_hash=contexts[OBLIGATION_CONTEXT_LAYERS[oid]]["hash"],
            inputs={}, witness=witness, checker_id="test", checker_version="v1",
        )
    theorem = load_verified_theory_statement(ROOT / "formal_toolchain/theory", "REFERENCE_PREFIX_EXTENSION")
    backend = TCB_BACKENDS[theorem["proof_object"]["backend"]]
    receipt = backend.verify(ROOT / "formal_toolchain/theory" / theorem["proof_object"]["path"], theorem=theorem)
    return taskset, contexts, predecessors, theorem, receipt


def test_fake_verified_receipt_rejected():
    taskset, contexts, predecessors, theorem, receipt = _fixture()
    receipt = {"status": "PASS", "verified": True, "backend": "fake"}
    with pytest.raises(ValueError, match="RECEIPT_INVALID"):
        build_parameterized_prefix_extension_certificate(
            reference_taskset=taskset,
            reference_taskset_certificate=predecessors["REFERENCE_TASKSET"],
            time_progress_certificate=predecessors["TIME_PROGRESS"],
            event_order_certificate=predecessors["EFFECTIVE_EVENT_ORDER"],
            contexts=contexts, context_hash="3" * 64,
            theorem_statement=theorem, theorem_proof_receipt=receipt,
        )


def test_fake_cross_layer_context_rejected():
    taskset, contexts, predecessors, theorem, receipt = _fixture()
    bad = obligation_certificate(
        obligation_id="TIME_PROGRESS", status="PASS", context_hash="2" * 64,
        inputs={}, witness={}, checker_id="test", checker_version="v1",
    )
    with pytest.raises(ValueError, match="predecessor context mismatch"):
        build_parameterized_prefix_extension_certificate(
            reference_taskset=taskset,
            reference_taskset_certificate=predecessors["REFERENCE_TASKSET"],
            time_progress_certificate=bad,
            event_order_certificate=predecessors["EFFECTIVE_EVENT_ORDER"],
            contexts=contexts, context_hash="3" * 64,
            theorem_statement=theorem, theorem_proof_receipt=receipt,
        )
