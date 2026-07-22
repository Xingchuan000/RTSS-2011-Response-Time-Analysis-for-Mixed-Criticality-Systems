import json
from pathlib import Path

from formal_toolchain.reference.model_conformance import load_reference_model_conformance_contract


def test_reference_conformance_requires_reference_semantics():
    contract = load_reference_model_conformance_contract()
    conditions = {row["condition_id"]: row for row in contract["conditions"]}
    for condition_id in (
        "SINGLE_PROCESSOR_PREEMPTIVE_FPPS",
        "ARRIVAL_CLASSIFICATION_UNIQUE_SWITCH",
        "QUIESCENT_LO_RECOVERY",
        "RELEASE_VERSION_SELECTION",
    ):
        assert "REFERENCE_SEMANTICS_CONTRACT" in conditions[condition_id]["predecessor_obligation_ids"]


def test_valid_prefix_partition_has_only_three_cases():
    proof = json.loads(Path(
        "formal_toolchain/theory/proofs/REFERENCE_PREFIX_EXTENSION.proof.json"
    ).read_text())
    assert proof["case_ids"] == [
        "SAME_TIMESTAMP_CLOSURE",
        "READY_SERVICE_OR_EARLIER_BOUNDARY",
        "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT",
    ]
