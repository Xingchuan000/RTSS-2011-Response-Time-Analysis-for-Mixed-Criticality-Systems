from pathlib import Path

from formal_toolchain.core.contexts import OBLIGATION_CONTEXT_LAYERS
from formal_toolchain.core.registry import load_registry
from formal_toolchain.verifier.checker_catalog import checker_for


ROOT = Path(__file__).resolve().parents[3]


def _active_required_entries():
    registry = load_registry(ROOT / "formal_toolchain/specs/obligation_registry.json")
    return [
        entry for entry in registry
        if entry["activation"] == "active" and entry["required"] is True
    ]


def test_every_active_required_obligation_has_context_and_checker():
    for entry in _active_required_entries():
        assert entry["id"] in OBLIGATION_CONTEXT_LAYERS, entry["id"]
        assert OBLIGATION_CONTEXT_LAYERS[entry["id"]] == entry["context_layer"], entry["id"]
        assert callable(checker_for(entry["id"])), entry["id"]


def test_new_core_has_no_explicit_unresolved_checker():
    core = {
        "BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION",
        "REFERENCE_MODEL_CONFORMANCE",
        "REFERENCE_TASKSET_SCHEDULABLE",
        "REFERENCE_HI_SUBSET_SAFETY",
        "EFFECTIVE_EVENT_FRONTIER_RELATION",
        "EARLY_STOP_CONFIGURATION_GATE",
        "REFERENCE_PREFIX_EXTENSION",
        "HI_BAD_CLOSED_PREFIX_REFLECTION",
        "FINITE_BAD_PREFIX_CONTRADICTION",
        "FINAL_CLAIM_COMPOSITION",
    }
    for obligation_id in core:
        checker = checker_for(obligation_id)
        assert checker.__name__ != f"verify_{obligation_id.lower()}"
