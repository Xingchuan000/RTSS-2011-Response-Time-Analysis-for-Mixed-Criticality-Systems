"""Current-source coherent E1-E6 model-mutation catalog.

The mutations intentionally alter both the executable runtime and its frozen
semantic mirror.  They are research-only negative controls: the ordinary source
is never edited because the coherent patch mutator applies them inside an
isolated source overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..python_binding import bind_symbol
from ...canonical import python_symbol_hash


@dataclass(frozen=True)
class ModelMutationCatalogEntry:
    mutation_id: str
    semantic_change_id: str
    expected_status: str
    expected_obligations: tuple[str, ...]
    patches: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            "mutation_id": self.mutation_id,
            "semantic_change_id": self.semantic_change_id,
            "expected_status": self.expected_status,
            "expected_obligations": list(self.expected_obligations),
            "patches": [dict(item) for item in self.patches],
        }


MODEL_MUTATION_CONTRACTS = {
    "E1": ("deadline_cleanup_removes_unfinished_job", "DEADLINE_OBSERVATION"),
    "E2": ("hi_job_truncate", "HI_NONTRUNCATION"),
    "E3": ("event_order_changed", "EFFECTIVE_EVENT_ORDER"),
    "E4": ("controller_overhead_changed", "CONTROLLER_POSTCLOSURE"),
    "E5": ("nonquiescent_recovery_changed", "MODE_SEMANTICS_CONFORMANCE"),
    "E6": ("unstable_demand_reads", "DEMAND_ORACLE_BATCH_CONTRACT"),
}


def _patch(
    root: Path,
    *,
    role: str,
    relative: str,
    symbol: str,
    before: str,
    after: str,
    occurrence: int = 1,
) -> dict:
    source = (root / relative).read_text(encoding="utf-8")
    bound = bind_symbol(source, symbol)
    actual = bound.source.count(before)
    if actual != occurrence:
        raise ValueError(
            f"MODEL_PATCH_BINDING_NOT_UNIQUE:{relative}:{symbol}:"
            f"expected={occurrence}:actual={actual}"
        )
    return {
        "role": role,
        "target_file": relative,
        "target_symbol": symbol,
        "before_ast_hash": python_symbol_hash(source, symbol),
        "before_snippet": before,
        "after_snippet": after,
        "occurrence": occurrence,
    }


def _runtime_pair(
    root: Path,
    *,
    symbol: str,
    before: str,
    after: str,
    occurrence: int = 1,
) -> tuple[dict, dict]:
    return (
        _patch(
            root,
            role="DEPLOYED_IMPLEMENTATION",
            relative="amc_py/event_runtime.py",
            symbol=symbol,
            before=before,
            after=after,
            occurrence=occurrence,
        ),
        _patch(
            root,
            role="FORMAL_SEMANTIC_MIRROR",
            relative="formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py",
            symbol=symbol,
            before=before,
            after=after,
            occurrence=occurrence,
        ),
    )


def build_current_source_model_catalog(root: Path) -> dict[str, ModelMutationCatalogEntry]:
    """Bind E1-E6 against the checked-out ordinary/frozen runtime source."""

    e1_before = """                if self.config.stop_at_first_miss:\n                    return False\n            return True\n"""
    e1_after = """                # Non-vacuity E1: deadline observation also destroys the\n                # unfinished job.  This violates the P0 observation-only contract.\n                if job in self.state.active_jobs:\n                    self.state.active_jobs.remove(job)\n                _invalidate_job_events(self.state, job)\n                if self.state.running_job is job:\n                    self.state.running_job = None\n                    self.state.run_started_at = None\n                if self.config.stop_at_first_miss:\n                    return False\n            return True\n"""

    e2_before = """            self.state.mode = SystemMode.HI\n            self.result.mode_switches.append(\n"""
    e2_after = """            # Non-vacuity E2: truncate an overrunning HI job at the\n            # observed LO budget instead of allowing AMC continuation.\n            if job.task.criticality is Criticality.HI:\n                job.actual_cost = job.executed_time\n            self.state.mode = SystemMode.HI\n            self.result.mode_switches.append(\n"""

    e3_pair = (
        _patch(
            root,
            role="DEPLOYED_IMPLEMENTATION",
            relative="amc_py/event_runtime.py",
            symbol="EventRuntimeEngine.__post_init__",
            before="        self.queue = EventQueue()\n",
            after="        self.queue = EventQueue(arrival_before_deadline=True)\n",
        ),
        _patch(
            root,
            role="FORMAL_SEMANTIC_MIRROR",
            relative="formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py",
            symbol="EventRuntimeEngine.__post_init__",
            before="        self.queue = EventQueue()\n",
            after="        self.queue = EventQueue(arrival_before_deadline=True)\n",
        ),
    )

    e4_before_deployed = "        self._advance_time(self.state.current_time)\n        update_payload = dict(updates)\n"
    e4_after_deployed = """        # Non-vacuity E4: charge one processor tick for each\n        # controller/budget update.\n        self._advance_time(self.state.current_time + 1)\n        update_payload = dict(updates)\n"""
    e4_before_frozen = "        self._advance_time(self.state.current_time)\n        update_payload = dict(updates)\n"
    e4_pair = (
        _patch(
            root,
            role="DEPLOYED_IMPLEMENTATION",
            relative="amc_py/event_runtime.py",
            symbol="EventRuntimeEngine.apply_budget_updates",
            before=e4_before_deployed,
            after=e4_after_deployed,
        ),
        _patch(
            root,
            role="FORMAL_SEMANTIC_MIRROR",
            relative="formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py",
            symbol="EventRuntimeEngine.apply_budget_updates",
            before=e4_before_frozen,
            after=e4_after_deployed,
        ),
    )

    e5_before = "    if state.mode is SystemMode.HI and not state.active_jobs and state.running_job is None:\n"
    e5_after = "    if state.mode is SystemMode.HI:\n"

    e6_before = "    raw_actual_cost = scenario.actual_cost_for(task, release_index)\n"
    e6_after = """    raw_actual_cost = scenario.actual_cost_for(task, release_index)\n    # Non-vacuity E6: the same demand key is no longer stable across releases.\n    raw_actual_cost += release_index % 2\n"""

    patches_by_id: dict[str, tuple[dict, ...]] = {
        "E1": _runtime_pair(root, symbol="EventRuntimeEngine._process_event", before=e1_before, after=e1_after),
        "E2": _runtime_pair(root, symbol="EventRuntimeEngine._process_event", before=e2_before, after=e2_after),
        "E3": e3_pair,
        "E4": e4_pair,
        "E5": _runtime_pair(root, symbol="_maybe_recover_to_lo", before=e5_before, after=e5_after),
        "E6": _runtime_pair(root, symbol="_build_job", before=e6_before, after=e6_after),
    }
    return build_model_mutations(patches_by_id)


def build_model_mutations(patches_by_id: Mapping[str, tuple[dict, ...]]) -> dict[str, ModelMutationCatalogEntry]:
    """Build E1-E6 entries and fail closed when a coherent pair is absent."""
    result = {}
    for mutation_id, (change_id, obligation) in MODEL_MUTATION_CONTRACTS.items():
        patches = tuple(dict(item) for item in patches_by_id.get(mutation_id, ()))
        roles = {str(item.get("role")) for item in patches}
        if "DEPLOYED_IMPLEMENTATION" not in roles:
            raise ValueError(f"{mutation_id}_DEPLOYED_IMPLEMENTATION_PATCH_MISSING")
        if not (roles & {"FORMAL_SEMANTIC_MIRROR", "FROZEN_IMPLEMENTATION"}):
            raise ValueError(f"{mutation_id}_FROZEN_MIRROR_PATCH_MISSING")
        if roles & {"VERIFIER_CHECKER", "AGGREGATOR", "EXPECTED_RESULT_CLASSIFIER"}:
            raise ValueError(f"{mutation_id}_FORBIDDEN_PATCH_ROLE")
        result[mutation_id] = ModelMutationCatalogEntry(
            mutation_id=mutation_id,
            semantic_change_id=change_id,
            expected_status="MODEL_CONFORMANCE_FAILED",
            expected_obligations=(obligation,),
            patches=patches,
        )
    return result


def validate_model_mutation_entry(entry: ModelMutationCatalogEntry) -> None:
    if not entry.patches:
        raise ValueError(f"{entry.mutation_id}_PATCHES_EMPTY")
    if not entry.semantic_change_id:
        raise ValueError(f"{entry.mutation_id}_SEMANTIC_CHANGE_ID_MISSING")
