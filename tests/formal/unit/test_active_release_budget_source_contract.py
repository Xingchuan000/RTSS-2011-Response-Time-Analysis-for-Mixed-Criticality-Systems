from pathlib import Path

from formal_toolchain.conformance.active_release_budget import (
    check_active_release_budget_source_contract,
)


def _write_runtime(root: Path, body: str) -> None:
    path = root / "amc_py" / "event_runtime.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "class EventRuntimeEngine:\n"
        "    def apply_budget_updates(self, updates):\n"
        + body,
        encoding="utf-8",
    )


def test_active_release_source_contract_accepts_future_release_update_only(tmp_path):
    _write_runtime(
        tmp_path,
        "        update_payload = dict(updates)\n"
        "        self.budget_state.apply_updates(update_payload)\n",
    )
    result = check_active_release_budget_source_contract(tmp_path)
    assert result["status"] == "PASS"
    assert result["witness"]["runtime_budget_at_release_write_count"] == 0


def test_active_release_source_contract_rejects_retroactive_write(tmp_path):
    _write_runtime(
        tmp_path,
        "        update_payload = dict(updates)\n"
        "        for job in self.state.active_jobs:\n"
        "            job.runtime_budget_at_release = update_payload[job.task.name]\n"
        "        self.budget_state.apply_updates(update_payload)\n",
    )
    result = check_active_release_budget_source_contract(tmp_path)
    assert result["status"] == "FAIL"
    assert result["route"] == "MODEL_CONFORMANCE_FAILED"
    assert result["failure"]["code"] == "ACTIVE_RELEASE_SNAPSHOT_MUTATED"
