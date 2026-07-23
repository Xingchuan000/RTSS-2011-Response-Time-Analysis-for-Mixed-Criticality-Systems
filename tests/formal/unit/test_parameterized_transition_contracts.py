from formal_toolchain.bridge.effect_compiler import CompiledConcreteEffect
from formal_toolchain.bridge.transition_cases import derive_parameterized_case_contract


def effect(sources, *, modified=(), semantics=(), queue=()):
    items = tuple({"ast_hash": f"{i + 1:064x}", "source": source} for i, source in enumerate(sources))
    return items, CompiledConcreteEffect(("(= c_job_0_present_post c_job_0_present)",), tuple(item["ast_hash"] for item in items), (), tuple(queue), tuple(modified), tuple(semantics), tuple(sources))


def test_empty_effect_ir_is_unresolved():
    _, compiled = effect([])
    contract = derive_parameterized_case_contract(case_id="PRIMARY_LO_RELEASE", effect_ir=[], compiled_effect=compiled, concrete_delta="", queue_relation_hash="x", expected_queue_relation_hash="x")
    assert contract.status == "UNRESOLVED"
    assert not contract.created_key_fresh_proved


def test_release_requires_real_freshness_evidence():
    ir, compiled = effect(["unrelated = 1"], semantics=("JOB_RELEASE",))
    contract = derive_parameterized_case_contract(case_id="PRIMARY_LO_RELEASE", effect_ir=ir, compiled_effect=compiled, concrete_delta="release_job_key", queue_relation_hash="x", expected_queue_relation_hash="x")
    assert contract.status == "UNRESOLVED"


def test_batch_without_decomposition_is_unresolved():
    ir, compiled = effect(["for arrival in events: _process_single_arrival_in_priority_order(arrival)"], modified=("released_ledger",), semantics=("JOB_RELEASE",))
    contract = derive_parameterized_case_contract(case_id="ARRIVAL_BATCH_NO_SWITCH", effect_ir=ir, compiled_effect=compiled, concrete_delta="(= c_job_0_post c_job_0)", queue_relation_hash="x", expected_queue_relation_hash="x")
    assert contract.status == "UNRESOLVED"
    assert contract.map_update_kind == "EXTEND_WITH_FINITE_RELEASE_BATCH"


def test_illegal_completion_write_breaks_frame():
    ir, compiled = effect(["active_jobs.remove(job)", "job.completion_time = now", "self.state.mode = SystemMode.LO"], modified=("active_jobs", "terminal_ledger", "mode"), semantics=("JOB_REMOVE", "TERMINAL_MARK", "MODE_UPDATE"))
    contract = derive_parameterized_case_contract(case_id="NORMAL_COMPLETION", effect_ir=ir, compiled_effect=compiled, concrete_delta="c_job_0_post", queue_relation_hash="x", expected_queue_relation_hash="x")
    assert contract.status == "UNRESOLVED"
    assert "PARAMETERIZED_UNEXPECTED_STATE_WRITE" in (contract.failure or "")
