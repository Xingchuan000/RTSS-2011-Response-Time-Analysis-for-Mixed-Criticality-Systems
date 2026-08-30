import ast
from pathlib import Path

ROOT = Path(__file__).parents[3]


def _function_calls(path: Path, name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            calls: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.add(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.add(child.func.attr)
            return calls
    raise AssertionError(f"missing function {name} in {path}")


def test_event_macro_is_source_specialized_and_controller_branch_specialized():
    kernel = ROOT / "formal_toolchain/v9_2/event_kernel.py"
    transition = ROOT / "formal_toolchain/v9_2/transition_encoder.py"
    text = kernel.read_text(encoding="utf-8")
    closure_calls = _function_calls(kernel, "_exact_p0_to_p7_closure")
    candidate_calls = _function_calls(kernel, "build_event_candidates")
    step_calls = _function_calls(kernel, "encode_event_step_for_source")

    assert "encode_p5_controller_enabled" in closure_calls
    assert "encode_p5_identity" in closure_calls
    assert "encode_p5_from_exact_pool" not in text
    assert "ExactControllerPool" not in text
    assert "_source_partition_definition" in candidate_calls
    assert "_declare_exact_periodic_successor" in candidate_calls
    assert {
        "_exact_p0_to_p7_closure",
        "build_event_candidates",
        "_silent_interval_core",
        "_event_destination_update",
    } <= step_calls
    assert "def encode_p5_controller_enabled" in transition.read_text(encoding="utf-8")


def test_canonical_source_partition_owns_simultaneous_timestamps_once():
    kernel = (ROOT / "formal_toolchain/v9_2/event_kernel.py").read_text(encoding="utf-8")
    assert 'rows: list[EventSource] = [EventSource("HORIZON"), EventSource("CONTROLLER")]' in kernel
    assert "*(value > chosen for value in earlier)" in kernel
    assert "*(value >= chosen for value in later)" in kernel
    assert "candidates.next_time == chosen" in kernel
    assert "Global ``min`` disjunction is intentionally absent" in kernel


def test_event_window_builds_only_graph_root_not_multi_event_unroll():
    path = ROOT / "formal_toolchain/v9_2/event_window_encoder.py"
    text = path.read_text(encoding="utf-8")
    assert "class EventGraphProblem" in text
    assert "build_event_graph_problem" in text
    assert "derive_finite_event_bound" in text
    assert "build_incremental_event_first_bad_window" not in text
    assert "append_exact_event_step" not in text
    assert "build_exact_controller_pool" not in text
    assert "event_step_or_terminal_stutter" not in text
    assert "event_layer_added_abstractions: tuple[str, ...] = ()" in text
    assert "exact_p5_in_event_window: bool = True" in text


def test_explicit_event_graph_solver_owns_combinatorial_search():
    path = ROOT / "formal_toolchain/v9_2/event_graph_solver.py"
    text = path.read_text(encoding="utf-8")
    assert "def solve_event_graph" in text
    assert "def dfs(" in text
    assert "enumerate_event_sources" in text
    assert "encode_event_step_for_source" in text
    assert "solver.push()" in text and "solver.pop()" in text
    assert "for depth in range(" not in text
    assert "timeout" not in text.lower()
    assert '"symbolic_multi_event_skeleton": False' in text
    assert "FRESH_SPECIALIZED_LEAF" not in text
    assert not (ROOT / "formal_toolchain/v9_2/incremental_event_bmc.py").exists()


def test_trusted_verifier_only_uses_explicit_event_graph():
    text = (ROOT / "formal_toolchain/v9_2/verifier.py").read_text(encoding="utf-8")
    assert "prove_event_refinement" in text
    assert "build_event_graph_problem" in text
    assert "solve_event_graph" in text
    assert "build_incremental_event_first_bad_window" not in text
    assert "solve_incremental_event_window" not in text
    assert "SOLVE_FIRST_BAD_EVENT_WINDOW_EVENT_GRAPH" in text


def test_event_refinement_contains_graph_and_bidirectional_gates():
    text = (ROOT / "formal_toolchain/v9_2/event_refinement.py").read_text(encoding="utf-8")
    for obligation in (
        "FULL_TO_EVENT_SEGMENT_SIMULATION",
        "EVENT_TO_FULL_SEGMENT_REALIZABILITY",
        "FIRST_HI_BAD_EVENT_PREFIX_REFLECTION",
        "EVENT_BAD_PREFIX_FULL_REALIZABILITY",
        "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY",
        "EVENT_WINDOW_ENCODING_SOUNDNESS",
        "EVENT_P5_GRAPH_BRANCH_SPECIALIZATION_EQUIVALENCE",
        "EVENT_EXPLICIT_GRAPH_SOURCE_PARTITION_EQUIVALENCE",
    ):
        assert obligation in text
    assert "EVENT_P5_POOL_SUPPORT_PROJECTION_EQUIVALENCE" not in text
    assert "EVENT_INCREMENTAL_TERMINAL_DEPTH_PARTITION_EQUIVALENCE" not in text
    assert "P0_P6_DEFINITIONAL_IDENTITY" in text
    assert "P7_DELTA1" in text


def test_event_window_uses_indexed_exact_demand_lookup_and_ssa_frames():
    environment = (ROOT / "formal_toolchain/v9_2/environment_encoder.py").read_text(encoding="utf-8")
    transition = (ROOT / "formal_toolchain/v9_2/transition_encoder.py").read_text(encoding="utf-8")
    symbolic = (ROOT / "formal_toolchain/v9_2/symbolic_state.py").read_text(encoding="utf-8")
    assert "A_lookup" in environment
    assert "lookup(relative)" in environment
    assert "declare_sparse_successor" in symbolic
    assert "left.eq(right)" in transition
    assert "encode_p5_invariant_summary" not in (ROOT / "formal_toolchain/v9_2/event_kernel.py").read_text(encoding="utf-8")


def test_periodic_event_candidates_use_quotient_free_exact_successor():
    kernel = ROOT / "formal_toolchain/v9_2/event_kernel.py"
    candidate_calls = _function_calls(kernel, "build_event_candidates")
    text = kernel.read_text(encoding="utf-8")
    assert "_declare_exact_periodic_successor" in candidate_calls
    assert "period_index * period" in text
    assert "nxt > t" in text
    assert "nxt <= t + period" in text
    assert "_next_periodic_after" not in text


def test_periodic_scalarization_reverse_direction_uses_constructed_witness():
    refinement = ROOT / "formal_toolchain/v9_2/event_refinement.py"
    text = refinement.read_text(encoding="utf-8")
    function = text.split("def _periodic_scalarization_counterexample", 1)[1].split(
        "def _release_tick_domain_counterexample", 1
    )[0]
    assert "reference_index = (t / period) + 1" in function
    assert "reference_witness = z3.And(" in function
    assert "z3.And(nxt == reference, z3.Not(reference_witness))" in function
    assert "z3.And(nxt == reference, z3.Not(scalar))" not in function
