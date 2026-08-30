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
    closure_wrapper_calls = _function_calls(kernel, "encode_event_node_closure")
    edge_calls = _function_calls(kernel, "encode_event_relative_edge")
    step_calls = _function_calls(kernel, "encode_event_step_for_source")

    assert "encode_p5_controller_enabled" in closure_calls
    assert "encode_p5_identity" in closure_calls
    assert "encode_p5_from_exact_pool" not in text
    assert "ExactControllerPool" not in text
    assert "_source_partition_definition" in candidate_calls
    assert "_declare_exact_periodic_successor" not in candidate_calls
    assert "_exact_p0_to_p7_closure" in closure_wrapper_calls
    assert {
        "build_event_candidates",
        "_silent_interval_service",
        "_event_destination_update",
    } <= edge_calls
    assert {"encode_event_node_closure", "encode_event_relative_edge"} <= step_calls
    assert "def encode_p5_controller_enabled" in transition.read_text(encoding="utf-8")


def test_canonical_source_partition_owns_simultaneous_timestamps_once():
    kernel = (ROOT / "formal_toolchain/v9_2/event_kernel.py").read_text(encoding="utf-8")
    assert 'rows: list[EventSource] = [EventSource("HORIZON"), EventSource("CONTROLLER")]' in kernel
    assert "*(value > chosen for value in earlier)" in kernel
    assert "*(value >= chosen for value in later)" in kernel
    assert "candidates.next_delta == chosen" in kernel
    assert "_min_expr" not in kernel


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
    assert "encode_event_node_closure" in text
    assert "encode_event_relative_edge" in text
    assert "enumerate_controller_policy_cases" in text
    assert "EVENT_GRAPH_NODE_CLOSURE_CHECK" in text
    assert "EVENT_GRAPH_SOURCE_TIME_CHECK" in text
    assert "EVENT_GRAPH_SILENT_SERVICE_CHECK" in text
    assert "EVENT_GRAPH_DESTINATION_CHECK" in text
    assert "EVENT_GRAPH_TIME_EDGE_CHECK" not in text
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


def test_event_refinement_contains_target_local_safety_dominance_gates():
    text = (ROOT / "formal_toolchain/v9_2/event_refinement.py").read_text(encoding="utf-8")
    for obligation in (
        "FULL_TO_EVENT_SEGMENT_SIMULATION",
        "FULL_TO_PROJECTED_EVENT_PREFIX_SIMULATION",
        "FIRST_HI_BAD_PROJECTED_EVENT_REFLECTION",
        "TARGET_LOCAL_FIXED_PRIORITY_INTERFERENCE_DOMINANCE",
        "TARGET_LOCAL_POLICY_STATE_RETENTION",
        "EVENT_LAZY_RELEASE_DEMAND_INDEPENDENCE",
        "EVENT_CONTROLLER_POLICY_CASE_PARTITION_EQUIVALENCE",
        "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY",
        "EVENT_WINDOW_ENCODING_SOUNDNESS",
        "EVENT_P5_GRAPH_BRANCH_SPECIALIZATION_EQUIVALENCE",
        "EVENT_EXPLICIT_GRAPH_SOURCE_PARTITION_EQUIVALENCE",
    ):
        assert obligation in text
    terminal_prefix = text.split("EVENT_TERMINAL_OBLIGATIONS = (", 1)[1].split(")", 1)[0]
    assert "EVENT_TO_FULL_SEGMENT_REALIZABILITY" not in terminal_prefix
    assert "EVENT_BAD_PREFIX_FULL_REALIZABILITY" not in terminal_prefix
    assert "P0_P6_DEFINITIONAL_IDENTITY" in text
    assert "P7_DELTA1" in text


def test_event_window_uses_lazy_release_demand_and_target_local_ssa_frames():
    environment = (ROOT / "formal_toolchain/v9_2/environment_encoder.py").read_text(encoding="utf-8")
    transition = (ROOT / "formal_toolchain/v9_2/transition_encoder.py").read_text(encoding="utf-8")
    symbolic = (ROOT / "formal_toolchain/v9_2/symbolic_state.py").read_text(encoding="utf-8")
    window = (ROOT / "formal_toolchain/v9_2/event_window_encoder.py").read_text(encoding="utf-8")
    assert "def declare_event_graph_environment" in environment
    assert "lazy_release_demands=True" in environment
    assert "if env.lazy_release_demands" in environment
    graph_env = environment.split("def declare_event_graph_environment", 1)[1].split("def demand_for_time", 1)[0]
    assert "constraints = (origin >= 0,)" in graph_env
    assert "modulus = lcm(" not in graph_env
    assert "periodic_phase_constraints(" not in graph_env
    assert "origin_time % modulus" not in graph_env
    assert "declare_event_graph_environment" in window
    assert "derive_target_scheduling_projection" in window
    assert "declare_sparse_successor" in symbolic
    assert "active_task_names" in symbolic
    assert "left.eq(right)" in transition
    assert "encode_p5_invariant_summary" not in (ROOT / "formal_toolchain/v9_2/event_kernel.py").read_text(encoding="utf-8")


def test_periodic_event_candidates_use_relative_eta_countdowns():
    kernel = ROOT / "formal_toolchain/v9_2/event_kernel.py"
    text = kernel.read_text(encoding="utf-8")
    function = text.split("def build_event_candidates", 1)[1].split(
        "def _silent_interval_service", 1
    )[0]
    assert "task.period)) - dispatch_state.eta[task.name]" in function
    assert "controller_delta" in function
    assert "period_index" not in function
    assert "_declare_exact_periodic_successor" not in text


def test_relative_countdown_refinement_matches_next_periodic_reference():
    refinement = ROOT / "formal_toolchain/v9_2/event_refinement.py"
    text = refinement.read_text(encoding="utf-8")
    assert "def _release_countdown_counterexample" in text
    assert "def _controller_countdown_counterexample" in text
    assert "RELATIVE_EVENT_COUNTDOWN_EQUIVALENCE" in text
    assert "PERIODIC_EVENT_CANDIDATE_SCALARIZATION" not in text
    release = text.split("def _release_countdown_counterexample", 1)[1].split(
        "def _controller_countdown_counterexample", 1
    )[0]
    assert "countdown = period - eta" in release
    assert "((t / period) + 1) * period - t" in release
