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


def _direct_parameter_attributes(path: Path, name: str, parameter: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return {
                child.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == parameter
            }
    raise AssertionError(f"missing function {name} in {path}")


def test_event_macro_uses_exact_controller_pool_and_exact_event_minimum():
    kernel = ROOT / "formal_toolchain/v9_2/event_kernel.py"
    closure_calls = _function_calls(kernel, "_exact_p0_to_p7_closure")
    pool_calls = _function_calls(kernel, "build_exact_controller_pool")
    pooled_p5_calls = _function_calls(kernel, "encode_p5_from_exact_pool")
    candidate_calls = _function_calls(kernel, "build_event_candidates")
    step_calls = _function_calls(kernel, "encode_event_step")
    transition = ROOT / "formal_toolchain/v9_2/transition_encoder.py"
    controller = ROOT / "formal_toolchain/v9_2/controller_encoder.py"
    numeric = ROOT / "formal_toolchain/v9_2/numeric_encoder.py"
    full_p5_calls = _function_calls(transition, "encode_p5_controller")

    assert "encode_p5_controller" in closure_calls
    assert "encode_p5_from_exact_pool" in closure_calls
    assert "declare_sparse_successor" in closure_calls
    assert "encode_p5_controller_effect" in pool_calls
    assert {
        "_controller_effect_state_equality",
        "encode_p5_controller_frame",
        "encode_p5_identity",
    } <= pooled_p5_calls
    assert {"encode_p5_controller_effect", "encode_p5_controller_frame"} <= full_p5_calls
    assert _direct_parameter_attributes(
        controller, "encode_controller_decision", "state"
    ) <= {"t", "budgets"}
    assert _direct_parameter_attributes(
        numeric, "encode_v11_full_10d_observation", "state"
    ) <= {"budgets", "chi"}
    assert "encode_p5_invariant_summary" not in kernel.read_text(encoding="utf-8")
    assert "_next_periodic_after" in candidate_calls
    assert "_exact_minimum_definition" in candidate_calls
    kernel_text = kernel.read_text(encoding="utf-8")
    assert "_min_expr(" not in kernel_text
    assert ".candidate.next_time" in kernel_text
    assert ".candidate.completion" in kernel_text
    assert "definition_formula" in kernel_text
    assert {
        "_exact_p0_to_p7_closure",
        "build_event_candidates",
        "_silent_interval_core",
        "_event_destination_update",
    } <= step_calls


def test_event_window_has_no_microstep_terminal_unroll_or_memory_slot_cap():
    path = ROOT / "formal_toolchain/v9_2/event_window_encoder.py"
    text = path.read_text(encoding="utf-8")
    assert "derive_finite_event_bound" in text
    assert "finite_event_bound" in text
    assert "event_layer_added_abstractions: tuple[str, ...] = ()" in text
    assert "exact_p5_in_event_window: bool = True" in text
    assert "microstep_terminal_fallback_used: bool = False" in text
    assert "deadline * 8" not in text
    assert "max_event_slots" not in text
    assert "build_first_bad_window" not in text
    assert "build_exact_controller_pool" in text
    assert "controller_bound=event_bound.controller_bound" in text
    # The monolithic builder remains as a reference encoding, while the
    # trusted solver uses the exact incremental terminal-depth partition.
    assert "event_step_or_terminal_stutter" in text
    assert "build_incremental_event_first_bad_window" in text
    assert "append_exact_event_step" in text
    assert "event_boundary_stutter(" not in text


def test_trusted_verifier_only_builds_event_first_bad_windows():
    path = ROOT / "formal_toolchain/v9_2/verifier.py"
    text = path.read_text(encoding="utf-8")
    assert "prove_event_refinement" in text
    assert "build_incremental_event_first_bad_window" in text
    assert "solve_incremental_event_window" in text
    assert "build_first_bad_window" not in text
    assert '"event_layer_added_abstractions": []' in text
    assert '"microstep_terminal_fallback_used": False' in text


def test_event_refinement_contains_bidirectional_and_differential_gates():
    path = ROOT / "formal_toolchain/v9_2/event_refinement.py"
    text = path.read_text(encoding="utf-8")
    for obligation in (
        "FULL_TO_EVENT_SEGMENT_SIMULATION",
        "EVENT_TO_FULL_SEGMENT_REALIZABILITY",
        "FIRST_HI_BAD_EVENT_PREFIX_REFLECTION",
        "EVENT_BAD_PREFIX_FULL_REALIZABILITY",
        "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY",
        "EVENT_WINDOW_ENCODING_SOUNDNESS",
        "EVENT_P5_POOL_SUPPORT_PROJECTION_EQUIVALENCE",
        "EVENT_TERMINAL_STUTTER_FACTORING_EQUIVALENCE",
        "EVENT_INCREMENTAL_TERMINAL_DEPTH_PARTITION_EQUIVALENCE",
    ):
        assert obligation in text
    # Differential consistency is compositional: the Event closure reuses the
    # exact Full P0--P6 encoder sequence, while SMT checks only the P7 quotient
    # point.  A monolithic duplicate-controller DELTA2 formula is forbidden.
    assert "P0_P6_DEFINITIONAL_IDENTITY" in text
    assert "P7_DELTA1" in text
    assert "_small_horizon_differential_counterexample" not in text
    assert "_full_ticks" not in text


def test_event_window_uses_indexed_exact_demand_lookup_and_ssa_frames():
    environment = ROOT / "formal_toolchain/v9_2/environment_encoder.py"
    transition = ROOT / "formal_toolchain/v9_2/transition_encoder.py"
    symbolic = ROOT / "formal_toolchain/v9_2/symbolic_state.py"
    env_text = environment.read_text(encoding="utf-8")
    transition_text = transition.read_text(encoding="utf-8")
    symbolic_text = symbolic.read_text(encoding="utf-8")
    assert "A_lookup" in env_text
    assert "lookup(relative)" in env_text
    assert "declare_sparse_successor" in symbolic_text
    assert "left.eq(right)" in transition_text
    assert "encode_p5_invariant_summary" not in (ROOT / "formal_toolchain/v9_2/event_kernel.py").read_text(encoding="utf-8")


def test_incremental_terminal_depth_solver_is_exact_and_fail_closed():
    solver = (ROOT / "formal_toolchain/v9_2/incremental_event_bmc.py").read_text(encoding="utf-8")
    window = (ROOT / "formal_toolchain/v9_2/event_window_encoder.py").read_text(encoding="utf-8")
    assert "for depth in range(0, max_depth + 1):" in solver
    assert "encoding.append_exact_event_step()" in solver
    assert '"fresh_solver_per_depth": True' in solver
    assert "solver.push()" not in solver and "solver.pop()" not in solver
    assert "CTRL_COUNT_" in solver
    assert "SRC_RELEASE_ANY" in solver
    assert "SRC_HI_DEADLINE_ANY" in solver
    assert "SRC_COMPLETION" in solver
    assert "This is the only path allowed to report window UNSAT" in solver
    assert '"terminal_stutter_used": False' in solver
    assert "encode_event_step(" in window
    incremental_method = window.split("def append_exact_event_step", 1)[1].split("def build_terminal_bad_query", 1)[0]
    assert "event_step_or_terminal_stutter" not in incremental_method
