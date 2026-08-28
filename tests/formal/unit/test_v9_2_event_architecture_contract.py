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


def test_event_macro_uses_exact_controller_and_exact_event_minimum():
    kernel = ROOT / "formal_toolchain/v9_2/event_kernel.py"
    closure_calls = _function_calls(kernel, "_exact_p0_to_p7_closure")
    candidate_calls = _function_calls(kernel, "build_event_candidates")
    step_calls = _function_calls(kernel, "encode_event_step")

    assert "encode_p5_controller" in closure_calls
    assert "encode_p5_invariant_summary" not in kernel.read_text(encoding="utf-8")
    assert "_next_periodic_after" in candidate_calls
    assert "_min_expr" in candidate_calls
    assert {"_exact_p0_to_p7_closure", "build_event_candidates", "_silent_interval_advance"} <= step_calls


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


def test_trusted_verifier_only_builds_event_first_bad_windows():
    path = ROOT / "formal_toolchain/v9_2/verifier.py"
    text = path.read_text(encoding="utf-8")
    assert "prove_event_refinement" in text
    assert "build_event_first_bad_window" in text
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
    ):
        assert obligation in text
    assert "DELTA1" in text
    assert "DELTA2" in text
