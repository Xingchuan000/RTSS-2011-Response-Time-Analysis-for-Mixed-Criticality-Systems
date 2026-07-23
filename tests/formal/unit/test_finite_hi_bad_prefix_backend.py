from pathlib import Path

from formal_toolchain.theory.backends.finite_hi_bad_prefix import (
    EXPECTED_N6_SOLVER_OBLIGATIONS, verify_finite_hi_bad_prefix_math,
)


def test_all_n6_solver_obligations_are_unsat():
    result = verify_finite_hi_bad_prefix_math()
    assert result["status"] == "PASS"
    assert tuple(sorted(result["obligations"])) == tuple(sorted(EXPECTED_N6_SOLVER_OBLIGATIONS))
    assert all(item["result"] == "UNSAT" for item in result["obligations"].values())


def test_service_and_demand_equalities_are_required_for_reflection():
    import z3

    ctx = z3.Context()
    c_service, r_service = z3.Ints("c_service r_service", ctx=ctx)
    c_demand, r_demand = z3.Ints("c_demand r_demand", ctx=ctx)
    solver = z3.Solver(ctx=ctx)
    solver.add(c_service < c_demand, r_service >= r_demand, c_service != r_service,
               c_demand != r_demand, c_service >= 0, r_service >= 0, c_demand > 0, r_demand > 0)
    assert solver.check() == z3.sat

