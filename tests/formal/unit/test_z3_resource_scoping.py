from __future__ import annotations

from formal_toolchain.core.z3_resources import configure_z3, new_solver


class _FakeSolver:
    def __init__(self, *, ctx):
        self.ctx = ctx
        self.params = {}

    def set(self, **kwargs):
        self.params.update(kwargs)


class _FakeZ3:
    def __init__(self):
        self.global_params = {}

    def set_param(self, name, value):
        self.global_params[name] = value

    def Solver(self, *, ctx):
        return _FakeSolver(ctx=ctx)


def test_z3_resource_limits_are_optional(monkeypatch):
    monkeypatch.delenv("AMC_FORMAL_Z3_MEMORY_MB", raising=False)
    z3 = _FakeZ3()
    solver = new_solver(z3, context=object())
    assert z3.global_params == {}
    assert solver.params == {}


def test_z3_resource_configuration_keeps_only_memory_cap(monkeypatch):
    monkeypatch.setenv("AMC_FORMAL_Z3_MEMORY_MB", "8192")
    # A stale timeout environment variable must not impose a proof cutoff.
    monkeypatch.setenv("AMC_FORMAL_Z3_TIMEOUT_MS", "180000")
    z3 = _FakeZ3()
    context = object()
    assert configure_z3(z3) is None
    solver = new_solver(z3, context=context)
    assert z3.global_params["memory_max_size"] == 8192
    assert solver.ctx is context
    assert solver.params == {}
