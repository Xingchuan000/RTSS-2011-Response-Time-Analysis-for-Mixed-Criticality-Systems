import importlib.util
import pytest

HAS_Z3 = importlib.util.find_spec("z3") is not None
requires_z3 = pytest.mark.skipif(not HAS_Z3, reason="z3-solver is required")
