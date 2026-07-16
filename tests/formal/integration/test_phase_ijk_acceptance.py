"""Phase I-K 正向集成入口验收。"""

import importlib.util

from scripts.run_phase_ijk_acceptance import main


def test_synthetic_fixture_runs_the_normal_phase_k_or_fails_closed(capsys):
    result = main(["--fixture", "synthetic_p0"])
    output = capsys.readouterr().out
    if importlib.util.find_spec("z3") is None:
        assert result == 1
        assert "PHASE_IJK_UNRESOLVED" in output
    else:
        assert result == 0
        assert "PHASE_IJK_ACCEPTED" in output
        assert "DEPLOYED_TREE_PROVED" not in output
