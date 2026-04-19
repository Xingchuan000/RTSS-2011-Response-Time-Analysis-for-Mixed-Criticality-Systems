import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

from amc_py.generator import make_generation_config


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_rtss11.py"
# 测试仅依赖 preset/metadata 逻辑，提前注入轻量 stub，避免强依赖绘图库。
matplotlib_stub = types.ModuleType("matplotlib")
matplotlib_stub.use = lambda *_args, **_kwargs: None  # type: ignore[assignment]
pyplot_stub = types.ModuleType("matplotlib.pyplot")
pyplot_stub.subplots = lambda *args, **kwargs: (None, None)  # type: ignore[assignment]
pyplot_stub.close = lambda *args, **kwargs: None  # type: ignore[assignment]
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)
sys.modules.setdefault("pandas", types.ModuleType("pandas"))

spec = importlib.util.spec_from_file_location("reproduce_rtss11", SCRIPT_PATH)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["reproduce_rtss11"] = module
spec.loader.exec_module(module)


get_figure_preset = module.get_figure_preset
make_runtime_generation_config = module.make_runtime_generation_config
write_figure_metadata = module.write_figure_metadata


def test_paper_figure_presets_match_rtss11_definition() -> None:
    fig1 = get_figure_preset("fig1", "paper")
    fig2 = get_figure_preset("fig2", "paper")
    fig3 = get_figure_preset("fig3", "paper")
    fig4 = get_figure_preset("fig4", "paper")
    fig5 = get_figure_preset("fig5", "paper")

    assert fig1.y_axis_metric == "schedulable_ratio"
    assert fig5.y_axis_metric == "schedulable_ratio"
    assert fig2.y_axis_metric == "weighted_schedulability"
    assert fig3.y_axis_metric == "weighted_schedulability"
    assert fig4.y_axis_metric == "weighted_schedulability"

    assert fig1.util_values[0] == 0.025
    assert fig1.util_values[-1] == 0.975
    assert len(fig1.util_values) == 39

    assert fig2.cf_values == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    assert fig3.cp_values == [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    assert fig4.n_values == [8, 24, 40, 56, 72, 88]



def test_figure_metadata_contains_metric_and_axis(tmp_path: Path) -> None:
    cfg = make_generation_config("paper")
    preset = get_figure_preset("fig5", "paper")
    args = argparse.Namespace(mode="paper", config=None, num_tasksets=10, seed=7)

    write_figure_metadata(tmp_path, "fig5", args, cfg, preset)

    payload = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert payload["figure_name"] == "fig5"
    assert payload["x_axis"] == "util"
    assert payload["y_axis_metric"] == "schedulable_ratio"
    assert payload["deadline_mode"] == "arbitrary_paper"
    assert payload["generator_profile"] == "mode:paper"
    assert payload["criticality_assignment"] == cfg.criticality_assignment


def test_fig5_runtime_uses_arbitrary_paper_deadline_mode() -> None:
    cfg = make_generation_config("paper")
    preset = get_figure_preset("fig5", "paper")
    runtime_cfg = make_runtime_generation_config(cfg, preset, total_util=0.5)
    assert preset.deadline_mode == "arbitrary_paper"
    assert runtime_cfg.deadline_mode == "arbitrary_paper"
