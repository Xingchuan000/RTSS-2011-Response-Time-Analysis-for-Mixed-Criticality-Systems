from __future__ import annotations

import pandas as pd
from scripts.select_probe_aware_tasksets import main
import sys


def test_probe_aware_selector_filters_and_sorts(tmp_path) -> None:
    headroom = pd.DataFrame(
        [
            {"candidate_seed": 1, "valid_increase_count_mean": 4, "valid_decrease_count_mean": 4, "increase_decrease_balance": 1.0, "baseline_total_events_per_1m": 120, "baseline_mode_changes_per_1m": 10, "baseline_lo_cancellation_ratio_total": 0.4},
            {"candidate_seed": 2, "valid_increase_count_mean": 4, "valid_decrease_count_mean": 4, "increase_decrease_balance": 1.0, "baseline_total_events_per_1m": 120, "baseline_mode_changes_per_1m": 10, "baseline_lo_cancellation_ratio_total": 0.4},
        ]
    )
    probe = pd.DataFrame(
        [
            {"candidate_seed": 1, "probe_stable_best_found": True, "probe_stable_best_relative_lo_cancellation_reduction": 0.06, "probe_tradeoff_risk": False},
            {"candidate_seed": 2, "probe_stable_best_found": True, "probe_stable_best_relative_lo_cancellation_reduction": 0.03, "probe_tradeoff_risk": True},
        ]
    )
    manifest = pd.DataFrame([{"candidate_seed": 1}, {"candidate_seed": 2}])
    hs = tmp_path / "h.csv"
    ps = tmp_path / "p.csv"
    ms = tmp_path / "m.csv"
    out_s = tmp_path / "o.csv"
    out_m = tmp_path / "om.csv"
    out_r = tmp_path / "r.csv"
    headroom.to_csv(hs, index=False)
    probe.to_csv(ps, index=False)
    manifest.to_csv(ms, index=False)

    argv = sys.argv
    sys.argv = ["x", "--headroom-summary", str(hs), "--probe-summary", str(ps), "--manifest-csv", str(ms), "--output-summary", str(out_s), "--output-manifest", str(out_m), "--output-rejections", str(out_r)]
    try:
        main()
    finally:
        sys.argv = argv

    out = pd.read_csv(out_s)
    assert out.iloc[0]["candidate_seed"] == 1
    rejected = pd.read_csv(out_r)
    assert any(rejected["candidate_seed"] == 2)
