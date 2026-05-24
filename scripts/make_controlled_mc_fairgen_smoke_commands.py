#!/usr/bin/env python3
"""生成 controlled MC-FairGen smoke pipeline 命令。"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--period-source", default="controlled_medium")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    src = args.period_source
    base = f"outputs/tasksets/{src}"
    text = f"""#!/usr/bin/env bash
set -euo pipefail

python scripts/generate_learnable_tasksets.py --workload mc_fairgen --mc-fairgen-period-source {src} --mc-fairgen-num-tasks 12 --num-tasksets 1001 --candidate-seed-start 0
python scripts/scan_taskset_headroom.py --workload mc_fairgen --mc-fairgen-period-source {src} --taskset-manifest {base}/fullscan_0_1000.csv --eval-seeds 200:206 --end-time 3000000
python scripts/probe_stable_improvement_tasksets.py --workload mc_fairgen --mc-fairgen-period-source {src} --taskset-manifest {base}/fullscan_0_1000.csv --seeds 200:206 --end-time 3000000 --output-summary outputs/taskset_probe/{src}/stable_probe_summary.csv --output-detail outputs/taskset_probe/{src}/stable_probe_detail.csv
python scripts/select_probe_aware_tasksets.py --headroom-summary outputs/taskset_slack_scan/{src}/headroom_quick_summary.csv --probe-summary outputs/taskset_probe/{src}/stable_probe_summary.csv --manifest-csv {base}/fullscan_0_1000.csv --top-k 10 --output-summary {base}/probe_aware_top10.csv --output-manifest {base}/probe_aware_top10_manifest.csv --output-rejections {base}/probe_aware_rejections.csv
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"accept rate log path: {base}")


if __name__ == "__main__":
    main()
