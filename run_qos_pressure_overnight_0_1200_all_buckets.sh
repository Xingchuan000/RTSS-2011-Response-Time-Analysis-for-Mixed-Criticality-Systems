#!/usr/bin/env bash
set -euo pipefail

cd /Users/x1ngchuan/Documents/AMC

mkdir -p outputs/tasksets logs

STAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="logs/qos_pressure_overnight_0_1200_all_buckets_${STAMP}.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== QoS-Pressure overnight scan started at $(date) ==="
echo "Log: $MAIN_LOG"

echo ""
echo "============================================================"
echo "Stage 1: quick baseline scan, seeds 0..1199, eval 200:204"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/scan_qos_pressure_tasksets.py \
  --workload mc_fairgen \
  --mc-fairgen-mode paper_learnable_headroom \
  --mc-fairgen-num-tasks 12 \
  --mc-fairgen-hi-ratio 0.5 \
  --mc-fairgen-period-source automotive \
  --mc-fairgen-u-hi-lo-min 0.20 \
  --mc-fairgen-u-hi-lo-max 0.35 \
  --mc-fairgen-u-hi-hi-min 0.45 \
  --mc-fairgen-u-hi-hi-max 0.70 \
  --mc-fairgen-u-lo-lo-min 0.25 \
  --mc-fairgen-u-lo-lo-max 0.45 \
  --mc-fairgen-hi-budget-rho-min 0.55 \
  --mc-fairgen-hi-budget-rho-max 0.75 \
  --mc-fairgen-lo-budget-rho-min 0.20 \
  --mc-fairgen-lo-budget-rho-max 0.40 \
  --mc-fairgen-hi-overrun-prob 0.08 \
  --mc-fairgen-lo-overrun-prob 0.12 \
  --mc-fairgen-hi-overrun-factor-min 1.02 \
  --mc-fairgen-hi-overrun-factor-max 1.25 \
  --mc-fairgen-lo-overrun-factor-min 1.02 \
  --mc-fairgen-lo-overrun-factor-max 1.25 \
  --seed-start 0 \
  --seed-end 1200 \
  --eval-seeds 200:204 \
  --end-time 1000000 \
  --agent-period 50000 \
  --output outputs/tasksets/qos_pressure_quick_scan_0_1200_eval200_204.csv

echo ""
echo "============================================================"
echo "Stage 1b: summarize quick scan"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/summarize_qos_pressure_scan.py \
  --scan-csv outputs/tasksets/qos_pressure_quick_scan_0_1200_eval200_204.csv \
  --output outputs/tasksets/qos_pressure_quick_scan_0_1200_eval200_204_summary.csv

echo ""
echo "============================================================"
echo "Stage 2: select top200 candidate seeds from quick scan"
echo "============================================================"

python - <<'PY'
import pandas as pd
from pathlib import Path

quick = Path("outputs/tasksets/qos_pressure_quick_scan_0_1200_eval200_204.csv")
df = pd.read_csv(quick)

required = [
    "candidate_seed",
    "baseline_hi_deadline_misses_sum",
    "baseline_lc_service_loss_mean",
    "baseline_released_lo_jobs_mean",
    "baseline_cancelled_lo_jobs_mean",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns in quick scan: {missing}")

cand = df[
    (df["baseline_hi_deadline_misses_sum"] == 0)
    & (df["baseline_lc_service_loss_mean"] >= 0.08)
    & (df["baseline_lc_service_loss_mean"] <= 0.35)
    & (df["baseline_released_lo_jobs_mean"] >= 100)
    & (df["baseline_cancelled_lo_jobs_mean"] >= 10)
].copy()

cand["dist_to_020"] = (cand["baseline_lc_service_loss_mean"] - 0.20).abs()
cand = cand.sort_values(["dist_to_020", "candidate_seed"]).head(200)

seeds = sorted(cand["candidate_seed"].astype(int).unique())

print(f"candidate count = {len(seeds)}")
print(f"first 30 seeds = {seeds[:30]}")

cand.to_csv("outputs/tasksets/qos_pressure_candidates_from_quick_0_1200_top200.csv", index=False)

with open("outputs/tasksets/qos_pressure_candidates_from_quick_0_1200_top200.txt", "w") as f:
    f.write(",".join(map(str, seeds)))

if len(seeds) == 0:
    raise SystemExit("No candidate seeds selected. Stop before formal rescan.")
PY

echo ""
echo "============================================================"
echo "Stage 3: formal baseline rescan for top200 candidates, eval 200:229"
echo "============================================================"

CANDIDATE_SEEDS=$(cat outputs/tasksets/qos_pressure_candidates_from_quick_0_1200_top200.txt)

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/scan_qos_pressure_tasksets.py \
  --workload mc_fairgen \
  --mc-fairgen-mode paper_learnable_headroom \
  --mc-fairgen-num-tasks 12 \
  --mc-fairgen-hi-ratio 0.5 \
  --mc-fairgen-period-source automotive \
  --mc-fairgen-u-hi-lo-min 0.20 \
  --mc-fairgen-u-hi-lo-max 0.35 \
  --mc-fairgen-u-hi-hi-min 0.45 \
  --mc-fairgen-u-hi-hi-max 0.70 \
  --mc-fairgen-u-lo-lo-min 0.25 \
  --mc-fairgen-u-lo-lo-max 0.45 \
  --mc-fairgen-hi-budget-rho-min 0.55 \
  --mc-fairgen-hi-budget-rho-max 0.75 \
  --mc-fairgen-lo-budget-rho-min 0.20 \
  --mc-fairgen-lo-budget-rho-max 0.40 \
  --mc-fairgen-hi-overrun-prob 0.08 \
  --mc-fairgen-lo-overrun-prob 0.12 \
  --mc-fairgen-hi-overrun-factor-min 1.02 \
  --mc-fairgen-hi-overrun-factor-max 1.25 \
  --mc-fairgen-lo-overrun-factor-min 1.02 \
  --mc-fairgen-lo-overrun-factor-max 1.25 \
  --candidate-seeds "$CANDIDATE_SEEDS" \
  --eval-seeds 200:229 \
  --end-time 1000000 \
  --agent-period 50000 \
  --output outputs/tasksets/qos_pressure_formal_baseline_candidates_0_1200_top200.csv

echo ""
echo "============================================================"
echo "Stage 3b: summarize formal baseline rescan"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/summarize_qos_pressure_scan.py \
  --scan-csv outputs/tasksets/qos_pressure_formal_baseline_candidates_0_1200_top200.csv \
  --output outputs/tasksets/qos_pressure_formal_baseline_candidates_0_1200_top200_summary.csv

echo ""
echo "============================================================"
echo "Stage 4: select formal medium top50 for static sweep"
echo "============================================================"

python - <<'PY'
import pandas as pd
from pathlib import Path

formal = Path("outputs/tasksets/qos_pressure_formal_baseline_candidates_0_1200_top200.csv")
df = pd.read_csv(formal)

med = df[
    (df["baseline_hi_deadline_misses_sum"] == 0)
    & (df["baseline_lc_service_loss_mean"] >= 0.10)
    & (df["baseline_lc_service_loss_mean"] <= 0.30)
    & (df["baseline_released_lo_jobs_mean"] >= 100)
    & (df["baseline_cancelled_lo_jobs_mean"] >= 10)
].copy()

med["dist_to_020"] = (med["baseline_lc_service_loss_mean"] - 0.20).abs()
med = med.sort_values(["dist_to_020", "candidate_seed"]).head(50)

seeds = sorted(med["candidate_seed"].astype(int).unique())

print(f"formal medium top50 count = {len(seeds)}")
print(f"seeds = {seeds}")

med.to_csv("outputs/tasksets/qos_pressure_formal_medium_top50_presweep_0_1200.csv", index=False)

with open("outputs/tasksets/qos_pressure_formal_medium_top50_presweep_0_1200.txt", "w") as f:
    f.write(",".join(map(str, seeds)))

if len(seeds) == 0:
    raise SystemExit("No formal medium seeds selected. Stop before static sweep.")
PY

echo ""
echo "============================================================"
echo "Stage 5: static sweep for formal medium top50"
echo "============================================================"

MEDIUM_SEEDS=$(cat outputs/tasksets/qos_pressure_formal_medium_top50_presweep_0_1200.txt)

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/scan_qos_pressure_tasksets.py \
  --workload mc_fairgen \
  --mc-fairgen-mode paper_learnable_headroom \
  --mc-fairgen-num-tasks 12 \
  --mc-fairgen-hi-ratio 0.5 \
  --mc-fairgen-period-source automotive \
  --mc-fairgen-u-hi-lo-min 0.20 \
  --mc-fairgen-u-hi-lo-max 0.35 \
  --mc-fairgen-u-hi-hi-min 0.45 \
  --mc-fairgen-u-hi-hi-max 0.70 \
  --mc-fairgen-u-lo-lo-min 0.25 \
  --mc-fairgen-u-lo-lo-max 0.45 \
  --mc-fairgen-hi-budget-rho-min 0.55 \
  --mc-fairgen-hi-budget-rho-max 0.75 \
  --mc-fairgen-lo-budget-rho-min 0.20 \
  --mc-fairgen-lo-budget-rho-max 0.40 \
  --mc-fairgen-hi-overrun-prob 0.08 \
  --mc-fairgen-lo-overrun-prob 0.12 \
  --mc-fairgen-hi-overrun-factor-min 1.02 \
  --mc-fairgen-hi-overrun-factor-max 1.25 \
  --mc-fairgen-lo-overrun-factor-min 1.02 \
  --mc-fairgen-lo-overrun-factor-max 1.25 \
  --candidate-seeds "$MEDIUM_SEEDS" \
  --eval-seeds 200:229 \
  --end-time 1000000 \
  --agent-period 50000 \
  --enable-static-sweep \
  --sweep-inc-ratios 0,0.025,0.035 \
  --sweep-dec-ratios 0,0.015 \
  --static-sweep-detail-output outputs/tasksets/qos_pressure_static_sweep_medium_top50_0_1200_detail.csv \
  --output outputs/tasksets/qos_pressure_static_sweep_medium_top50_0_1200.csv

echo ""
echo "============================================================"
echo "Stage 5b: summarize static sweep scan"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/summarize_qos_pressure_scan.py \
  --scan-csv outputs/tasksets/qos_pressure_static_sweep_medium_top50_0_1200.csv \
  --output outputs/tasksets/qos_pressure_static_sweep_medium_top50_0_1200_summary.csv

echo ""
echo "============================================================"
echo "Stage 6: select final MEDIUM top20 from static sweep result"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/select_qos_pressure_tasksets.py \
  --scan-csv outputs/tasksets/qos_pressure_static_sweep_medium_top50_0_1200.csv \
  --bucket medium \
  --top-k 20 \
  --min-baseline-lc-service-loss 0.10 \
  --max-baseline-lc-service-loss 0.30 \
  --min-released-lo-jobs 100 \
  --min-cancelled-lo-jobs 10 \
  --min-mode-changes 0 \
  --min-static-sweep-reduction 0.00 \
  --output outputs/tasksets/mc_fairgen_qos_pressure_medium_top20_0_1200.csv \
  --rejections-output outputs/tasksets/mc_fairgen_qos_pressure_medium_top20_0_1200_rejections.csv

echo ""
echo "============================================================"
echo "Stage 6b: select EASY top20 from quick scan"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/select_qos_pressure_tasksets.py \
  --scan-csv outputs/tasksets/qos_pressure_quick_scan_0_1200_eval200_204.csv \
  --bucket easy \
  --top-k 20 \
  --min-baseline-lc-service-loss 0.00 \
  --max-baseline-lc-service-loss 0.10 \
  --min-released-lo-jobs 100 \
  --min-cancelled-lo-jobs 10 \
  --min-mode-changes 0 \
  --min-static-sweep-reduction 0.00 \
  --output outputs/tasksets/mc_fairgen_qos_pressure_easy_top20_0_1200.csv \
  --rejections-output outputs/tasksets/mc_fairgen_qos_pressure_easy_top20_0_1200_rejections.csv

echo ""
echo "============================================================"
echo "Stage 6c: select HARD top20 from quick scan"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/select_qos_pressure_tasksets.py \
  --scan-csv outputs/tasksets/qos_pressure_quick_scan_0_1200_eval200_204.csv \
  --bucket hard \
  --top-k 20 \
  --min-baseline-lc-service-loss 0.30 \
  --max-baseline-lc-service-loss 0.50 \
  --min-released-lo-jobs 100 \
  --min-cancelled-lo-jobs 10 \
  --min-mode-changes 0 \
  --min-static-sweep-reduction 0.00 \
  --output outputs/tasksets/mc_fairgen_qos_pressure_hard_top20_0_1200.csv \
  --rejections-output outputs/tasksets/mc_fairgen_qos_pressure_hard_top20_0_1200_rejections.csv

echo ""
echo "============================================================"
echo "Stage 7: final bucket inspection"
echo "============================================================"

python - <<'PY'
import pandas as pd
from pathlib import Path

for bucket in ["easy", "medium", "hard"]:
    path = Path(f"outputs/tasksets/mc_fairgen_qos_pressure_{bucket}_top20_0_1200.csv")
    print(f"\n=== {bucket.upper()} ===")
    if not path.exists():
        print(f"missing: {path}")
        continue

    df = pd.read_csv(path)
    print("rows:", len(df))

    cols = [
        "candidate_seed",
        "baseline_lc_service_loss_mean",
        "baseline_lc_qos_mean",
        "baseline_mode_changes_mean",
        "qos_pressure_bucket",
    ]
    if "static_sweep_relative_lc_loss_reduction" in df.columns:
        cols.append("static_sweep_relative_lc_loss_reduction")
    if "recommended_for_qos_dqn" in df.columns:
        cols.append("recommended_for_qos_dqn")
    cols = [c for c in cols if c in df.columns]

    if len(df) > 0:
        print(df[cols].to_string(index=False))
PY

echo ""
echo "=== QoS-Pressure overnight scan finished at $(date) ==="
echo ""
echo "Main outputs:"
echo "  outputs/tasksets/qos_pressure_quick_scan_0_1200_eval200_204.csv"
echo "  outputs/tasksets/qos_pressure_quick_scan_0_1200_eval200_204_summary.csv"
echo "  outputs/tasksets/qos_pressure_formal_baseline_candidates_0_1200_top200.csv"
echo "  outputs/tasksets/qos_pressure_formal_baseline_candidates_0_1200_top200_summary.csv"
echo "  outputs/tasksets/qos_pressure_static_sweep_medium_top50_0_1200.csv"
echo "  outputs/tasksets/qos_pressure_static_sweep_medium_top50_0_1200_summary.csv"
echo "  outputs/tasksets/mc_fairgen_qos_pressure_easy_top20_0_1200.csv"
echo "  outputs/tasksets/mc_fairgen_qos_pressure_medium_top20_0_1200.csv"
echo "  outputs/tasksets/mc_fairgen_qos_pressure_hard_top20_0_1200.csv"
echo ""
echo "Log:"
echo "  $MAIN_LOG"
