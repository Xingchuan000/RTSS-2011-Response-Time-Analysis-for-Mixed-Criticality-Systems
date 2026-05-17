#!/usr/bin/env bash
set -euo pipefail

cd /Users/x1ngchuan/Documents/AMC

mkdir -p outputs/tasksets logs

STAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="logs/qos_pressure_strong_medium_0_1200_${STAMP}.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== QoS-Pressure STRONG-MEDIUM scan started at $(date) ==="
echo "Log: $MAIN_LOG"

COMMON_ARGS=(
  --workload mc_fairgen
  --mc-fairgen-mode paper_learnable_headroom
  --mc-fairgen-num-tasks 12
  --mc-fairgen-hi-ratio 0.5
  --mc-fairgen-period-source automotive
  --mc-fairgen-u-hi-lo-min 0.20
  --mc-fairgen-u-hi-lo-max 0.35
  --mc-fairgen-u-hi-hi-min 0.45
  --mc-fairgen-u-hi-hi-max 0.70
  --mc-fairgen-u-lo-lo-min 0.25
  --mc-fairgen-u-lo-lo-max 0.45
  --mc-fairgen-hi-budget-rho-min 0.55
  --mc-fairgen-hi-budget-rho-max 0.75
  --mc-fairgen-lo-budget-rho-min 0.10
  --mc-fairgen-lo-budget-rho-max 0.30
  --mc-fairgen-hi-overrun-prob 0.08
  --mc-fairgen-lo-overrun-prob 0.20
  --mc-fairgen-hi-overrun-factor-min 1.02
  --mc-fairgen-hi-overrun-factor-max 1.25
  --mc-fairgen-lo-overrun-factor-min 1.02
  --mc-fairgen-lo-overrun-factor-max 1.25
  --end-time 1000000
  --agent-period 50000
)

echo ""
echo "============================================================"
echo "Stage 1: quick scan 0..1199, eval 200:204, stronger pressure"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/scan_qos_pressure_tasksets.py \
  "${COMMON_ARGS[@]}" \
  --seed-start 0 \
  --seed-end 1200 \
  --eval-seeds 200:204 \
  --output outputs/tasksets/qos_pressure_strong_quick_scan_0_1200_eval200_204_lop020_lorho010_030.csv

echo ""
echo "============================================================"
echo "Stage 1b: summarize quick scan"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/summarize_qos_pressure_scan.py \
  --scan-csv outputs/tasksets/qos_pressure_strong_quick_scan_0_1200_eval200_204_lop020_lorho010_030.csv \
  --output outputs/tasksets/qos_pressure_strong_quick_scan_0_1200_eval200_204_lop020_lorho010_030_summary.csv

echo ""
echo "============================================================"
echo "Stage 2: save EASY top20 from quick scan"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/select_qos_pressure_tasksets.py \
  --scan-csv outputs/tasksets/qos_pressure_strong_quick_scan_0_1200_eval200_204_lop020_lorho010_030.csv \
  --bucket easy \
  --top-k 20 \
  --min-baseline-lc-service-loss 0.00 \
  --max-baseline-lc-service-loss 0.10 \
  --min-released-lo-jobs 100 \
  --min-cancelled-lo-jobs 0 \
  --min-mode-changes 0 \
  --min-static-sweep-reduction 0.00 \
  --output outputs/tasksets/mc_fairgen_qos_pressure_strong_easy_top20_0_1200_lop020_lorho010_030.csv \
  --rejections-output outputs/tasksets/mc_fairgen_qos_pressure_strong_easy_top20_0_1200_lop020_lorho010_030_rejections.csv

echo ""
echo "============================================================"
echo "Stage 3: save HARD top20 from quick scan"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/select_qos_pressure_tasksets.py \
  --scan-csv outputs/tasksets/qos_pressure_strong_quick_scan_0_1200_eval200_204_lop020_lorho010_030.csv \
  --bucket hard \
  --top-k 20 \
  --min-baseline-lc-service-loss 0.30 \
  --max-baseline-lc-service-loss 0.50 \
  --min-released-lo-jobs 100 \
  --min-cancelled-lo-jobs 10 \
  --min-mode-changes 0 \
  --min-static-sweep-reduction 0.00 \
  --output outputs/tasksets/mc_fairgen_qos_pressure_strong_hard_top20_0_1200_lop020_lorho010_030.csv \
  --rejections-output outputs/tasksets/mc_fairgen_qos_pressure_strong_hard_top20_0_1200_lop020_lorho010_030_rejections.csv

echo ""
echo "============================================================"
echo "Stage 4: select strong-medium candidates from quick scan"
echo "Target quick loss range: 0.14~0.35, prefer center 0.22"
echo "============================================================"

python - <<'PY'
import pandas as pd
from pathlib import Path

quick = Path("outputs/tasksets/qos_pressure_strong_quick_scan_0_1200_eval200_204_lop020_lorho010_030.csv")
df = pd.read_csv(quick)

cand = df[
    (df["baseline_hi_deadline_misses_sum"] == 0)
    & (df["baseline_lc_service_loss_mean"] >= 0.14)
    & (df["baseline_lc_service_loss_mean"] <= 0.35)
    & (df["baseline_released_lo_jobs_mean"] >= 100)
    & (df["baseline_cancelled_lo_jobs_mean"] >= 10)
].copy()

cand["dist_to_022"] = (cand["baseline_lc_service_loss_mean"] - 0.22).abs()
cand = cand.sort_values(["dist_to_022", "candidate_seed"]).head(250)

seeds = sorted(cand["candidate_seed"].astype(int).unique())

print(f"strong-medium candidate count = {len(seeds)}")
print(f"first 40 seeds = {seeds[:40]}")

cand.to_csv("outputs/tasksets/qos_pressure_strong_medium_candidates_from_quick_0_1200_top250_lop020_lorho010_030.csv", index=False)

with open("outputs/tasksets/qos_pressure_strong_medium_candidates_from_quick_0_1200_top250_lop020_lorho010_030.txt", "w") as f:
    f.write(",".join(map(str, seeds)))

if len(seeds) == 0:
    raise SystemExit("No strong-medium candidates selected. Stop before formal rescan.")
PY

echo ""
echo "============================================================"
echo "Stage 5: formal baseline rescan for strong-medium top250"
echo "============================================================"

CANDIDATE_SEEDS=$(cat outputs/tasksets/qos_pressure_strong_medium_candidates_from_quick_0_1200_top250_lop020_lorho010_030.txt)

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/scan_qos_pressure_tasksets.py \
  "${COMMON_ARGS[@]}" \
  --candidate-seeds "$CANDIDATE_SEEDS" \
  --eval-seeds 200:229 \
  --output outputs/tasksets/qos_pressure_strong_formal_baseline_candidates_0_1200_top250_lop020_lorho010_030.csv

echo ""
echo "============================================================"
echo "Stage 5b: summarize formal baseline rescan"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/summarize_qos_pressure_scan.py \
  --scan-csv outputs/tasksets/qos_pressure_strong_formal_baseline_candidates_0_1200_top250_lop020_lorho010_030.csv \
  --output outputs/tasksets/qos_pressure_strong_formal_baseline_candidates_0_1200_top250_lop020_lorho010_030_summary.csv

echo ""
echo "============================================================"
echo "Stage 6: select formal strong-medium top80 for static sweep"
echo "Formal loss target: 0.15~0.30, prefer center 0.22"
echo "============================================================"

python - <<'PY'
import pandas as pd
from pathlib import Path

formal = Path("outputs/tasksets/qos_pressure_strong_formal_baseline_candidates_0_1200_top250_lop020_lorho010_030.csv")
df = pd.read_csv(formal)

med = df[
    (df["baseline_hi_deadline_misses_sum"] == 0)
    & (df["baseline_lc_service_loss_mean"] >= 0.15)
    & (df["baseline_lc_service_loss_mean"] <= 0.30)
    & (df["baseline_released_lo_jobs_mean"] >= 100)
    & (df["baseline_cancelled_lo_jobs_mean"] >= 10)
].copy()

med["dist_to_022"] = (med["baseline_lc_service_loss_mean"] - 0.22).abs()
med = med.sort_values(["dist_to_022", "candidate_seed"]).head(80)

seeds = sorted(med["candidate_seed"].astype(int).unique())

print(f"formal strong-medium top80 count = {len(seeds)}")
print(f"seeds = {seeds}")

med.to_csv("outputs/tasksets/qos_pressure_strong_formal_medium_top80_presweep_0_1200_lop020_lorho010_030.csv", index=False)

with open("outputs/tasksets/qos_pressure_strong_formal_medium_top80_presweep_0_1200_lop020_lorho010_030.txt", "w") as f:
    f.write(",".join(map(str, seeds)))

if len(seeds) == 0:
    raise SystemExit("No formal strong-medium seeds selected. Stop before static sweep.")
PY

echo ""
echo "============================================================"
echo "Stage 7: static sweep for formal strong-medium top80"
echo "============================================================"

MEDIUM_SEEDS=$(cat outputs/tasksets/qos_pressure_strong_formal_medium_top80_presweep_0_1200_lop020_lorho010_030.txt)

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/scan_qos_pressure_tasksets.py \
  "${COMMON_ARGS[@]}" \
  --candidate-seeds "$MEDIUM_SEEDS" \
  --eval-seeds 200:229 \
  --enable-static-sweep \
  --sweep-inc-ratios 0,0.015,0.025,0.035 \
  --sweep-dec-ratios 0,0.010,0.015 \
  --static-sweep-detail-output outputs/tasksets/qos_pressure_strong_static_sweep_medium_top80_0_1200_lop020_lorho010_030_detail.csv \
  --output outputs/tasksets/qos_pressure_strong_static_sweep_medium_top80_0_1200_lop020_lorho010_030.csv

echo ""
echo "============================================================"
echo "Stage 7b: summarize static sweep"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/summarize_qos_pressure_scan.py \
  --scan-csv outputs/tasksets/qos_pressure_strong_static_sweep_medium_top80_0_1200_lop020_lorho010_030.csv \
  --output outputs/tasksets/qos_pressure_strong_static_sweep_medium_top80_0_1200_lop020_lorho010_030_summary.csv

echo ""
echo "============================================================"
echo "Stage 8: select final STRONG-MEDIUM top20"
echo "============================================================"

KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/select_qos_pressure_tasksets.py \
  --scan-csv outputs/tasksets/qos_pressure_strong_static_sweep_medium_top80_0_1200_lop020_lorho010_030.csv \
  --bucket medium \
  --top-k 20 \
  --min-baseline-lc-service-loss 0.15 \
  --max-baseline-lc-service-loss 0.30 \
  --min-released-lo-jobs 100 \
  --min-cancelled-lo-jobs 10 \
  --min-mode-changes 0 \
  --min-static-sweep-reduction 0.00 \
  --target-loss-center 0.22 \
  --output outputs/tasksets/mc_fairgen_qos_pressure_strong_medium_top20_0_1200_lop020_lorho010_030.csv \
  --rejections-output outputs/tasksets/mc_fairgen_qos_pressure_strong_medium_top20_0_1200_lop020_lorho010_030_rejections.csv

echo ""
echo "============================================================"
echo "Stage 9: final inspection"
echo "============================================================"

python - <<'PY'
import pandas as pd
from pathlib import Path

files = {
    "easy": "outputs/tasksets/mc_fairgen_qos_pressure_strong_easy_top20_0_1200_lop020_lorho010_030.csv",
    "strong_medium": "outputs/tasksets/mc_fairgen_qos_pressure_strong_medium_top20_0_1200_lop020_lorho010_030.csv",
    "hard": "outputs/tasksets/mc_fairgen_qos_pressure_strong_hard_top20_0_1200_lop020_lorho010_030.csv",
}

for name, path in files.items():
    p = Path(path)
    print(f"\n=== {name.upper()} ===")
    if not p.exists():
        print(f"missing: {path}")
        continue
    df = pd.read_csv(p)
    print("rows:", len(df))
    cols = [
        "candidate_seed",
        "baseline_lc_service_loss_mean",
        "baseline_lc_qos_mean",
        "baseline_cancelled_lo_jobs_mean",
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
echo "=== QoS-Pressure STRONG-MEDIUM scan finished at $(date) ==="
echo ""
echo "Main outputs:"
echo "  outputs/tasksets/qos_pressure_strong_quick_scan_0_1200_eval200_204_lop020_lorho010_030_summary.csv"
echo "  outputs/tasksets/qos_pressure_strong_formal_baseline_candidates_0_1200_top250_lop020_lorho010_030_summary.csv"
echo "  outputs/tasksets/qos_pressure_strong_static_sweep_medium_top80_0_1200_lop020_lorho010_030_summary.csv"
echo "  outputs/tasksets/mc_fairgen_qos_pressure_strong_easy_top20_0_1200_lop020_lorho010_030.csv"
echo "  outputs/tasksets/mc_fairgen_qos_pressure_strong_medium_top20_0_1200_lop020_lorho010_030.csv"
echo "  outputs/tasksets/mc_fairgen_qos_pressure_strong_hard_top20_0_1200_lop020_lorho010_030.csv"
echo "Log:"
echo "  $MAIN_LOG"
