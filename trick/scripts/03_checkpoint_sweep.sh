#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRICK_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$TRICK_DIR")"

source "$TRICK_DIR/hcm0031/default.env"

if [[ ! -f "$GS_REPO/train.py" ]]; then
  echo "Lỗi: GS_REPO chưa đúng. Hiện tại: $GS_REPO" >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR/point_cloud" ]]; then
  echo "Lỗi: không thấy checkpoint trong $MODEL_DIR/point_cloud" >&2
  exit 1
fi

mkdir -p "$SWEEP_OUT_DIR"
SUMMARY_CSV="$SWEEP_OUT_DIR/summary.csv"
echo "iteration,psnr,ssim,lpips,score,renders_dir,eval_csv" > "$SUMMARY_CSV"

mapfile -t ITER_DIRS < <(find "$MODEL_DIR/point_cloud" -maxdepth 1 -mindepth 1 -type d -name 'iteration_*' | sort -V)
if [[ ${#ITER_DIRS[@]} -eq 0 ]]; then
  echo "Lỗi: không có iteration nào để quét." >&2
  exit 1
fi

for ITER_DIR in "${ITER_DIRS[@]}"; do
  ITERATION="${ITER_DIR##*_}"
  RENDERS_DIR="$SWEEP_OUT_DIR/renders_$ITERATION"
  EVAL_CSV="$SWEEP_OUT_DIR/eval_$ITERATION.csv"

  echo "== Sweep iteration $ITERATION =="

  python3 "$PROJECT_ROOT/pipeline/scripts/render_round1_test_poses.py" \
    --scene "$SCENE" \
    --dataset_root "$DATASET_ROOT" \
    --model_dir "$MODEL_DIR" \
    --iteration "$ITERATION" \
    --out_dir "$RENDERS_DIR" \
    >"$SWEEP_OUT_DIR/render_$ITERATION.log" 2>&1

  python3 "$PROJECT_ROOT/pipeline/scripts/eval_round1_metrics.py" \
    --scene "$SCENE" \
    --dataset_root "$DATASET_ROOT" \
    --renders_dir "$RENDERS_DIR" \
    --out_csv "$EVAL_CSV" \
    --psnr_max "$SWEEP_PSNR_MAX" \
    >"$SWEEP_OUT_DIR/eval_$ITERATION.log" 2>&1

  METRIC_LINE="$(python3 - "$EVAL_CSV" <<'PY'
import csv
import sys
from statistics import mean

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("empty eval csv")

fields = ("psnr", "ssim", "lpips", "score")
vals = [mean(float(r[k]) for r in rows) for k in fields]
print(",".join(f"{v:.4f}" for v in vals))
PY
)"

  echo "$ITERATION,$METRIC_LINE,$RENDERS_DIR,$EVAL_CSV" >> "$SUMMARY_CSV"
done

echo "Sweep summary: $SUMMARY_CSV"
