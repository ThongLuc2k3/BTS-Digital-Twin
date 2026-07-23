#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRICK_DIR="$(dirname "$SCRIPT_DIR")"

source "$TRICK_DIR/hcm0031/default.env"

MASK_ROOT="$TRICK_DIR/hcm0031/m0_mask"
BOOTSTRAP_DIR="$MASK_ROOT/bootstrap_masks"
MANUAL_DIR="$MASK_ROOT/manual_masks"
METRICS_DIR="$MASK_ROOT/metrics"
MASK_DIR="${MASK_DIR:-$MANUAL_DIR}"
RENDERS_DIR="${RENDERS_DIR:-$WORK_ROOT/$SCENE/renders}"
OUT_CSV="$METRICS_DIR/masked_eval.csv"
SUMMARY_TXT="$METRICS_DIR/masked_eval_summary.txt"
MIN_MASK_COVERAGE="${MIN_MASK_COVERAGE:-0.001}"

mkdir -p "$METRICS_DIR"

if ! find "$MASK_DIR" -maxdepth 1 -type f -name '*.png' | grep -q .; then
  if find "$BOOTSTRAP_DIR" -maxdepth 1 -type f -name '*.png' | grep -q .; then
    MASK_DIR="$BOOTSTRAP_DIR"
  else
    echo "Lỗi: chưa có mask trong $MANUAL_DIR hoặc $BOOTSTRAP_DIR" >&2
    exit 1
  fi
fi

python3 "$SCRIPT_DIR/eval_round1_mask_metrics.py" \
  --scene "$SCENE" \
  --dataset_root "$DATASET_ROOT" \
  --renders_dir "$RENDERS_DIR" \
  --mask_dir "$MASK_DIR" \
  --out_csv "$OUT_CSV" \
  --summary_txt "$SUMMARY_TXT" \
  --psnr_max "$SWEEP_PSNR_MAX" \
  --min_coverage "$MIN_MASK_COVERAGE"

echo "Mask dir used: $MASK_DIR"
echo "Masked summary: $SUMMARY_TXT"
