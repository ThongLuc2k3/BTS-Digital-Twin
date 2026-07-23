#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRICK_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$TRICK_DIR")"

source "$TRICK_DIR/hcm0031/default.env"

if [[ ! -x "$COLMAP_BIN" && "$(basename "$COLMAP_BIN")" != "colmap" ]]; then
  echo "Lỗi: COLMAP_BIN chưa đúng. Hiện tại: $COLMAP_BIN" >&2
  exit 1
fi

export RUN_DENSE=1
export RUN_TRAIN=0
export RUN_RENDER=0
export RUN_EVAL=0

echo "== Dense pilot =="
echo "scene=$SCENE"
echo "dataset_root=$DATASET_ROOT"
echo "work_root=$WORK_ROOT"

bash "$PROJECT_ROOT/pipeline/scripts/05_run_b2_pilot.sh" "$SCENE"

echo "Dense summary: $LOG_DIR/04_colmap_dense_summary.txt"
