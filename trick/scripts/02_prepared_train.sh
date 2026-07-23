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

export SOURCE_MODE="${SOURCE_MODE:-prepared}"
export ITERATIONS="${ITERATIONS:-30000}"
export SH_DEGREE="${SH_DEGREE:-3}"
export DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
export RESOLUTION="${RESOLUTION:--1}"
export ANTIALIASING="${ANTIALIASING:-1}"
export EXPOSURE_COMP="${EXPOSURE_COMP:-1}"
export SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-1}"
export PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"

echo "== Prepared train =="
echo "scene=$SCENE"
echo "source_mode=$SOURCE_MODE"
echo "iterations=$ITERATIONS"
echo "model_dir=$MODEL_DIR"

bash "$PROJECT_ROOT/pipeline/scripts/03_train_3dgs.sh" "$SCENE"

echo "Train log: $WORK_ROOT/$SCENE/train.log"
