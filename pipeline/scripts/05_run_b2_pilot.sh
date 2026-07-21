#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PIPELINE_DIR")"

if [[ $# -ne 1 ]]; then
  echo "Cách dùng: $0 <SCENE>" >&2
  exit 1
fi

SCENE="$1"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/Dataset/VAI_NVS_DATA/phase1/public_set}"
WORK_ROOT="${WORK_ROOT:-$PIPELINE_DIR/work}"
MODEL_DIR="${MODEL_DIR:-$WORK_ROOT/$SCENE/gs_model}"
ITERATION="${ITERATION:--1}"
RUN_DENSE="${RUN_DENSE:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SOURCE_MODE="${SOURCE_MODE:-prepared}"
RENDER_OUT_DIR="${RENDER_OUT_DIR:-$WORK_ROOT/$SCENE/renders_b2_pilot}"
EVAL_OUT_CSV="${EVAL_OUT_CSV:-$WORK_ROOT/$SCENE/eval_metrics_b2_pilot.csv}"
TOWER_BBOX3D_JSON="${TOWER_BBOX3D_JSON:-$WORK_ROOT/$SCENE/tower_bbox3d.json}"
SKYLINE_TOP_FRAC="${SKYLINE_TOP_FRAC:-0.3}"
LOG_DIR="$WORK_ROOT/$SCENE/logs"
SUMMARY_FILE="$LOG_DIR/05_b2_pilot_summary.txt"

mkdir -p "$LOG_DIR"

if [[ "$RUN_DENSE" == "1" ]]; then
  "$SCRIPT_DIR/04_run_colmap_dense.sh" "$SCENE"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  SOURCE_MODE="$SOURCE_MODE" "$SCRIPT_DIR/03_train_3dgs.sh" "$SCENE"
fi

if [[ "$RUN_RENDER" == "1" ]]; then
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Lỗi: không thấy MODEL_DIR=$MODEL_DIR" >&2
    exit 1
  fi
  python3 "$SCRIPT_DIR/render_round1_test_poses.py" \
    --scene "$SCENE" \
    --dataset_root "$DATASET_ROOT" \
    --model_dir "$MODEL_DIR" \
    --iteration "$ITERATION" \
    --out_dir "$RENDER_OUT_DIR" \
    >"$LOG_DIR/05_render_b2_pilot.log" 2>&1
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ ! -d "$RENDER_OUT_DIR" ]]; then
    echo "Lỗi: không thấy RENDER_OUT_DIR=$RENDER_OUT_DIR" >&2
    exit 1
  fi

  EVAL_ARGS=(
    --scene "$SCENE"
    --dataset_root "$DATASET_ROOT"
    --renders_dir "$RENDER_OUT_DIR"
    --out_csv "$EVAL_OUT_CSV"
    --skyline_top_frac "$SKYLINE_TOP_FRAC"
  )

  if [[ -f "$TOWER_BBOX3D_JSON" ]]; then
    EVAL_ARGS+=(--tower_bbox3d_json "$TOWER_BBOX3D_JSON")
  fi

  python3 "$SCRIPT_DIR/eval_round1_metrics.py" \
    "${EVAL_ARGS[@]}" \
    >"$LOG_DIR/05_eval_b2_pilot.log" 2>&1
fi

{
  echo "scene=$SCENE"
  echo "dataset_root=$DATASET_ROOT"
  echo "work_root=$WORK_ROOT"
  echo "run_dense=$RUN_DENSE"
  echo "run_train=$RUN_TRAIN"
  echo "run_render=$RUN_RENDER"
  echo "run_eval=$RUN_EVAL"
  echo "source_mode=$SOURCE_MODE"
  echo "model_dir=$MODEL_DIR"
  echo "render_out_dir=$RENDER_OUT_DIR"
  echo "eval_out_csv=$EVAL_OUT_CSV"
  echo "tower_bbox3d_json=$TOWER_BBOX3D_JSON"
  echo "skyline_top_frac=$SKYLINE_TOP_FRAC"
  echo "dense_summary=$LOG_DIR/04_colmap_dense_summary.txt"
  echo "render_log=$LOG_DIR/05_render_b2_pilot.log"
  echo "eval_log=$LOG_DIR/05_eval_b2_pilot.log"
} >"$SUMMARY_FILE"

echo "B2 pilot workflow xong: scene=$SCENE"
echo "Summary: $SUMMARY_FILE"
