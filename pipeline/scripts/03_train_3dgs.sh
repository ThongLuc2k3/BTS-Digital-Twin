#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GS_REPO:-}" ]]; then
  echo "Lỗi: chưa set GS_REPO." >&2
  exit 1
fi
if [[ ! -f "$GS_REPO/train.py" ]]; then
  echo "Lỗi: không thấy $GS_REPO/train.py" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PIPELINE_DIR")"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/Dataset/VAI_NVS_DATA_ROUND2}"

ITERATIONS="${ITERATIONS:-30000}"
SH_DEGREE="${SH_DEGREE:-3}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
RESOLUTION="${RESOLUTION:--1}"
ANTIALIASING="${ANTIALIASING:-1}"
EXPOSURE_COMP="${EXPOSURE_COMP:-1}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-1}"
START_CHECKPOINT="${START_CHECKPOINT:-}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"

if [[ $# -ne 1 ]]; then
  echo "Cách dùng: $0 <SCENE>" >&2
  exit 1
fi

SCENE="$1"
SOURCE_DIR="$DATASET_ROOT/$SCENE/train"
MODEL_DIR="$PIPELINE_DIR/work/$SCENE/gs_model"
LOG_FILE="$PIPELINE_DIR/work/$SCENE/train.log"

if [[ ! -d "$SOURCE_DIR/images" ]]; then
  echo "Lỗi: không thấy $SOURCE_DIR/images" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_DIR/sparse/0" ]]; then
  echo "Lỗi: không thấy $SOURCE_DIR/sparse/0" >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"

SAVE_ITERATIONS=()
for v in 7000 15000 "$ITERATIONS"; do
  if [[ "$v" -le "$ITERATIONS" ]]; then
    SAVE_ITERATIONS+=("$v")
  fi
done
SAVE_ITERATIONS=($(printf "%s\n" "${SAVE_ITERATIONS[@]}" | awk '!seen[$0]++'))

EXTRA_ARGS=()
if [[ "$ANTIALIASING" == "1" ]]; then
  EXTRA_ARGS+=(--antialiasing)
fi
if [[ "$EXPOSURE_COMP" == "1" ]]; then
  EXTRA_ARGS+=(--train_test_exp)
fi
if [[ -n "$START_CHECKPOINT" ]]; then
  if [[ ! -f "$START_CHECKPOINT" ]]; then
    echo "Lỗi: START_CHECKPOINT không tồn tại: $START_CHECKPOINT" >&2
    exit 1
  fi
  EXTRA_ARGS+=(--start_checkpoint "$START_CHECKPOINT")
fi
if [[ "$SAVE_FINAL_CHECKPOINT" == "1" ]]; then
  EXTRA_ARGS+=(--checkpoint_iterations "$ITERATIONS")
fi

echo "===== Train 3DGS: scene=$SCENE iterations=$ITERATIONS antialiasing=$ANTIALIASING exposure=$EXPOSURE_COMP ====="
if [[ -n "$START_CHECKPOINT" ]]; then
  echo "Resume từ: $START_CHECKPOINT"
fi

python "$GS_REPO/train.py" \
  -s "$SOURCE_DIR" \
  -m "$MODEL_DIR" \
  --iterations "$ITERATIONS" \
  --save_iterations "${SAVE_ITERATIONS[@]}" \
  --test_iterations "$ITERATIONS" \
  --sh_degree "$SH_DEGREE" \
  --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
  --resolution "$RESOLUTION" \
  "${EXTRA_ARGS[@]}" \
  > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep "$PROGRESS_INTERVAL"
  LAST_PROGRESS=$(grep -oE "[0-9]+/${ITERATIONS}" "$LOG_FILE" 2>/dev/null | tail -1 || true)
  if [[ -n "$LAST_PROGRESS" ]]; then
    echo "[$SCENE] tiến độ: $LAST_PROGRESS"
  fi
done

set +e
wait "$TRAIN_PID"
STATUS=$?
set -e

if [[ $STATUS -ne 0 ]]; then
  echo "[LỖI] Train thất bại. 50 dòng cuối log:" >&2
  tail -n 50 "$LOG_FILE" >&2
  exit $STATUS
fi

cat > "$MODEL_DIR/pipeline_train_flags.json" <<EOF
{"antialiasing": $( [[ "$ANTIALIASING" == "1" ]] && echo true || echo false ), "exposure_comp": $( [[ "$EXPOSURE_COMP" == "1" ]] && echo true || echo false )}
EOF

echo "Xong. Model dir: $MODEL_DIR"
echo "Point cloud cuối: $MODEL_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
if [[ "$SAVE_FINAL_CHECKPOINT" == "1" ]]; then
  echo "Checkpoint resume: $MODEL_DIR/chkpnt${ITERATIONS}.pth"
fi
