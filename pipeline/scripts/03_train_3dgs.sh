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
TRAIN_GUI_IP="${TRAIN_GUI_IP:-127.0.0.1}"
TRAIN_GUI_PORT="${TRAIN_GUI_PORT:-}"
LOW_VRAM_PROFILE="${LOW_VRAM_PROFILE:-auto}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-}"
DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-}"
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-}"
OPACITY_RESET_INTERVAL="${OPACITY_RESET_INTERVAL:-}"
PERCENT_DENSE="${PERCENT_DENSE:-}"
SAVE_ITERATIONS_OVERRIDE="${SAVE_ITERATIONS_OVERRIDE:-}"
CHECKPOINT_ITERATIONS_OVERRIDE="${CHECKPOINT_ITERATIONS_OVERRIDE:-}"

if [[ $# -ne 1 ]]; then
  echo "Cách dùng: $0 <SCENE>" >&2
  exit 1
fi

SCENE="$1"
PREPARED_SOURCE_DIR="$PIPELINE_DIR/work/$SCENE/colmap/dense"
RAW_SOURCE_DIR="$DATASET_ROOT/$SCENE/train"
SOURCE_MODE="${SOURCE_MODE:-auto}"
case "$SOURCE_MODE" in
  auto)
    if [[ -d "$PREPARED_SOURCE_DIR/images" && -d "$PREPARED_SOURCE_DIR/sparse/0" ]]; then
      SOURCE_DIR="$PREPARED_SOURCE_DIR"
    else
      SOURCE_DIR="$RAW_SOURCE_DIR"
    fi
    ;;
  prepared)
    SOURCE_DIR="$PREPARED_SOURCE_DIR"
    ;;
  raw)
    SOURCE_DIR="$RAW_SOURCE_DIR"
    ;;
  *)
    echo "Lỗi: SOURCE_MODE phải là auto, prepared, hoặc raw. Hiện tại: $SOURCE_MODE" >&2
    exit 1
    ;;
esac
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

build_iteration_list() {
  local raw="$1"
  local max_iter="$2"
  local item
  local out=()

  raw="${raw//,/ }"
  for item in $raw; do
    if [[ "$item" =~ ^[0-9]+$ ]] && [[ "$item" -le "$max_iter" ]]; then
      out+=("$item")
    fi
  done

  if [[ ${#out[@]} -eq 0 ]]; then
    return 0
  fi

  printf "%s\n" "${out[@]}" | awk '!seen[$0]++' | sort -n
}

SAVE_ITERATIONS=()
if [[ -n "$SAVE_ITERATIONS_OVERRIDE" ]]; then
  while IFS= read -r v; do
    [[ -n "$v" ]] && SAVE_ITERATIONS+=("$v")
  done < <(build_iteration_list "$SAVE_ITERATIONS_OVERRIDE" "$ITERATIONS")
else
  for v in 7000 15000 "$ITERATIONS"; do
    if [[ "$v" -le "$ITERATIONS" ]]; then
      SAVE_ITERATIONS+=("$v")
    fi
  done
  SAVE_ITERATIONS=($(printf "%s\n" "${SAVE_ITERATIONS[@]}" | awk '!seen[$0]++' | sort -n))
fi

CHECKPOINT_ITERATIONS=()
if [[ -n "$CHECKPOINT_ITERATIONS_OVERRIDE" ]]; then
  while IFS= read -r v; do
    [[ -n "$v" ]] && CHECKPOINT_ITERATIONS+=("$v")
  done < <(build_iteration_list "$CHECKPOINT_ITERATIONS_OVERRIDE" "$ITERATIONS")
fi

if [[ "$LOW_VRAM_PROFILE" == "auto" ]]; then
  if [[ -d /content || -d /kaggle/working ]]; then
    LOW_VRAM_PROFILE="1"
  else
    LOW_VRAM_PROFILE="0"
  fi
fi

if [[ "$LOW_VRAM_PROFILE" == "1" ]]; then
  if [[ "$RESOLUTION" == "-1" ]]; then
    RESOLUTION="4"
  fi
  if [[ -z "$PYTORCH_CUDA_ALLOC_CONF" ]]; then
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  fi
  if [[ -z "$DENSIFY_UNTIL_ITER" ]]; then
    DENSIFY_UNTIL_ITER="0"
  fi
  if [[ -z "$DENSIFY_FROM_ITER" ]]; then
    DENSIFY_FROM_ITER="0"
  fi
  if [[ -z "$DENSIFICATION_INTERVAL" ]]; then
    DENSIFICATION_INTERVAL="0"
  fi
  if [[ -z "$OPACITY_RESET_INTERVAL" ]]; then
    OPACITY_RESET_INTERVAL="300000"
  fi
  if [[ -z "$PERCENT_DENSE" ]]; then
    PERCENT_DENSE="0.0"
  fi
  if [[ -z "$SAVE_ITERATIONS_OVERRIDE" ]]; then
    SAVE_ITERATIONS=(2000 4000 6000 8000 10000 12000 15000 "$ITERATIONS")
    SAVE_ITERATIONS=($(printf "%s\n" "${SAVE_ITERATIONS[@]}" | awk -v max="$ITERATIONS" '$1 <= max && !seen[$1]++' | sort -n))
  fi
  if [[ -z "$CHECKPOINT_ITERATIONS_OVERRIDE" ]]; then
    CHECKPOINT_ITERATIONS=(2000 4000 6000 8000 10000 12000 15000 "$ITERATIONS")
    CHECKPOINT_ITERATIONS=($(printf "%s\n" "${CHECKPOINT_ITERATIONS[@]}" | awk -v max="$ITERATIONS" '$1 <= max && !seen[$1]++' | sort -n))
  fi
fi

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
if [[ ${#CHECKPOINT_ITERATIONS[@]} -gt 0 ]]; then
  EXTRA_ARGS+=(--checkpoint_iterations "${CHECKPOINT_ITERATIONS[@]}")
elif [[ "$SAVE_FINAL_CHECKPOINT" == "1" ]]; then
  EXTRA_ARGS+=(--checkpoint_iterations "$ITERATIONS")
fi
if [[ -n "$DENSIFY_UNTIL_ITER" ]]; then
  EXTRA_ARGS+=(--densify_until_iter "$DENSIFY_UNTIL_ITER")
fi
if [[ -n "$DENSIFY_FROM_ITER" ]]; then
  EXTRA_ARGS+=(--densify_from_iter "$DENSIFY_FROM_ITER")
fi
if [[ -n "$DENSIFICATION_INTERVAL" ]]; then
  EXTRA_ARGS+=(--densification_interval "$DENSIFICATION_INTERVAL")
fi
if [[ -n "$OPACITY_RESET_INTERVAL" ]]; then
  EXTRA_ARGS+=(--opacity_reset_interval "$OPACITY_RESET_INTERVAL")
fi
if [[ -n "$PERCENT_DENSE" ]]; then
  EXTRA_ARGS+=(--percent_dense "$PERCENT_DENSE")
fi

if [[ -z "$TRAIN_GUI_PORT" ]]; then
  TRAIN_GUI_PORT="$(python3 - <<'PY'
import socket
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
)"
fi

EXTRA_ARGS+=(--ip "$TRAIN_GUI_IP" --port "$TRAIN_GUI_PORT")

echo "===== Train 3DGS: scene=$SCENE iterations=$ITERATIONS antialiasing=$ANTIALIASING exposure=$EXPOSURE_COMP ====="
echo "SOURCE_MODE=$SOURCE_MODE"
echo "SOURCE_DIR=$SOURCE_DIR"
echo "TRAIN_GUI_IP=$TRAIN_GUI_IP"
echo "TRAIN_GUI_PORT=$TRAIN_GUI_PORT"
echo "LOW_VRAM_PROFILE=$LOW_VRAM_PROFILE"
echo "RESOLUTION=$RESOLUTION"
echo "DENSIFY_UNTIL_ITER=${DENSIFY_UNTIL_ITER:-unset}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset}"
echo "SAVE_ITERATIONS=${SAVE_ITERATIONS[*]}"
if [[ ${#CHECKPOINT_ITERATIONS[@]} -gt 0 ]]; then
  echo "CHECKPOINT_ITERATIONS=${CHECKPOINT_ITERATIONS[*]}"
fi
if [[ -n "$START_CHECKPOINT" ]]; then
  echo "Resume từ: $START_CHECKPOINT"
fi

if [[ -n "$PYTORCH_CUDA_ALLOC_CONF" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF
fi

python3 "$GS_REPO/train.py" \
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
