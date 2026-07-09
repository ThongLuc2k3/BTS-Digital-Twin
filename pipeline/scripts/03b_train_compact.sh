#!/usr/bin/env bash
# Train "Compact Gaussian" (Lee et al., arXiv:2311.13681, mục 3.1) cho 1 hoặc
# nhiều scene — bản song song với 03_train_3dgs.sh nhưng gọi
# pipeline/extra/train_compact.py (Gaussian Volume Mask + bảo vệ vùng chi tiết
# cao ăng-ten/RRU/cáp) thay vì train.py gốc.
#
# LƯU Ý QUAN TRỌNG: pipeline/extra/compact_gaussian.py và train_compact.py
# được viết để đặt CÙNG THƯ MỤC với train.py gốc trong $GS_REPO (vì
# train_compact.py làm `from compact_gaussian import ...`, và Python chỉ tìm
# thấy module cùng thư mục hoặc trong PYTHONPATH). Script này TỰ COPY 2 file đó
# vào $GS_REPO mỗi lần chạy (ghi đè, để luôn khớp bản mới nhất trong repo pipeline
# — không tự sửa tay file trong $GS_REPO).
#
# Cài đặt 1 lần (giống 03_train_3dgs.sh — xem chi tiết comment đầu file đó):
#   git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
#   cd gaussian-splatting && git checkout 54c035f7834b564019656c3e3fcc3646292f727d
#   pip install submodules/diff-gaussian-rasterization submodules/simple-knn
#   export GS_REPO=/path/to/gaussian-splatting
#
# Cách dùng:
#   ./03b_train_compact.sh HCM0181                 # train 1 scene
#   ./03b_train_compact.sh HCM0181 HCM0193 hcm0031 # train nhiều scene liên tiếp
#   ITERATIONS=15000 ./03b_train_compact.sh HCM0181
#   LAMBDA_MASK=1e-3 ./03b_train_compact.sh HCM0181   # nén mạnh hơn (mặc định 5e-4, xem train_compact.py)
#
# Cờ riêng của Compact Gaussian (forward thẳng xuống train_compact.py, xem
# --help của script đó hoặc comment đầu file để hiểu từng tham số):
#   LAMBDA_MASK            (mặc định 5e-4)  -> --lambda_mask
#   MASK_LR                 (mặc định 1e-2)  -> --mask_lr
#   MASK_PRUNE_FROM_ITER    (mặc định 1500)  -> --mask_prune_from_iter
#   MASK_PRUNE_INTERVAL     (mặc định 100)   -> --mask_prune_interval
#   HARD_PROTECT_DETAIL     (mặc định 1; "0" -> thêm --no_hard_protect, chỉ giảm
#                            trọng số loss cho vùng chi tiết cao thay vì bảo vệ cứng)
#
# Vùng chi tiết cao (ăng-ten/RRU/cáp) cần bảo vệ khỏi bị nén: nếu có sẵn
# pipeline/work/<scene>/detail_regions.json (dùng pipeline/extra/pick_detail_boxes.py
# để tạo — xem comment đầu file đó) thì TỰ truyền --detail_regions_json cho
# scene đó. Scene nào chưa có file này thì train bình thường (Compact Gaussian
# vẫn chạy, chỉ là không bảo vệ vùng nào riêng) — không lỗi cả loop.
#
# Input/Output/checkpoint/dọn đĩa: giống hệt 03_train_3dgs.sh, xem comment đầu
# file đó — chỉ khác log/model nằm ở 03b_train_compact.log (không đụng tới
# gs_model của 1 lần train vanilla trước đó cho cùng scene, vì MODEL_DIR khác
# tên: gs_model_compact).

set -euo pipefail

if [[ -z "${GS_REPO:-}" ]]; then
  echo "Lỗi: chưa set biến môi trường GS_REPO (đường dẫn tới repo graphdeco-inria/gaussian-splatting đã clone)." >&2
  exit 1
fi
if [[ ! -f "$GS_REPO/train.py" ]]; then
  echo "Lỗi: không thấy $GS_REPO/train.py — kiểm tra lại GS_REPO." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
EXTRA_DIR="$PIPELINE_DIR/extra"

if [[ ! -f "$EXTRA_DIR/compact_gaussian.py" || ! -f "$EXTRA_DIR/train_compact.py" ]]; then
  echo "Lỗi: không thấy $EXTRA_DIR/compact_gaussian.py hoặc train_compact.py — kiểm tra lại clone repo." >&2
  exit 1
fi
cp -f "$EXTRA_DIR/compact_gaussian.py" "$GS_REPO/compact_gaussian.py"
cp -f "$EXTRA_DIR/train_compact.py" "$GS_REPO/train_compact.py"
echo "Đã copy compact_gaussian.py + train_compact.py vào \$GS_REPO ($GS_REPO)."

# Giảm lỗi CUDA OOM do phân mảnh bộ nhớ — giống 03_train_3dgs.sh.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

ITERATIONS="${ITERATIONS:-30000}"
SH_DEGREE="${SH_DEGREE:-3}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
RESOLUTION="${RESOLUTION:--1}"
CLEANUP_DENSE_IMAGES="${CLEANUP_DENSE_IMAGES:-1}"
LAMBDA_MASK="${LAMBDA_MASK:-5e-4}"
MASK_LR="${MASK_LR:-1e-2}"
MASK_PRUNE_FROM_ITER="${MASK_PRUNE_FROM_ITER:-1500}"
MASK_PRUNE_INTERVAL="${MASK_PRUNE_INTERVAL:-100}"
HARD_PROTECT_DETAIL="${HARD_PROTECT_DETAIL:-1}"

if [[ $# -eq 0 ]]; then
  echo "Cách dùng: $0 <scene1> [scene2 ...]" >&2
  exit 1
fi

SAVE_ITERATIONS=()
for v in 7000 15000 "$ITERATIONS"; do
  if [[ "$v" -le "$ITERATIONS" ]]; then
    SAVE_ITERATIONS+=("$v")
  fi
done
SAVE_ITERATIONS=($(printf "%s\n" "${SAVE_ITERATIONS[@]}" | awk '!seen[$0]++'))

for SCENE in "$@"; do
  SOURCE_DIR="$PIPELINE_DIR/work/$SCENE/colmap/dense"
  MODEL_DIR="$PIPELINE_DIR/work/$SCENE/gs_model_compact"
  LOG_FILE="$PIPELINE_DIR/work/$SCENE/03b_train_compact.log"

  if [[ ! -d "$SOURCE_DIR/sparse/0" ]]; then
    echo "[BỎ QUA] $SCENE: chưa thấy $SOURCE_DIR/sparse/0 — chạy 01_run_colmap.py --scene $SCENE trước." >&2
    continue
  fi

  AVAIL_KB=$(df -Pk "$PIPELINE_DIR" | tail -1 | awk '{print $4}')
  AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
  echo "[$SCENE] Đĩa còn trống: ${AVAIL_GB}GB"
  if [[ "$AVAIL_GB" -lt 5 ]]; then
    echo "[LỖI] Đĩa còn dưới 5GB trước khi train $SCENE — dừng lại để tránh hỏng notebook giữa chừng." >&2
    echo "       Dọn bớt (vd rm -rf pipeline/work/<scene cũ>/colmap/dense/images) rồi chạy lại." >&2
    exit 1
  fi

  DETAIL_ARGS=()
  DETAIL_JSON="$PIPELINE_DIR/work/$SCENE/detail_regions.json"
  if [[ -f "$DETAIL_JSON" ]]; then
    DETAIL_ARGS+=(--detail_regions_json "$DETAIL_JSON")
    echo "  [detail-regions] $SCENE: dùng $DETAIL_JSON"
  else
    echo "  [detail-regions] $SCENE: không có $DETAIL_JSON — train bình thường (chạy pick_detail_boxes.py trước nếu muốn khai báo vùng chi tiết cao)."
  fi

  HARD_PROTECT_ARGS=()
  if [[ "$HARD_PROTECT_DETAIL" == "0" ]]; then
    HARD_PROTECT_ARGS+=(--no_hard_protect)
  fi

  echo "===== Train Compact Gaussian: $SCENE ($ITERATIONS iterations, sh_degree=$SH_DEGREE, densify_grad_threshold=$DENSIFY_GRAD_THRESHOLD, lambda_mask=$LAMBDA_MASK) — log: $LOG_FILE ====="
  python "$GS_REPO/train_compact.py" \
    -s "$SOURCE_DIR" \
    -m "$MODEL_DIR" \
    --iterations "$ITERATIONS" \
    --save_iterations "${SAVE_ITERATIONS[@]}" \
    --test_iterations "$ITERATIONS" \
    --sh_degree "$SH_DEGREE" \
    --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
    --resolution "$RESOLUTION" \
    --lambda_mask "$LAMBDA_MASK" \
    --mask_lr "$MASK_LR" \
    --mask_prune_from_iter "$MASK_PRUNE_FROM_ITER" \
    --mask_prune_interval "$MASK_PRUNE_INTERVAL" \
    "${HARD_PROTECT_ARGS[@]}" \
    "${DETAIL_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
  TRAIN_PID=$!

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "${PROGRESS_INTERVAL:-60}"
    LAST_PROGRESS=$(grep -oE "[0-9]+/${ITERATIONS}" "$LOG_FILE" 2>/dev/null | tail -1)
    if [[ -n "$LAST_PROGRESS" ]]; then
      echo "  [$SCENE] tiến độ: $LAST_PROGRESS iterations"
    fi
  done

  set +e
  wait "$TRAIN_PID"
  STATUS=$?
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "[LỖI] Train thất bại cho $SCENE (exit $STATUS) — 50 dòng cuối log:" >&2
    tail -n 50 "$LOG_FILE" >&2
    LAST_CKPT=$(ls -d "$MODEL_DIR"/point_cloud/iteration_* 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
    if [[ -n "$LAST_CKPT" ]]; then
      echo "[CỨU ĐƯỢC] Vẫn còn checkpoint gần nhất tại: $LAST_CKPT (dùng tạm để render nếu cần)." >&2
    fi
    exit $STATUS
  fi
  echo "-> Xong $SCENE. Model: $MODEL_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"

  if [[ "$CLEANUP_DENSE_IMAGES" == "1" && -d "$SOURCE_DIR/images" ]]; then
    FREED_KB=$(du -sk "$SOURCE_DIR/images" 2>/dev/null | awk '{print $1}')
    rm -rf "$SOURCE_DIR/images"
    echo "  [dọn đĩa] Đã xoá $SOURCE_DIR/images (~$((FREED_KB / 1024))MB, không cần cho render/eval/package) — giữ lại sparse/0."
  fi
done
