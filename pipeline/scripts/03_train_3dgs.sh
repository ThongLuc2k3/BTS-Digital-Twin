#!/usr/bin/env bash
# Train 3D Gaussian Splatting cho 1 hoặc nhiều scene, dùng repo GỐC
# graphdeco-inria/gaussian-splatting (không tự viết lại trainer — quá nhiều chi
# tiết dễ sai: densification, adaptive density control, SH coefficients...).
#
# Cài đặt 1 lần (máy có GPU CUDA, chạy trước khi dùng script này):
#   git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
#   cd gaussian-splatting
#   conda env create --file environment.yml   # hoặc tự pip install theo requirements.txt của repo
#   conda activate gaussian_splatting
#
# Set biến môi trường GS_REPO trỏ tới thư mục clone ở trên trước khi chạy script này:
#   export GS_REPO=/path/to/gaussian-splatting
#
# Cách dùng:
#   ./03_train_3dgs.sh HCM0181                 # train 1 scene
#   ./03_train_3dgs.sh HCM0181 HCM0193 hcm0031 # train nhiều scene liên tiếp
#   ITERATIONS=15000 ./03_train_3dgs.sh HCM0181   # đổi số iteration (mặc định 30000 của repo)
#   PROGRESS_INTERVAL=30 ./03_train_3dgs.sh HCM0181  # in tiến độ mỗi 30s thay vì 60s mặc định
#
# Nếu bị "CUDA out of memory" (hay gặp với scene nhiều chi tiết mảnh — dây cáp,
# khung thép BTS — vì số Gaussian sinh ra qua densify tăng rất nhanh), thử lần
# lượt theo thứ tự (mỗi lần giảm 1 mức, không cần giảm hết cùng lúc):
#   1) Không cần làm gì — script đã tự set PYTORCH_CUDA_ALLOC_CONF để giảm phân
#      mảnh bộ nhớ (đúng như gợi ý trong thông báo lỗi gốc của PyTorch).
#   2) SH_DEGREE=2 ./03_train_3dgs.sh HCM0181          (giảm dữ liệu màu/Gaussian, ảnh hưởng chất lượng ít)
#   3) DENSIFY_GRAD_THRESHOLD=0.0004 ./03_train_3dgs.sh HCM0181   (hạn chế sinh thêm Gaussian, mặc định repo 0.0002)
#   4) RESOLUTION=2 ./03_train_3dgs.sh HCM0181          (train ở nửa độ phân giải, giảm mạnh nhất nhưng ảnh hưởng chi tiết)
#   Có thể kết hợp nhiều biến cùng lúc, vd: SH_DEGREE=2 DENSIFY_GRAD_THRESHOLD=0.0004 ./03_train_3dgs.sh HCM0181
#
# Input mong đợi: pipeline/work/<scene>/colmap/dense/{images/,sparse/0/}
#                 (do 01_run_colmap.py tạo ra)
# Output: pipeline/work/<scene>/gs_model/point_cloud/iteration_<N>/point_cloud.ply
#         (có checkpoint giữa chừng ở 7000/15000 — nếu train bị crash muộn hơn,
#          vẫn dùng được model ở checkpoint gần nhất thay vì mất trắng)

set -euo pipefail

if [[ -z "${GS_REPO:-}" ]]; then
  echo "Lỗi: chưa set biến môi trường GS_REPO (đường dẫn tới repo graphdeco-inria/gaussian-splatting đã clone)." >&2
  exit 1
fi
if [[ ! -f "$GS_REPO/train.py" ]]; then
  echo "Lỗi: không thấy $GS_REPO/train.py — kiểm tra lại GS_REPO." >&2
  exit 1
fi

# Giảm lỗi CUDA OOM do phân mảnh bộ nhớ (khuyến nghị chính thức của PyTorch khi
# gặp "reserved but unallocated memory is large") — không đánh đổi chất lượng,
# nên bật mặc định luôn, không cần người dùng tự nhớ set.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
ITERATIONS="${ITERATIONS:-30000}"
SH_DEGREE="${SH_DEGREE:-3}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
RESOLUTION="${RESOLUTION:--1}"

if [[ $# -eq 0 ]]; then
  echo "Cách dùng: $0 <scene1> [scene2 ...]" >&2
  exit 1
fi

# Checkpoint giữa chừng ở 7000/15000 (nếu ITERATIONS đủ lớn) để không mất trắng
# nếu crash muộn hơn (vd OOM ở densify) — trước đây chỉ lưu đúng lúc kết thúc.
SAVE_ITERATIONS=()
for v in 7000 15000 "$ITERATIONS"; do
  if [[ "$v" -le "$ITERATIONS" ]]; then
    SAVE_ITERATIONS+=("$v")
  fi
done
SAVE_ITERATIONS=($(printf "%s\n" "${SAVE_ITERATIONS[@]}" | awk '!seen[$0]++'))

for SCENE in "$@"; do
  SOURCE_DIR="$PIPELINE_DIR/work/$SCENE/colmap/dense"
  MODEL_DIR="$PIPELINE_DIR/work/$SCENE/gs_model"
  LOG_FILE="$PIPELINE_DIR/work/$SCENE/03_train_3dgs.log"

  if [[ ! -d "$SOURCE_DIR/sparse/0" ]]; then
    echo "[BỎ QUA] $SCENE: chưa thấy $SOURCE_DIR/sparse/0 — chạy 01_run_colmap.py --scene $SCENE trước." >&2
    continue
  fi

  # train.py in progress bar (tqdm) qua hàng chục nghìn iteration — rất dài nếu
  # hiện hết ra console/notebook, nên vẫn redirect toàn bộ ra file log. Nhưng
  # chạy nền (&) rồi định kỳ lấy đúng số "hiện tại/ITERATIONS" cuối cùng trong
  # log để in 1 dòng gọn ra console — biết đang chạy tới đâu mà không bị spam.
  # Đổi tần suất bằng PROGRESS_INTERVAL=<giây> (mặc định 60s).
  echo "===== Train 3DGS: $SCENE ($ITERATIONS iterations, sh_degree=$SH_DEGREE, densify_grad_threshold=$DENSIFY_GRAD_THRESHOLD) — log: $LOG_FILE ====="
  python "$GS_REPO/train.py" \
    -s "$SOURCE_DIR" \
    -m "$MODEL_DIR" \
    --iterations "$ITERATIONS" \
    --save_iterations "${SAVE_ITERATIONS[@]}" \
    --test_iterations "$ITERATIONS" \
    --sh_degree "$SH_DEGREE" \
    --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
    --resolution "$RESOLUTION" \
    > "$LOG_FILE" 2>&1 &
  # Không dùng --eval: ta muốn dùng TOÀN BỘ ảnh train/images/ để train (không
  # giữ lại phần nào làm test nội bộ của repo), vì việc tự đánh giá chất lượng
  # đã làm riêng trên public_set bằng 05_eval_metrics.py.
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
done
