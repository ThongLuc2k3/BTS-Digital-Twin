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
#
# Input mong đợi: pipeline/work/<scene>/colmap/dense/{images/,sparse/0/}
#                 (do 01_run_colmap.py tạo ra)
# Output: pipeline/work/<scene>/gs_model/point_cloud/iteration_<N>/point_cloud.ply

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
ITERATIONS="${ITERATIONS:-30000}"

if [[ $# -eq 0 ]]; then
  echo "Cách dùng: $0 <scene1> [scene2 ...]" >&2
  exit 1
fi

for SCENE in "$@"; do
  SOURCE_DIR="$PIPELINE_DIR/work/$SCENE/colmap/dense"
  MODEL_DIR="$PIPELINE_DIR/work/$SCENE/gs_model"
  LOG_FILE="$PIPELINE_DIR/work/$SCENE/train.log"

  if [[ ! -d "$SOURCE_DIR/sparse/0" ]]; then
    echo "[BỎ QUA] $SCENE: chưa thấy $SOURCE_DIR/sparse/0 — chạy 01_run_colmap.py --scene $SCENE trước." >&2
    continue
  fi

  # train.py in progress bar (tqdm) qua hàng chục nghìn iteration — rất dài nếu
  # hiện hết ra console/notebook. Redirect toàn bộ ra file log, notebook chỉ
  # thấy 2 dòng (bắt đầu/kết thúc) mỗi scene. Muốn xem tiến độ lúc đang chạy:
  # mở 1 cell khác gõ  !tail -n 30 "<LOG_FILE>"
  echo "===== Train 3DGS: $SCENE ($ITERATIONS iterations) — log: $LOG_FILE ====="
  set +e
  python "$GS_REPO/train.py" \
    -s "$SOURCE_DIR" \
    -m "$MODEL_DIR" \
    --iterations "$ITERATIONS" \
    --save_iterations "$ITERATIONS" \
    --test_iterations "$ITERATIONS" \
    > "$LOG_FILE" 2>&1
  # Không dùng --eval: ta muốn dùng TOÀN BỘ ảnh train/images/ để train (không
  # giữ lại phần nào làm test nội bộ của repo), vì việc tự đánh giá chất lượng
  # đã làm riêng trên public_set bằng 05_eval_metrics.py.
  STATUS=$?
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "[LỖI] Train thất bại cho $SCENE (exit $STATUS) — 50 dòng cuối log:" >&2
    tail -n 50 "$LOG_FILE" >&2
    exit $STATUS
  fi
  echo "-> Xong $SCENE. Model: $MODEL_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
done
