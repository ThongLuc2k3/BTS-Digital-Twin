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

  if [[ ! -d "$SOURCE_DIR/sparse/0" ]]; then
    echo "[BỎ QUA] $SCENE: chưa thấy $SOURCE_DIR/sparse/0 — chạy 01_run_colmap.py --scene $SCENE trước." >&2
    continue
  fi

  echo "===== Train 3DGS: $SCENE ($ITERATIONS iterations) ====="
  python "$GS_REPO/train.py" \
    -s "$SOURCE_DIR" \
    -m "$MODEL_DIR" \
    --iterations "$ITERATIONS" \
    --save_iterations "$ITERATIONS" \
    --test_iterations "$ITERATIONS"
  # Không dùng --eval: ta muốn dùng TOÀN BỘ ảnh train/images/ để train (không
  # giữ lại phần nào làm test nội bộ của repo), vì việc tự đánh giá chất lượng
  # đã làm riêng trên public_set bằng 05_eval_metrics.py.

  echo "-> Model: $MODEL_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"
done
