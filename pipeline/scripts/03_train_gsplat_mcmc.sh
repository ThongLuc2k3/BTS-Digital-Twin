#!/usr/bin/env bash
# Train 3D Gaussian Splatting bằng gsplat (nerfstudio-project/gsplat) với chiến
# lược MCMC densification (3DGS-MCMC, Kheradmand et al. 2024) thay cho
# densify/prune mặc định của repo Inria — xem Kết quả/prompt_76diem.md mục
# "gsplat + MCMC densification" (cú đặt cược chính của lộ trình).
#
# VÌ SAO: MCMC dùng SỐ GAUSSIAN CỐ ĐỊNH (--strategy.cap_max) do mình đặt, không
# tăng vô hạn theo densify_grad_threshold như repo Inria -> hết vĩnh viễn CUDA
# OOM (nguyên nhân đã từng OOM ở nhánh feature/depth-anything-v2, buộc phải
# tăng densify_grad_threshold — cái hack đó không cần nữa). Gaussian "chết" ở
# nền được tự động relocate vào vùng lỗi cao (ăng-ten/cáp) theo opacity, không
# cần khai báo hộp bao 3D thủ công như nhánh compact/compact-gaussian.
#
# QUAN TRỌNG — antialiasing: gsplat gọi antialiasing (Mip-Splatting EWA filter)
# là `--antialiased` (KHÔNG PHẢI `--antialiasing` như repo Inria) và nó ảnh
# hưởng cách rasterize (rasterize_mode="antialiased"/"classic" nội bộ). ĐÚNG
# BÀI HỌC từ bug 10 điểm trên nhánh Inria (xem 03_train_3dgs.sh): script này tự
# ghi lại giá trị THẬT đã dùng ra `pipeline_train_flags.json` ngay từ đầu, để
# 04_render_gsplat_test_poses.py không bao giờ phải "đoán" — không lặp lại kiểu
# lỗi train/render lệch cấu hình.
#
# Cài đặt 1 lần (máy có GPU CUDA):
#   pip install gsplat                                    # core rasterization lib
#   git clone https://github.com/nerfstudio-project/gsplat.git   # cần examples/simple_trainer.py
#   (không cần build lại rasterizer riêng — gsplat cài qua pip đã có sẵn CUDA kernel,
#    examples/ chỉ cần clone để lấy code trainer, KHÔNG cần build gì thêm trong đó)
#   pip install -r gsplat/examples/requirements.txt
#
# Set biến môi trường GSPLAT_REPO trỏ tới thư mục clone ở trên:
#   export GSPLAT_REPO=/path/to/gsplat
#
# Cách dùng:
#   ./03_train_gsplat_mcmc.sh HCM0181                  # train 1 scene, mặc định:
#                                                        #   cap_max=2000000, antialiased BẬT
#   CAP_MAX=1000000 ./03_train_gsplat_mcmc.sh HCM0181  # giảm budget Gaussian nếu vẫn OOM
#   ANTIALIASED=0 ./03_train_gsplat_mcmc.sh HCM0181    # tắt antialiasing để A/B so sánh
#   MAX_STEPS=15000 ./03_train_gsplat_mcmc.sh HCM0181  # rút ngắn để A/B rẻ (xem giao thức
#                                                        #   kiểm chứng rẻ trong prompt_76diem.md)
#
# Input mong đợi: pipeline/work/<scene>/colmap/dense/{images/,sparse/0/}
#                 (do 01_run_colmap.py tạo ra — CÙNG input với nhánh Inria, không
#                  cần chạy lại COLMAP)
# Output: pipeline/work/<scene>/gsplat_model/ckpts/ckpt_<step>_rank0.pt
#         + pipeline/work/<scene>/gsplat_model/pipeline_train_flags.json
#
# Nếu vẫn OOM dù đã dùng MCMC (ít khả năng hơn nhiều so với repo Inria, nhưng
# scene BTS nhiều chi tiết mảnh vẫn có thể cần): giảm CAP_MAX trước tiên (đây là
# đòn bẩy chính, không như densify_grad_threshold của Inria không có giới hạn
# trên), sau đó mới cân nhắc SH_DEGREE=2.

set -euo pipefail

if [[ -z "${GSPLAT_REPO:-}" ]]; then
  echo "Lỗi: chưa set biến môi trường GSPLAT_REPO (đường dẫn tới repo nerfstudio-project/gsplat đã clone)." >&2
  echo "  export GSPLAT_REPO=/path/to/gsplat" >&2
  exit 1
fi
EXAMPLES_DIR="$GSPLAT_REPO/examples"
TRAINER="$EXAMPLES_DIR/simple_trainer.py"
if [[ ! -f "$TRAINER" ]]; then
  echo "Lỗi: không thấy $TRAINER — kiểm tra lại GSPLAT_REPO (cần clone repo, không chỉ pip install gsplat)." >&2
  exit 1
fi
# QUAN TRỌNG: simple_trainer.py dùng import tương đối trong package examples/
# (vd `from datasets.colmap import Parser`) — CHẠY TỪ BÊN TRONG thư mục
# examples/ (đúng cách chính gsplat ghi trong docstring: `cd examples && python
# simple_trainer.py ...`), KHÔNG gọi bằng đường dẫn tuyệt đối từ thư mục khác,
# nếu không sẽ lỗi ModuleNotFoundError ngay khi import.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
MAX_STEPS="${MAX_STEPS:-30000}"
SH_DEGREE="${SH_DEGREE:-3}"
CAP_MAX="${CAP_MAX:-2000000}"
ANTIALIASED="${ANTIALIASED:-1}"
BILATERAL_GRID="${BILATERAL_GRID:-1}"
DATA_FACTOR="${DATA_FACTOR:-1}"

if [[ $# -eq 0 ]]; then
  echo "Cách dùng: $0 <scene1> [scene2 ...]" >&2
  exit 1
fi

# Kiểm tra sớm tên cờ CLI thật sự tồn tại trên bản gsplat đang cài — API của
# gsplat/tyro có thể đổi giữa các version, KHÔNG giả định mù mà báo lỗi rõ nếu
# thiếu (đúng tinh thần kiểm tra ANTIALIASING trong 03_train_3dgs.sh).
HELP_TEXT="$(cd "$EXAMPLES_DIR" && python simple_trainer.py mcmc --help 2>&1 || true)"
for flag in "data_dir" "result_dir" "max_steps" "sh_degree" "antialiased" "cap_max"; do
  if ! grep -q -- "$flag" <<< "$HELP_TEXT"; then
    echo "Lỗi: không thấy cờ '--$flag' trong 'python $TRAINER mcmc --help'." >&2
    echo "  API của gsplat có thể đã đổi so với lúc viết script này — kiểm tra thủ công:" >&2
    echo "  python $TRAINER mcmc --help | less" >&2
    exit 1
  fi
done
HAS_BILATERAL=1
if [[ "$BILATERAL_GRID" == "1" ]] && ! grep -q -- "post_processing" <<< "$HELP_TEXT"; then
  echo "[CẢNH BÁO] Không thấy cờ '--post_processing' — bản gsplat này có thể không hỗ trợ" >&2
  echo "  bilateral grid qua cờ đó. Tắt bilateral grid cho lần chạy này, chỉ dùng antialiased+MCMC." >&2
  HAS_BILATERAL=0
fi

for SCENE in "$@"; do
  SOURCE_DIR="$PIPELINE_DIR/work/$SCENE/colmap/dense"
  MODEL_DIR="$PIPELINE_DIR/work/$SCENE/gsplat_model"
  LOG_FILE="$PIPELINE_DIR/work/$SCENE/03_train_gsplat_mcmc.log"

  if [[ ! -d "$SOURCE_DIR/sparse/0" ]]; then
    echo "[BỎ QUA] $SCENE: chưa thấy $SOURCE_DIR/sparse/0 — chạy 01_run_colmap.py --scene $SCENE trước." >&2
    continue
  fi

  AVAIL_KB=$(df -Pk "$PIPELINE_DIR" | tail -1 | awk '{print $4}')
  AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
  echo "[$SCENE] Đĩa còn trống: ${AVAIL_GB}GB"
  if [[ "$AVAIL_GB" -lt 5 ]]; then
    echo "[LỖI] Đĩa còn dưới 5GB trước khi train $SCENE — dừng lại để tránh hỏng notebook giữa chừng." >&2
    exit 1
  fi

  EXTRA_ARGS=()
  if [[ "$ANTIALIASED" == "1" ]]; then
    EXTRA_ARGS+=(--antialiased)
  fi
  if [[ "$BILATERAL_GRID" == "1" && "$HAS_BILATERAL" == "1" ]]; then
    EXTRA_ARGS+=(--post_processing bilateral_grid)
  fi

  echo "===== Train gsplat+MCMC: $SCENE (max_steps=$MAX_STEPS, sh_degree=$SH_DEGREE, cap_max=$CAP_MAX, antialiased=$ANTIALIASED, bilateral_grid=$( [[ "$BILATERAL_GRID" == "1" && "$HAS_BILATERAL" == "1" ]] && echo 1 || echo 0)) — log: $LOG_FILE ====="
  (
    cd "$EXAMPLES_DIR" && python simple_trainer.py mcmc \
      --data_dir "$SOURCE_DIR" \
      --data_factor "$DATA_FACTOR" \
      --result_dir "$MODEL_DIR" \
      --max_steps "$MAX_STEPS" \
      --sh_degree "$SH_DEGREE" \
      --strategy.cap_max "$CAP_MAX" \
      "${EXTRA_ARGS[@]}"
  ) > "$LOG_FILE" 2>&1 &
  TRAIN_PID=$!

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "${PROGRESS_INTERVAL:-60}"
    LAST_PROGRESS=$(grep -oE "[0-9]+/${MAX_STEPS}" "$LOG_FILE" 2>/dev/null | tail -1)
    if [[ -n "$LAST_PROGRESS" ]]; then
      echo "  [$SCENE] tiến độ: $LAST_PROGRESS steps"
    fi
  done

  set +e
  wait "$TRAIN_PID"
  STATUS=$?
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "[LỖI] Train thất bại cho $SCENE (exit $STATUS) — 50 dòng cuối log:" >&2
    tail -n 50 "$LOG_FILE" >&2
    exit $STATUS
  fi
  LAST_CKPT=$(ls -t "$MODEL_DIR"/ckpts/ckpt_*_rank0.pt 2>/dev/null | head -1 || true)
  echo "-> Xong $SCENE. Checkpoint: ${LAST_CKPT:-KHÔNG TÌM THẤY — kiểm tra log}"

  # Ghi lại đúng cấu hình THẬT đã dùng — để 04_render_gsplat_test_poses.py không
  # bao giờ phải đoán antialiased là gì (bài học trực tiếp từ bug 10 điểm ở
  # nhánh Inria, xem đầu file này).
  cat > "$MODEL_DIR/pipeline_train_flags.json" <<EOF
{"antialiased": $( [[ "$ANTIALIASED" == "1" ]] && echo true || echo false ), "bilateral_grid": $( [[ "$BILATERAL_GRID" == "1" && "$HAS_BILATERAL" == "1" ]] && echo true || echo false ), "sh_degree": $SH_DEGREE, "cap_max": $CAP_MAX}
EOF
done
