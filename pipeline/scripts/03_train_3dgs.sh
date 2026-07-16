#!/usr/bin/env bash
# Train 3D Gaussian Splatting cho 1 hoặc nhiều scene, dùng repo GỐC
# graphdeco-inria/gaussian-splatting (không tự viết lại trainer — quá nhiều chi
# tiết dễ sai: densification, adaptive density control, SH coefficients...).
#
# HƯỚNG ĐI MIP-SPLATTING (port từ nhánh feature/mip-splatting round 1, xem
# WORKLOG.md): repo Inria GỐC (từ bản cập nhật 10/2024, commit đã pin bên dưới)
# đã TÍCH HỢP SẴN đúng "EWA Filter" của Mip-Splatting làm cờ `--antialiasing`
# (đối chiếu trực tiếp README.md + gaussian_renderer/__init__.py của repo thật,
# không suy đoán) — KHÔNG cần clone riêng autonomousvision/mip-splatting hay đổi
# rasterizer. Cùng bản này cũng có sẵn depth regularization (`--depths`) và
# exposure compensation (`--train_test_exp`) — cả 3 hướng giờ chỉ còn là bật cờ
# + (với depth) chuẩn bị depth map trước, không cần viết lại loss/model.
#
# Cài đặt 1 lần (máy có GPU CUDA, chạy trước khi dùng script này):
#   git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
#   cd gaussian-splatting
#   git checkout 54c035f7834b564019656c3e3fcc3646292f727d   # PIN commit đã xác nhận có antialiasing/depth/exposure
#   git submodule update --init --recursive                  # re-sync submodule đúng theo commit vừa checkout
#   conda env create --file environment.yml   # hoặc tự pip install theo requirements.txt của repo
#   conda activate gaussian_splatting
#
# Set biến môi trường GS_REPO trỏ tới thư mục clone ở trên trước khi chạy script này:
#   export GS_REPO=/path/to/gaussian-splatting
#
# Cách dùng:
#   ./03_train_3dgs.sh HCM0421                 # train 1 scene (mặc định: antialiasing BẬT)
#   ./03_train_3dgs.sh HCM0421 HCM0539 chair   # train nhiều scene liên tiếp
#   ITERATIONS=15000 ./03_train_3dgs.sh HCM0421   # đổi số iteration (mặc định 30000 của repo)
#   PROGRESS_INTERVAL=30 ./03_train_3dgs.sh HCM0421  # in tiến độ mỗi 30s thay vì 60s mặc định
#   ANTIALIASING=0 ./03_train_3dgs.sh HCM0421     # tắt antialiasing để A/B so với bản có (mặc định BẬT)
#   DEPTH_PRIOR=1 ./03_train_3dgs.sh HCM0421      # bật depth regularization — CẦN chạy
#                                                  # 08_generate_depth_priors.py cho scene này TRƯỚC
#                                                  # (tạo work/<scene>/colmap/dense/depths_any/ +
#                                                  # sparse/0/depth_params.json), không có thì tự bỏ qua + cảnh báo
#   EXPOSURE_COMP=1 ./03_train_3dgs.sh HCM0421    # bật exposure/appearance compensation (--train_test_exp,
#                                                  # đã đối chiếu source: an toàn dùng chung với KHÔNG --eval,
#                                                  # không ảnh hưởng loss vòng lặp chính — chỉ tối ưu thêm
#                                                  # affine exposure/ảnh train, lúc render pose mới vẫn bỏ qua
#                                                  # exposure vì pose đó không có exposure đã học, xem 04_render_test_poses.py)
#
# Nếu bị "CUDA out of memory" (hay gặp với scene nhiều chi tiết mảnh — dây cáp,
# khung thép BTS — vì số Gaussian sinh ra qua densify tăng rất nhanh), thử lần
# lượt theo thứ tự (mỗi lần giảm 1 mức, không cần giảm hết cùng lúc):
#   1) Không cần làm gì — script đã tự set PYTORCH_CUDA_ALLOC_CONF để giảm phân
#      mảnh bộ nhớ (đúng như gợi ý trong thông báo lỗi gốc của PyTorch).
#   2) SH_DEGREE=2 ./03_train_3dgs.sh HCM0421          (giảm dữ liệu màu/Gaussian, ảnh hưởng chất lượng ít)
#   3) DENSIFY_GRAD_THRESHOLD=0.0004 ./03_train_3dgs.sh HCM0421   (hạn chế sinh thêm Gaussian, mặc định repo 0.0002)
#   4) RESOLUTION=2 ./03_train_3dgs.sh HCM0421          (train ở nửa độ phân giải, giảm mạnh nhất nhưng ảnh hưởng chi tiết)
#   Có thể kết hợp nhiều biến cùng lúc, vd: SH_DEGREE=2 DENSIFY_GRAD_THRESHOLD=0.0004 ./03_train_3dgs.sh HCM0421
#
# Input mong đợi: pipeline/work/<scene>/colmap/dense/{images/,sparse/0/}
#                 (do 01_run_colmap.py tạo ra)
# Output: pipeline/work/<scene>/gs_model/point_cloud/iteration_<N>/point_cloud.ply
#         (có checkpoint giữa chừng ở 7000/15000 — nếu train bị crash muộn hơn,
#          vẫn dùng được model ở checkpoint gần nhất thay vì mất trắng)
#
# Dọn đĩa giữa các scene: sau khi train xong 1 scene, script tự xoá
# colmap/dense/images/ của scene đó (bản ảnh full-res undistort chỉ train.py
# cần — 04_render_test_poses.py/05_eval_metrics.py/06_package_submission.py
# không đụng tới, chỉ cần point_cloud.ply). Quan trọng khi chạy nhiều scene
# trong 1 lệnh (vd Bước 7/8): nếu không xoá, dữ liệu dense của các scene TRƯỚC
# vẫn nằm nguyên trên đĩa cộng dồn tới khi hết dung lượng giữa chừng (đã từng
# gặp "OSError: No space left on device" ngay tại checkpoint 15000 của scene
# thứ 4 trong 1 lần chạy thật). Set CLEANUP_DENSE_IMAGES=0 nếu muốn giữ lại để
# debug COLMAP sau này (vd nghi ngờ ảnh input sai).
#
# Tập trung train vào 1 vùng nhỏ (vd ăn-ten, CHỈ scene BTS) — xem
# pipeline/scripts/07_build_antenna_weights.py và apply_antenna_patch.py. Set
# ANTENNA_FOCUS=1 để bật: với MỖI scene, nếu có sẵn
# pipeline/work/<scene>/antenna_weights.json thì tự truyền --antenna_weights_json
# cho train.py (yêu cầu đã chạy apply_antenna_patch.py trên $GS_REPO trước, script
# tự kiểm tra và báo lỗi rõ nếu chưa vá). Scene nào chưa có file đó thì train bình
# thường (không lỗi cả loop). ANTENNA_WEIGHT (tuỳ chọn) ghi đè hệ số nhân loss.
# LƯU Ý TƯƠNG THÍCH: apply_antenna_patch.py được viết/test trên 1 bản train.py CŨ
# hơn commit đã pin ở trên (bản đó chưa có antialiasing/depth-reg/exposure/fused_ssim/
# sparse_adam) — patch vá theo ngữ cảnh dòng lệnh cụ thể nên CÓ THỂ không áp được sạch
# (hoặc áp sai chỗ) lên commit mới. Nếu cần dùng ANTENNA_FOCUS=1 CÙNG LÚC với
# antialiasing/depth-prior, hãy tự kiểm tra lại `apply_antenna_patch.py --gs_repo
# "$GS_REPO"` chạy sạch không lỗi trước, đừng mặc định nó vẫn đúng.

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
CLEANUP_DENSE_IMAGES="${CLEANUP_DENSE_IMAGES:-1}"
ANTENNA_FOCUS="${ANTENNA_FOCUS:-0}"
ANTIALIASING="${ANTIALIASING:-1}"
DEPTH_PRIOR="${DEPTH_PRIOR:-0}"
EXPOSURE_COMP="${EXPOSURE_COMP:-0}"

if [[ $# -eq 0 ]]; then
  echo "Cách dùng: $0 <scene1> [scene2 ...]" >&2
  exit 1
fi

if [[ "$ANTENNA_FOCUS" == "1" ]] && ! grep -q "antenna_weights_json" "$GS_REPO/train.py"; then
  echo "Lỗi: ANTENNA_FOCUS=1 nhưng $GS_REPO/train.py chưa được vá — chạy trước:" >&2
  echo "  python apply_antenna_patch.py --gs_repo \"$GS_REPO\"" >&2
  exit 1
fi

if [[ "$ANTIALIASING" == "1" ]] && ! grep -q "antialiasing" "$GS_REPO/arguments/__init__.py" 2>/dev/null; then
  echo "Lỗi: ANTIALIASING=1 nhưng \$GS_REPO có vẻ là bản clone CŨ (trước 10/2024, chưa có cờ" >&2
  echo "  --antialiasing). Checkout đúng commit đã pin (xem comment đầu file này) rồi thử lại," >&2
  echo "  hoặc set ANTIALIASING=0 nếu cố ý muốn train bản không chống alias để so sánh." >&2
  exit 1
fi
if [[ "$DEPTH_PRIOR" == "1" ]] && ! grep -q "depth_l1_weight" "$GS_REPO/train.py" 2>/dev/null; then
  echo "Lỗi: DEPTH_PRIOR=1 nhưng \$GS_REPO chưa có depth regularization (bản clone cũ) — checkout" >&2
  echo "  đúng commit đã pin rồi thử lại." >&2
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

  # Cảnh báo sớm nếu đĩa sắp hết TRƯỚC KHI đâm đầu vào train (có thể mất vài
  # tiếng) — tốt hơn là để nó chết giữa chừng lúc lưu checkpoint như đã từng
  # gặp. Ngưỡng 5GB là ước lượng an toàn (1 checkpoint point_cloud.ply có thể
  # nặng cỡ vài trăm MB tới hơn 1GB tuỳ số Gaussian sau densify).
  AVAIL_KB=$(df -Pk "$PIPELINE_DIR" | tail -1 | awk '{print $4}')
  AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
  echo "[$SCENE] Đĩa còn trống: ${AVAIL_GB}GB"
  if [[ "$AVAIL_GB" -lt 5 ]]; then
    echo "[LỖI] Đĩa còn dưới 5GB trước khi train $SCENE — dừng lại để tránh hỏng notebook giữa chừng." >&2
    echo "       Dọn bớt (vd rm -rf pipeline/work/<scene cũ>/colmap/dense/images) rồi chạy lại." >&2
    exit 1
  fi

  ANTENNA_ARGS=()
  ANTENNA_JSON="$PIPELINE_DIR/work/$SCENE/antenna_weights.json"
  if [[ "$ANTENNA_FOCUS" == "1" ]]; then
    if [[ -f "$ANTENNA_JSON" ]]; then
      ANTENNA_ARGS+=(--antenna_weights_json "$ANTENNA_JSON")
      if [[ -n "${ANTENNA_WEIGHT:-}" ]]; then
        ANTENNA_ARGS+=(--antenna_weight "$ANTENNA_WEIGHT")
      fi
      echo "  [antenna-focus] $SCENE: dùng $ANTENNA_JSON"
    else
      echo "  [antenna-focus] $SCENE: không có $ANTENNA_JSON — train bình thường (chạy 07_build_antenna_weights.py trước nếu muốn bật)."
    fi
  fi

  MIP_ARGS=()
  if [[ "$ANTIALIASING" == "1" ]]; then
    MIP_ARGS+=(--antialiasing)
  fi

  DEPTH_DIR="$SOURCE_DIR/depths_any"
  DEPTH_PARAMS="$SOURCE_DIR/sparse/0/depth_params.json"
  if [[ "$DEPTH_PRIOR" == "1" ]]; then
    if [[ -d "$DEPTH_DIR" && -f "$DEPTH_PARAMS" ]]; then
      MIP_ARGS+=(--depths depths_any)
      echo "  [depth-prior] $SCENE: dùng $DEPTH_DIR + $DEPTH_PARAMS"
    else
      echo "  [depth-prior] $SCENE: THIẾU $DEPTH_DIR hoặc $DEPTH_PARAMS — train KHÔNG depth prior" \
           "(chạy 08_generate_depth_priors.py $SCENE trước nếu muốn bật)."
    fi
  fi

  if [[ "$EXPOSURE_COMP" == "1" ]]; then
    MIP_ARGS+=(--train_test_exp)
  fi

  # train.py in progress bar (tqdm) qua hàng chục nghìn iteration — rất dài nếu
  # hiện hết ra console/notebook, nên vẫn redirect toàn bộ ra file log. Nhưng
  # chạy nền (&) rồi định kỳ lấy đúng số "hiện tại/ITERATIONS" cuối cùng trong
  # log để in 1 dòng gọn ra console — biết đang chạy tới đâu mà không bị spam.
  # Đổi tần suất bằng PROGRESS_INTERVAL=<giây> (mặc định 60s).
  echo "===== Train 3DGS: $SCENE ($ITERATIONS iterations, sh_degree=$SH_DEGREE, densify_grad_threshold=$DENSIFY_GRAD_THRESHOLD, antialiasing=$ANTIALIASING, depth_prior=$DEPTH_PRIOR, exposure_comp=$EXPOSURE_COMP) — log: $LOG_FILE ====="
  python "$GS_REPO/train.py" \
    -s "$SOURCE_DIR" \
    -m "$MODEL_DIR" \
    --iterations "$ITERATIONS" \
    --save_iterations "${SAVE_ITERATIONS[@]}" \
    --test_iterations "$ITERATIONS" \
    --sh_degree "$SH_DEGREE" \
    --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
    --resolution "$RESOLUTION" \
    "${ANTENNA_ARGS[@]}" \
    "${MIP_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
  # Không dùng --eval: ta muốn dùng TOÀN BỘ ảnh train/images/ để train (không
  # giữ lại phần nào làm test nội bộ của repo), vì việc tự đánh giá chất lượng
  # đã làm riêng trên public_set bằng 05_eval_metrics.py.
  TRAIN_PID=$!

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "${PROGRESS_INTERVAL:-60}"
    LAST_PROGRESS=$(grep -oE "[0-9]+/${ITERATIONS}" "$LOG_FILE" 2>/dev/null | tail -1 || true)
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

  # `antialiasing` là field của PipelineParams trong repo Inria gốc, còn cfg_args
  # mà train.py tự ghi CHỈ chứa ModelParams (xem train.py::training(), dòng
  # `tb_writer = prepare_output_and_logger(dataset)` với dataset=lp.extract(args))
  # -> cfg_args KHÔNG BAO GIỜ có field antialiasing, dù có bật --antialiasing hay
  # không (đã đối chiếu trực tiếp source tại commit đã pin). 04_render_test_poses.py
  # từng tự tin "auto-detect" antialiasing từ cfg_args để khớp train/render, nhưng
  # vì field đó luôn vắng mặt nên nó luôn ngầm định antialiasing=False bất kể lúc
  # train đã bật hay chưa — gây lệch train/render (Gaussian train ra opacity đã bù
  # EWA nhưng render lại không bù) mà KHÔNG hề báo lỗi, chỉ âm thầm ra ảnh kém hơn.
  # Ghi lại đúng giá trị thật đã dùng lúc train ra 1 file riêng để render đọc lại.
  cat > "$MODEL_DIR/pipeline_train_flags.json" <<EOF
{"antialiasing": $( [[ "$ANTIALIASING" == "1" ]] && echo true || echo false ), "depth_prior": $( [[ "$DEPTH_PRIOR" == "1" ]] && echo true || echo false ), "exposure_comp": $( [[ "$EXPOSURE_COMP" == "1" ]] && echo true || echo false ), "antenna_focus": $( [[ "$ANTENNA_FOCUS" == "1" ]] && echo true || echo false )}
EOF

  if [[ "$CLEANUP_DENSE_IMAGES" == "1" && -d "$SOURCE_DIR/images" ]]; then
    FREED_KB=$(du -sk "$SOURCE_DIR/images" 2>/dev/null | awk '{print $1}')
    rm -rf "$SOURCE_DIR/images"
    echo "  [dọn đĩa] Đã xoá $SOURCE_DIR/images (~$((FREED_KB / 1024))MB, không cần cho render/eval/package) — giữ lại sparse/0."
  fi
  if [[ "$CLEANUP_DENSE_IMAGES" == "1" && -d "$DEPTH_DIR" ]]; then
    rm -rf "$DEPTH_DIR"
    echo "  [dọn đĩa] Đã xoá $DEPTH_DIR (depth map chỉ cần lúc train, không cần cho render/eval/package)."
  fi
done
