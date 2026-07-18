#!/usr/bin/env bash
# BACKUP/thử nghiệm — train 3DGS theo kiểu "coarse-to-fine": chạy nhiều giai đoạn,
# giai đoạn đầu ở độ phân giải THẤP (ít Gaussian sinh ra hơn, nhẹ VRAM), rồi
# resume (đúng nghĩa: load lại optimizer + Gaussian qua checkpoint .pth của repo
# gốc, KHÔNG phải train lại từ đầu) ở độ phân giải cao dần tới đúng độ phân giải
# yêu cầu ở giai đoạn cuối. Mục tiêu: giảm rủi ro CUDA OOM (đã gặp thật với
# DEPTH_PRIOR=1, xem WORKLOG.md 2026-07-18) mà vẫn ra checkpoint cuối cùng ở ĐÚNG
# độ phân giải gốc (không đánh đổi chất lượng vĩnh viễn như RESOLUTION=2 đơn
# giản của 03_train_3dgs.sh).
#
# KHÔNG PHẢI phương án chính — chỉ dùng nếu 03_train_3dgs.sh (kể cả đã hạ
# SH_DEGREE=2 + tăng DENSIFY_GRAD_THRESHOLD, xem comment ở đó) vẫn OOM. CHƯA
# test thật trên GPU (không có GPU cục bộ) — chỉ verify được cú pháp lệnh +
# logic ghép giai đoạn bằng cách đọc trực tiếp source thật của train.py (pin
# commit 54c035f7834b564019656c3e3fcc3646292f727d), KHÔNG suy đoán:
#   - `--resolution` nạp lại từ ảnh gốc MỖI LẦN chạy process (không bị "đông
#     cứng" trong checkpoint) -> đổi resolution giữa các giai đoạn hợp lệ.
#   - `--start_checkpoint <path>.pth` khôi phục ĐẦY ĐỦ (Gaussian + optimizer +
#     SỐ ITERATION đã chạy qua `first_iter`) — không phải chỉ load .ply tĩnh.
#   - `spatial_lr_scale` (ảnh hưởng learning rate vị trí) tính từ world-space
#     (kích thước cảnh 3D thật), KHÔNG phụ thuộc resolution ảnh -> an toàn khi
#     đổi resolution giữa các giai đoạn.
#   - `densify_until_iter` mặc định CỐ ĐỊNH 15000 (không tự co giãn theo
#     --iterations) — dùng số ITERATION TUYỆT ĐỐI xuyên suốt các giai đoạn
#     (nhờ first_iter resume đúng), nên hành vi densify vẫn nhất quán, không
#     bị "reset" mỗi giai đoạn.
#   - HẠN CHẾ ĐÃ BIẾT (chấp nhận được, không chặn): `restore()` gọi lại
#     `training_setup()` mỗi giai đoạn -> lịch giảm learning-rate vị trí
#     (xyz) và trọng số depth-loss bị tính lại theo `--iterations` MỚI của
#     từng giai đoạn thay vì 1 đường cong mượt xuyên suốt như chạy 1 lần duy
#     nhất — có thể gây dao động nhỏ ở ranh giới giai đoạn, không phải lỗi
#     cú pháp/crash, chỉ là khác biệt nhỏ so với train 1 mạch.
#
# Cách dùng (giống hệt biến môi trường của 03_train_3dgs.sh, cộng thêm 2 biến
# PROGRESSIVE_* bên dưới) — CHỈ nhận 1 scene mỗi lần chạy (khác 03_train_3dgs.sh
# nhận nhiều scene) để dễ theo dõi log từng giai đoạn:
#   GS_REPO=/path/to/gaussian-splatting \
#   DEPTH_PRIOR=1 ITERATIONS=15000 \
#   ./03_train_3dgs_progressive.sh HCM0421
#
#   PROGRESSIVE_RESOLUTIONS="4 2 1"     # mặc định — mỗi giai đoạn 1 giá trị --resolution
#                                        # (1/2/4/8 = chia độ phân giải gốc cho số đó,
#                ,                       #  1 = ĐÚNG độ phân giải gốc, xem utils/camera_utils.py)
#   PROGRESSIVE_FRACTIONS="0.3 0.6 1.0" # mặc định — % ITERATIONS mà mỗi giai đoạn
#                                        # kết thúc (số TUYỆT ĐỐI, tăng dần, số cuối PHẢI = 1.0)
#   (2 danh sách trên phải cùng số phần tử, cách nhau bằng dấu cách)
#
# Ví dụ đổi số giai đoạn/tỉ lệ:
#   PROGRESSIVE_RESOLUTIONS="8 4 2 1" PROGRESSIVE_FRACTIONS="0.2 0.4 0.7 1.0" \
#   DEPTH_PRIOR=1 ./03_train_3dgs_progressive.sh HCM0421
#
# ANTENNA_FOCUS CHƯA hỗ trợ ở script này (tổ hợp antenna-focus + progressive-resolution
# chưa nghĩ tới/chưa test — script sẽ báo lỗi rõ nếu ANTENNA_FOCUS=1).
#
# Output: giống hệt 03_train_3dgs.sh — pipeline/work/<scene>/gs_model/point_cloud/
# iteration_<N>/point_cloud.ply ở các mốc chuẩn (7000/15000/ITERATIONS), cộng thêm
# file chkpnt<N>.pth ở CUỐI MỖI giai đoạn KHÔNG PHẢI giai đoạn cuối (dùng để resume
# nội bộ, có thể xoá sau khi train xong nếu không cần).

set -euo pipefail

if [[ -z "${GS_REPO:-}" ]]; then
  echo "Lỗi: chưa set biến môi trường GS_REPO (đường dẫn tới repo graphdeco-inria/gaussian-splatting đã clone)." >&2
  exit 1
fi
if [[ ! -f "$GS_REPO/train.py" ]]; then
  echo "Lỗi: không thấy $GS_REPO/train.py — kiểm tra lại GS_REPO." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
ITERATIONS="${ITERATIONS:-30000}"
SH_DEGREE="${SH_DEGREE:-3}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
CLEANUP_DENSE_IMAGES="${CLEANUP_DENSE_IMAGES:-1}"
ANTENNA_FOCUS="${ANTENNA_FOCUS:-0}"
ANTIALIASING="${ANTIALIASING:-1}"
DEPTH_PRIOR="${DEPTH_PRIOR:-0}"
EXPOSURE_COMP="${EXPOSURE_COMP:-0}"
PROGRESSIVE_RESOLUTIONS="${PROGRESSIVE_RESOLUTIONS:-4 2 1}"
PROGRESSIVE_FRACTIONS="${PROGRESSIVE_FRACTIONS:-0.3 0.6 1.0}"

if [[ $# -ne 1 ]]; then
  echo "Cách dùng: $0 <1 scene duy nhất>  (khác 03_train_3dgs.sh — script này chỉ nhận 1 scene/lần)" >&2
  exit 1
fi
SCENE="$1"

if [[ "$ANTENNA_FOCUS" == "1" ]]; then
  echo "Lỗi: ANTENNA_FOCUS=1 CHƯA hỗ trợ ở script progressive-resolution này (chưa test tổ hợp" >&2
  echo "  antenna-focus + đổi resolution nhiều giai đoạn) — dùng 03_train_3dgs.sh nếu cần antenna-focus." >&2
  exit 1
fi
if [[ "$ANTIALIASING" == "1" ]] && ! grep -q "antialiasing" "$GS_REPO/arguments/__init__.py" 2>/dev/null; then
  echo "Lỗi: ANTIALIASING=1 nhưng \$GS_REPO có vẻ là bản clone CŨ (chưa có cờ --antialiasing)." >&2
  echo "  Checkout đúng commit đã pin (xem comment đầu 03_train_3dgs.sh) rồi thử lại." >&2
  exit 1
fi
if [[ "$DEPTH_PRIOR" == "1" ]] && ! grep -q "depth_l1_weight" "$GS_REPO/train.py" 2>/dev/null; then
  echo "Lỗi: DEPTH_PRIOR=1 nhưng \$GS_REPO chưa có depth regularization (bản clone cũ)." >&2
  exit 1
fi

read -ra RES_LIST <<< "$PROGRESSIVE_RESOLUTIONS"
read -ra FRAC_LIST <<< "$PROGRESSIVE_FRACTIONS"
if [[ ${#RES_LIST[@]} -eq 0 || ${#RES_LIST[@]} -ne ${#FRAC_LIST[@]} ]]; then
  echo "Lỗi: PROGRESSIVE_RESOLUTIONS (${#RES_LIST[@]} phần tử) và PROGRESSIVE_FRACTIONS (${#FRAC_LIST[@]} phần tử) phải cùng số lượng và >0." >&2
  exit 1
fi
LAST_FRAC="${FRAC_LIST[-1]}"
if [[ "$LAST_FRAC" != "1.0" && "$LAST_FRAC" != "1" ]]; then
  echo "Lỗi: phần tử CUỐI của PROGRESSIVE_FRACTIONS phải là 1.0 (giai đoạn cuối phải kết thúc đúng ITERATIONS), đang là $LAST_FRAC." >&2
  exit 1
fi

# Tính mốc iteration TUYỆT ĐỐI kết thúc mỗi giai đoạn từ % ITERATIONS, dùng
# python cho chắc (bash không làm tròn số thực gọn) — đồng thời validate tăng
# dần nghiêm ngặt (giai đoạn sau phải > giai đoạn trước, không thì train.py sẽ
# vòng lặp 0 iteration hoặc lỗi khó hiểu).
# QUAN TRỌNG: dùng $(...) (command substitution) chứ KHÔNG dùng < <(...) (process
# substitution) — lỗi thật đã bắt được khi tự test bằng train.py giả: process
# substitution chạy trong subshell riêng, `set -e` ở shell cha KHÔNG bắt được nếu
# python thất bại (assert tăng dần sai) — script từng ÂM THẦM chạy tiếp với
# PHASE_ENDS thiếu phần tử (vd chỉ 1/3 giai đoạn) mà không báo lỗi gì, in ra
# "xong toàn bộ" sai sự thật. command substitution + `|| exit 1` tường minh mới
# chắc chắn dừng lại đúng lúc.
PHASE_ENDS_RAW="$(python3 -c "
iters = $ITERATIONS
fracs = [float(x) for x in '''$PROGRESSIVE_FRACTIONS'''.split()]
prev = 0
for f in fracs:
    v = round(iters * f)
    assert v > prev, f'Mốc giai đoạn không tăng dần: {v} <= {prev} (kiểm tra lại PROGRESSIVE_FRACTIONS)'
    print(v)
    prev = v
")" || { echo "Lỗi: PROGRESSIVE_FRACTIONS không hợp lệ (từng giá trị x ITERATIONS phải tăng dần nghiêm ngặt) — xem lỗi python ở trên." >&2; exit 1; }
PHASE_ENDS=()
while IFS= read -r line; do PHASE_ENDS+=("$line"); done <<< "$PHASE_ENDS_RAW"
if [[ ${#PHASE_ENDS[@]} -ne ${#RES_LIST[@]} ]]; then
  echo "Lỗi nội bộ: số mốc giai đoạn tính được (${#PHASE_ENDS[@]}) khác số phần tử PROGRESSIVE_RESOLUTIONS (${#RES_LIST[@]})." >&2
  exit 1
fi

SOURCE_DIR="$PIPELINE_DIR/work/$SCENE/colmap/dense"
MODEL_DIR="$PIPELINE_DIR/work/$SCENE/gs_model"
LOG_DIR="$PIPELINE_DIR/work/$SCENE"

if [[ ! -d "$SOURCE_DIR/sparse/0" ]]; then
  echo "Lỗi: chưa thấy $SOURCE_DIR/sparse/0 — chạy 01_run_colmap.py --scene $SCENE trước." >&2
  exit 1
fi

AVAIL_KB=$(df -Pk "$PIPELINE_DIR" | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
echo "[$SCENE] Đĩa còn trống: ${AVAIL_GB}GB"
if [[ "$AVAIL_GB" -lt 5 ]]; then
  echo "[LỖI] Đĩa còn dưới 5GB trước khi train $SCENE — dừng lại." >&2
  exit 1
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

# Checkpoint .ply chuẩn (7000/15000/ITERATIONS) như 03_train_3dgs.sh, để render/eval/
# package dùng lại được ngay cả khi crash giữa chừng 1 giai đoạn sau này.
STANDARD_SAVE_ITERATIONS=()
for v in 7000 15000 "$ITERATIONS"; do
  if [[ "$v" -le "$ITERATIONS" ]]; then
    STANDARD_SAVE_ITERATIONS+=("$v")
  fi
done

PREV_CKPT=""
N_PHASES=${#PHASE_ENDS[@]}
for ((i = 0; i < N_PHASES; i++)); do
  PHASE_NUM=$((i + 1))
  RES="${RES_LIST[$i]}"
  PHASE_END="${PHASE_ENDS[$i]}"
  IS_LAST=$([[ $PHASE_NUM -eq $N_PHASES ]] && echo 1 || echo 0)

  # .ply lưu ở giai đoạn này: mọi mốc chuẩn <= PHASE_END (chưa lưu ở giai đoạn
  # trước) cộng chính PHASE_END (luôn lưu, để chắc chắn có snapshot resume/debug
  # ở ranh giới giai đoạn kể cả khi PHASE_END không trùng mốc chuẩn 7000/15000).
  SAVE_ITERS_THIS_PHASE=()
  for v in "${STANDARD_SAVE_ITERATIONS[@]}" "$PHASE_END"; do
    if [[ "$v" -le "$PHASE_END" ]]; then
      SAVE_ITERS_THIS_PHASE+=("$v")
    fi
  done
  SAVE_ITERS_THIS_PHASE=($(printf "%s\n" "${SAVE_ITERS_THIS_PHASE[@]}" | awk '!seen[$0]++'))

  CKPT_ARGS=()
  if [[ "$IS_LAST" == "0" ]]; then
    CKPT_ARGS+=(--checkpoint_iterations "$PHASE_END")
  fi
  RESUME_ARGS=()
  if [[ -n "$PREV_CKPT" ]]; then
    RESUME_ARGS+=(--start_checkpoint "$PREV_CKPT")
  fi

  LOG_FILE="$LOG_DIR/03_train_3dgs_progressive_phase${PHASE_NUM}.log"
  echo "===== [$SCENE] Giai đoạn $PHASE_NUM/$N_PHASES: resolution=$RES, tới iteration $PHASE_END" \
       "(sh_degree=$SH_DEGREE, densify_grad_threshold=$DENSIFY_GRAD_THRESHOLD," \
       "antialiasing=$ANTIALIASING, depth_prior=$DEPTH_PRIOR, exposure_comp=$EXPOSURE_COMP)" \
       "— log: $LOG_FILE ====="
  python "$GS_REPO/train.py" \
    -s "$SOURCE_DIR" \
    -m "$MODEL_DIR" \
    --iterations "$PHASE_END" \
    --save_iterations "${SAVE_ITERS_THIS_PHASE[@]}" \
    --test_iterations "$PHASE_END" \
    --sh_degree "$SH_DEGREE" \
    --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
    --resolution "$RES" \
    "${CKPT_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${MIP_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
  TRAIN_PID=$!

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "${PROGRESS_INTERVAL:-60}"
    LAST_PROGRESS=$(grep -oE "[0-9]+/${PHASE_END}" "$LOG_FILE" 2>/dev/null | tail -1 || true)
    if [[ -n "$LAST_PROGRESS" ]]; then
      echo "  [$SCENE] giai đoạn $PHASE_NUM tiến độ: $LAST_PROGRESS iterations"
    fi
  done

  set +e
  wait "$TRAIN_PID"
  STATUS=$?
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "[LỖI] Train thất bại ở giai đoạn $PHASE_NUM cho $SCENE (exit $STATUS) — 50 dòng cuối log:" >&2
    tail -n 50 "$LOG_FILE" >&2
    LAST_CKPT_PLY=$(ls -d "$MODEL_DIR"/point_cloud/iteration_* 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
    if [[ -n "$LAST_CKPT_PLY" ]]; then
      echo "[CỨU ĐƯỢC] Vẫn còn checkpoint .ply gần nhất tại: $LAST_CKPT_PLY (dùng tạm để render nếu cần)." >&2
    fi
    exit $STATUS
  fi
  echo "-> Xong giai đoạn $PHASE_NUM/$N_PHASES cho $SCENE."
  PREV_CKPT="$MODEL_DIR/chkpnt${PHASE_END}.pth"
  if [[ "$IS_LAST" == "0" && ! -f "$PREV_CKPT" ]]; then
    echo "[LỖI] Không thấy checkpoint resume $PREV_CKPT sau giai đoạn $PHASE_NUM — không thể tiếp tục giai đoạn sau." >&2
    exit 1
  fi
done

echo "-> Xong toàn bộ $N_PHASES giai đoạn cho $SCENE. Model cuối: $MODEL_DIR/point_cloud/iteration_$ITERATIONS/point_cloud.ply"

# Giống hệt 03_train_3dgs.sh — xem comment gốc ở đó về lý do cần file này (antialiasing
# không nằm trong cfg_args mà train.py tự ghi, phải lưu riêng để 04_render_test_poses.py
# tự phát hiện đúng khi render).
cat > "$MODEL_DIR/pipeline_train_flags.json" <<EOF
{"antialiasing": $( [[ "$ANTIALIASING" == "1" ]] && echo true || echo false ), "depth_prior": $( [[ "$DEPTH_PRIOR" == "1" ]] && echo true || echo false ), "exposure_comp": $( [[ "$EXPOSURE_COMP" == "1" ]] && echo true || echo false ), "antenna_focus": false, "progressive": true}
EOF

if [[ "$CLEANUP_DENSE_IMAGES" == "1" && -d "$SOURCE_DIR/images" ]]; then
  FREED_KB=$(du -sk "$SOURCE_DIR/images" 2>/dev/null | awk '{print $1}')
  rm -rf "$SOURCE_DIR/images"
  echo "  [dọn đĩa] Đã xoá $SOURCE_DIR/images (~$((FREED_KB / 1024))MB) — giữ lại sparse/0."
fi
if [[ "$CLEANUP_DENSE_IMAGES" == "1" && -d "$DEPTH_DIR" ]]; then
  rm -rf "$DEPTH_DIR"
  echo "  [dọn đĩa] Đã xoá $DEPTH_DIR."
fi

# Dọn checkpoint .pth trung gian (chỉ cần để resume nội bộ giữa các giai đoạn,
# không cần cho render/eval/package — .ply mới là thứ các script sau dùng).
for ((i = 0; i < N_PHASES - 1; i++)); do
  MID_CKPT="$MODEL_DIR/chkpnt${PHASE_ENDS[$i]}.pth"
  if [[ -f "$MID_CKPT" ]]; then
    rm -f "$MID_CKPT"
  fi
done
echo "  [dọn đĩa] Đã xoá các checkpoint .pth trung gian giữa các giai đoạn (giữ lại .ply)."
