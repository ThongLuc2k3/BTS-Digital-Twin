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
COLMAP_BIN="${COLMAP_BIN:-$(command -v colmap || true)}"
PREPARE_SCENE="${PREPARE_SCENE:-1}"
RUN_STEREO="${RUN_STEREO:-1}"
PATCH_MATCH_MAX_IMAGE_SIZE="${PATCH_MATCH_MAX_IMAGE_SIZE:-2000}"
PATCH_MATCH_GEOM_CONSISTENCY="${PATCH_MATCH_GEOM_CONSISTENCY:-true}"
PATCH_MATCH_WINDOW_RADIUS="${PATCH_MATCH_WINDOW_RADIUS:-5}"
PATCH_MATCH_NUM_SAMPLES="${PATCH_MATCH_NUM_SAMPLES:-15}"
PATCH_MATCH_NUM_ITERATIONS="${PATCH_MATCH_NUM_ITERATIONS:-5}"
PATCH_MATCH_GPU_INDEX="${PATCH_MATCH_GPU_INDEX:--1}"
PATCH_MATCH_CACHE_SIZE="${PATCH_MATCH_CACHE_SIZE:-32}"
FUSION_MIN_NUM_PIXELS="${FUSION_MIN_NUM_PIXELS:-5}"
FUSION_MAX_REPROJ_ERROR="${FUSION_MAX_REPROJ_ERROR:-2.0}"
FUSION_MAX_DEPTH_ERROR="${FUSION_MAX_DEPTH_ERROR:-0.01}"
FUSION_MAX_NORMAL_ERROR="${FUSION_MAX_NORMAL_ERROR:-10.0}"

if [[ -z "$COLMAP_BIN" ]]; then
  echo "Lỗi: không tìm thấy 'colmap'. Hãy set COLMAP_BIN=/path/to/colmap" >&2
  exit 1
fi

if [[ "$PREPARE_SCENE" == "1" ]]; then
  python3 "$SCRIPT_DIR/prepare_round1_scene.py" \
    --scene "$SCENE" \
    --dataset_root "$DATASET_ROOT" \
    --work_root "$WORK_ROOT"
fi

DENSE_DIR="$WORK_ROOT/$SCENE/colmap/dense"
LOG_DIR="$WORK_ROOT/$SCENE/logs"
PATCH_LOG="$LOG_DIR/04_patch_match_stereo.log"
FUSION_LOG="$LOG_DIR/04_stereo_fusion.log"
SUMMARY_FILE="$LOG_DIR/04_colmap_dense_summary.txt"
FUSED_PLY="$DENSE_DIR/fused.ply"

if [[ ! -d "$DENSE_DIR/images" || ! -d "$DENSE_DIR/sparse/0" ]]; then
  echo "Lỗi: chưa có workspace dense hợp lệ ở $DENSE_DIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

if [[ "$RUN_STEREO" != "1" ]]; then
  {
    echo "scene=$SCENE"
    echo "dense_dir=$DENSE_DIR"
    echo "colmap_bin=$COLMAP_BIN"
    echo "patch_match_seconds=skipped"
    echo "stereo_fusion_seconds=skipped"
    echo "depth_map_files=0"
    echo "normal_map_files=0"
    echo "consistency_graph_files=0"
    echo "fused_ply="
    echo "fused_ply_bytes=0"
    echo "patch_log="
    echo "fusion_log="
  } >"$SUMMARY_FILE"
  echo "COLMAP dense pilot xong (RUN_STEREO=0, bỏ qua patch_match_stereo/stereo_fusion): scene=$SCENE"
  echo "images/sparse đã sẵn sàng ở $DENSE_DIR (đủ để train, không có depth_maps vì không dùng --depths)"
  echo "Summary: $SUMMARY_FILE"
  exit 0
fi

run_timed() {
  local log_file="$1"
  shift
  local start_ts end_ts status elapsed
  start_ts="$(date +%s)"
  set +e
  "$@" >"$log_file" 2>&1
  status=$?
  set -e
  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  echo "$status" "$elapsed"
}

read -r patch_status patch_secs < <(
  run_timed "$PATCH_LOG" \
    "$COLMAP_BIN" patch_match_stereo \
      --workspace_path "$DENSE_DIR" \
      --workspace_format COLMAP \
      --PatchMatchStereo.max_image_size "$PATCH_MATCH_MAX_IMAGE_SIZE" \
      --PatchMatchStereo.geom_consistency "$PATCH_MATCH_GEOM_CONSISTENCY" \
      --PatchMatchStereo.window_radius "$PATCH_MATCH_WINDOW_RADIUS" \
      --PatchMatchStereo.num_samples "$PATCH_MATCH_NUM_SAMPLES" \
      --PatchMatchStereo.num_iterations "$PATCH_MATCH_NUM_ITERATIONS" \
      --PatchMatchStereo.gpu_index "$PATCH_MATCH_GPU_INDEX" \
      --PatchMatchStereo.cache_size "$PATCH_MATCH_CACHE_SIZE"
)

if [[ "$patch_status" -ne 0 ]]; then
  echo "[LỖI] patch_match_stereo thất bại. Xem log: $PATCH_LOG" >&2
  tail -n 50 "$PATCH_LOG" >&2 || true
  exit "$patch_status"
fi

read -r fusion_status fusion_secs < <(
  run_timed "$FUSION_LOG" \
    "$COLMAP_BIN" stereo_fusion \
      --workspace_path "$DENSE_DIR" \
      --workspace_format COLMAP \
      --input_type geometric \
      --output_path "$FUSED_PLY" \
      --StereoFusion.min_num_pixels "$FUSION_MIN_NUM_PIXELS" \
      --StereoFusion.max_reproj_error "$FUSION_MAX_REPROJ_ERROR" \
      --StereoFusion.max_depth_error "$FUSION_MAX_DEPTH_ERROR" \
      --StereoFusion.max_normal_error "$FUSION_MAX_NORMAL_ERROR"
)

if [[ "$fusion_status" -ne 0 ]]; then
  echo "[LỖI] stereo_fusion thất bại. Xem log: $FUSION_LOG" >&2
  tail -n 50 "$FUSION_LOG" >&2 || true
  exit "$fusion_status"
fi

depth_count=0
normal_count=0
consistency_count=0
if [[ -d "$DENSE_DIR/stereo/depth_maps" ]]; then
  depth_count=$(find "$DENSE_DIR/stereo/depth_maps" -maxdepth 1 -type f | wc -l | tr -d ' ')
fi
if [[ -d "$DENSE_DIR/stereo/normal_maps" ]]; then
  normal_count=$(find "$DENSE_DIR/stereo/normal_maps" -maxdepth 1 -type f | wc -l | tr -d ' ')
fi
if [[ -d "$DENSE_DIR/stereo/consistency_graphs" ]]; then
  consistency_count=$(find "$DENSE_DIR/stereo/consistency_graphs" -maxdepth 1 -type f | wc -l | tr -d ' ')
fi

{
  echo "scene=$SCENE"
  echo "dense_dir=$DENSE_DIR"
  echo "colmap_bin=$COLMAP_BIN"
  echo "patch_match_seconds=$patch_secs"
  echo "stereo_fusion_seconds=$fusion_secs"
  echo "depth_map_files=$depth_count"
  echo "normal_map_files=$normal_count"
  echo "consistency_graph_files=$consistency_count"
  if [[ -f "$FUSED_PLY" ]]; then
    echo "fused_ply=$FUSED_PLY"
    echo "fused_ply_bytes=$(stat -c %s "$FUSED_PLY")"
  else
    echo "fused_ply="
    echo "fused_ply_bytes=0"
  fi
  echo "patch_log=$PATCH_LOG"
  echo "fusion_log=$FUSION_LOG"
} >"$SUMMARY_FILE"

echo "COLMAP dense pilot xong: scene=$SCENE"
echo "PatchMatch: ${patch_secs}s | StereoFusion: ${fusion_secs}s"
echo "Depth maps: $depth_count | Normal maps: $normal_count | Consistency graphs: $consistency_count"
if [[ -f "$FUSED_PLY" ]]; then
  echo "Fused PLY: $FUSED_PLY"
fi
echo "Summary: $SUMMARY_FILE"
