#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRICK_DIR="$(dirname "$SCRIPT_DIR")"

source "$TRICK_DIR/hcm0031/default.env"

MASK_ROOT="$TRICK_DIR/hcm0031/m0_mask"
BOOTSTRAP_DIR="$MASK_ROOT/bootstrap_masks"
TOWER_BBOX3D_JSON="${TOWER_BBOX3D_JSON:-$WORK_ROOT/$SCENE/tower_bbox3d.json}"
MASK_DILATE_PX="${MASK_DILATE_PX:-12}"

python3 "$SCRIPT_DIR/bootstrap_tower_masks.py" \
  --scene "$SCENE" \
  --dataset_root "$DATASET_ROOT" \
  --tower_bbox3d_json "$TOWER_BBOX3D_JSON" \
  --out_dir "$BOOTSTRAP_DIR" \
  --dilate_px "$MASK_DILATE_PX"

echo "Bootstrap mask dir: $BOOTSTRAP_DIR"
