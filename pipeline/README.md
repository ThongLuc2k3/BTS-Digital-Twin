# Pipeline Mới

Pipeline này chỉ giữ 1 workflow cho `round1 public_set`:

1. train `30000`
2. lấy `gs_model`
3. resume từ link Google Drive của `gs_model`
4. so sánh model cũ và model mới

## File chính

- [kaggle_round1_train_30k.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_train_30k.ipynb)
- [kaggle_round1_resume_from_drive.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_resume_from_drive.ipynb)
- [scripts/03_train_3dgs.sh](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/03_train_3dgs.sh)
- [scripts/render_round1_test_poses.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/render_round1_test_poses.py)
- [scripts/eval_round1_metrics.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/eval_round1_metrics.py) — hỗ trợ thêm `--tower_bbox3d_json` để tính metric riêng cho vùng crop trụ (xem `P0_RESTORE_CHECKLIST.md` mục M0 - tower crop)
- [scripts/estimate_object_bbox3d.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/estimate_object_bbox3d.py) — ước lượng bounding box 3D của trụ từ sparse COLMAP + vài ảnh mốc

## Dataset giả định

Round1 public_set dùng:

```text
Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/train/images
Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/train/sparse/0
Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/test/images
Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/test/test_poses.csv
```

## Biến môi trường quan trọng

- `GS_REPO`: đường dẫn repo `gaussian-splatting`
- `DATASET_ROOT`: gốc dataset, trỏ tới `Dataset/VAI_NVS_DATA/phase1/public_set`
- `ITERATIONS`: iteration đích
- `START_CHECKPOINT`: file `.pth` để resume
- `SAVE_FINAL_CHECKPOINT=1`: lưu `chkpnt<ITERATIONS>.pth`
- `ANTIALIASING=1`
- `EXPOSURE_COMP=1`

## Gợi ý mặc định

- notebook 1: train `30000`
- notebook 2: resume từ `chkpnt*.pth` lớn nhất có trong `gs_model`

## Pilot `B2` cho `hcm0031`

Repo hiện có sẵn workflow pilot cho nhánh:

`P0 -> M0 -> B2`

Ở mức script:

- [scripts/04_run_colmap_dense.sh](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/04_run_colmap_dense.sh): chạy `prepare_round1_scene.py`, `patch_match_stereo`, `stereo_fusion`, ghi log và timing.
- [scripts/05_run_b2_pilot.sh](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/05_run_b2_pilot.sh): orchestration cho dense pilot, train tùy chọn, render, và chấm lại `M0`.
- [scripts/manage_b2_artifacts.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/manage_b2_artifacts.py): audit `B2`, đóng gói curated bundle, hoặc re-eval `latest iteration`.
- [scripts/generate_b2_variant_notebooks.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/generate_b2_variant_notebooks.py): sinh các notebook biến thể `B2_done`.

### Chạy dense pilot tối thiểu

```bash
export COLMAP_BIN=/path/to/colmap
python3 -c "import pycolmap"
bash pipeline/scripts/04_run_colmap_dense.sh hcm0031
```

Kết quả chính:

- `pipeline/work/hcm0031/logs/04_patch_match_stereo.log`
- `pipeline/work/hcm0031/logs/04_stereo_fusion.log`
- `pipeline/work/hcm0031/logs/04_colmap_dense_summary.txt`

### Chạy pilot `B2` end-to-end

Nếu đã có model trong `pipeline/work/hcm0031/gs_model` và chỉ muốn re-render + re-eval:

```bash
export COLMAP_BIN=/path/to/colmap
export GS_REPO=/path/to/gaussian-splatting
bash pipeline/scripts/05_run_b2_pilot.sh hcm0031
```

Nếu muốn train/resume lại từ source `prepared` sau khi dense stereo xong:

```bash
export COLMAP_BIN=/path/to/colmap
export GS_REPO=/path/to/gaussian-splatting
export RUN_TRAIN=1
export SOURCE_MODE=prepared
bash pipeline/scripts/05_run_b2_pilot.sh hcm0031
```

### Ghi chú quan trọng

- `03_train_3dgs.sh` giờ hỗ trợ `SOURCE_MODE=auto|raw|prepared`.
- `SOURCE_MODE=prepared` dùng `pipeline/work/<SCENE>/colmap/dense` làm source train.
- `B2` trong repo hiện tại mới đóng gói chắc phần `dense stereo` pilot và vòng `render/eval`.
- `depth regularization` thật vẫn phụ thuộc support của nhánh `GS_REPO` đang train.

### Notebook biến thể `B2_done`

Các notebook trong [downloads](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/downloads) được tách theo mục tiêu debug cụ thể:

- [B2_done.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/downloads/B2_done.ipynb): notebook gốc đang dùng để chạy và audit `B2`.
- [B2_done 1.ipynb](</home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin/downloads/B2_done 1.ipynb>): bật `depth supervision` thật bằng cách patch `03_train_3dgs.sh` trong repo clone để truyền `--depths .../stereo/depth_maps` vào `GS_REPO`.
- [B2_done 2.ipynb](</home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin/downloads/B2_done 2.ipynb>): giống bản 1, nhưng ép `LOW_VRAM_PROFILE=0` để bỏ profile low-VRAM và trả về train profile mặc định nếu GPU chịu được.
- [B2_done 2 safe.ipynb](</home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin/downloads/B2_done 2 safe.ipynb>): fallback an toàn hơn, vẫn giữ `LOW_VRAM_PROFILE=1` nhưng override các biến densification để tránh bị khóa Gaussian như run cũ.

Ba notebook biến thể đều:

- giữ nguyên notebook gốc
- patch repo clone tạm thời trong runtime notebook, không sửa file gốc trong workspace
- fail sớm nếu `GS_REPO` clone không lộ rõ support cho `--depths`
