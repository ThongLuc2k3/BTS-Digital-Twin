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
- [scripts/eval_round1_metrics.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/eval_round1_metrics.py)

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
