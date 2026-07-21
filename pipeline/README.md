# Pipeline Mới

Pipeline này chỉ giữ 1 workflow:

1. train từ đầu
2. lưu checkpoint `.pth`
3. resume lên iteration cao hơn

## File chính

- [kaggle_train_resume.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_train_resume.ipynb)
- [kaggle_round1_public_resume.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_public_resume.ipynb)
- [scripts/03_train_3dgs.sh](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/03_train_3dgs.sh)
- [scripts/render_round1_test_poses.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/render_round1_test_poses.py)
- [scripts/eval_round1_metrics.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/eval_round1_metrics.py)

## Dataset giả định

Script train mới dùng trực tiếp dataset đang có cấu trúc:

```text
Dataset/VAI_NVS_DATA_ROUND2/<SCENE>/train/images
Dataset/VAI_NVS_DATA_ROUND2/<SCENE>/train/sparse/0
```

Không còn phụ thuộc các script cũ như holdout, depth, antenna, refine.

Round1 public_set dùng:

```text
Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/train/images
Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/train/sparse/0
Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/test/images
Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/test/test_poses.csv
```

## Biến môi trường quan trọng

- `GS_REPO`: đường dẫn repo `gaussian-splatting`
- `DATASET_ROOT`: gốc dataset, mặc định `Dataset/VAI_NVS_DATA_ROUND2`
- `ITERATIONS`: iteration đích
- `START_CHECKPOINT`: file `.pth` để resume
- `SAVE_FINAL_CHECKPOINT=1`: lưu `chkpnt<ITERATIONS>.pth`
- `ANTIALIASING=1`
- `EXPOSURE_COMP=1`

## Gợi ý mặc định

- chạy đầu: `ITERATIONS=30000`, `SAVE_FINAL_CHECKPOINT=1`
- chạy sau: `START_CHECKPOINT=.../chkpnt30000.pth`, `ITERATIONS=60000`
