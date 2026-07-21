# BTS Digital Twin

Repo này đã được rút gọn lại cho 1 hướng duy nhất:

- train `30000`
- lấy `gs_model`
- resume nhiều lần từ `gs_model` qua Google Drive
- mỗi lần resume đều so sánh model cũ và model mới trên `round1 public_set`

Không còn giữ các nhánh thử nghiệm cũ.

## Cấu trúc chính

- [KAGGLE_5_BUOC.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/KAGGLE_5_BUOC.md): hướng dẫn 1 trang
- [pipeline/README.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/README.md): mô tả workflow mới
- [pipeline/kaggle_round1_train_30k.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_train_30k.ipynb): notebook train `30000` và xuất `gs_model`
- [pipeline/kaggle_round1_resume_from_drive.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_resume_from_drive.ipynb): notebook resume từ link Drive của `gs_model`
- [pipeline/scripts/03_train_3dgs.sh](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/03_train_3dgs.sh): script train/resume

## Cách dùng đề xuất

1. Chạy [pipeline/kaggle_round1_train_30k.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_train_30k.ipynb) để lấy `gs_model`
2. Upload `gs_model` lên Google Drive
3. Chạy [pipeline/kaggle_round1_resume_from_drive.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_resume_from_drive.ipynb) với link Drive đó
4. Nếu muốn resume tiếp, lấy `gs_model` mới rồi chạy lại notebook resume

## Chuẩn checkpoint bắt buộc

Muốn resume sau này, phải giữ nguyên thư mục `gs_model/` với tối thiểu:

```text
gs_model/
  cfg_args
  pipeline_train_flags.json
  chkpnt30000.pth
  point_cloud/
    iteration_30000/
      point_cloud.ply
```

Không chỉ giữ riêng `point_cloud.ply`.
