# BTS Digital Twin

Repo này đã được rút gọn lại cho 1 hướng duy nhất:

- train từ đầu
- lưu checkpoint đúng chuẩn
- resume lên iteration cao hơn ở lần chạy sau
- kiểm chứng ý tưởng resume trên round1 public_set trước khi dùng cho round2

Không còn giữ các nhánh thử nghiệm cũ.

## Cấu trúc chính

- [KAGGLE_5_BUOC.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/KAGGLE_5_BUOC.md): hướng dẫn 1 trang
- [pipeline/README.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/README.md): mô tả workflow mới
- [pipeline/kaggle_train_resume.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_train_resume.ipynb): notebook Kaggle cho round2
- [pipeline/kaggle_round1_public_resume.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_public_resume.ipynb): notebook Kaggle để test score trên round1 public_set
- [pipeline/scripts/03_train_3dgs.sh](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/03_train_3dgs.sh): script train/resume

## Quyết định kỹ thuật

Tôi chọn hướng `train lại sạch + lưu checkpoint để resume` thay vì cố thêm cải tiến mới vào baseline hiện tại.

Lý do:

- xác suất thành công cao hơn
- ít rủi ro hơn so với thêm kiến trúc/loss mới
- mở được đường `30k -> 60k -> 90k`
- quản lý checkpoint rõ ràng hơn

## Cách dùng đề xuất

1. Chạy [pipeline/kaggle_round1_public_resume.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_public_resume.ipynb) trên 1 scene `round1 public_set`
2. So sánh score ở `30000` và `60000`
3. Nếu score tăng thật, dùng cùng workflow đó cho [pipeline/kaggle_train_resume.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_train_resume.ipynb) ở round2

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
