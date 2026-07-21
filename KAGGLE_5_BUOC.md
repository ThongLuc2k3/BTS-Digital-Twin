# 5 Bước Chạy Kaggle

Notebook:

- Train `30000`: [pipeline/kaggle_round1_train_30k.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_train_30k.ipynb)
- Resume từ Drive: [pipeline/kaggle_round1_resume_from_drive.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_resume_from_drive.ipynb)

Mục tiêu:

- notebook 1: train `30000`, render, chấm điểm, xuất `gs_model`
- notebook 2: nhận link Google Drive của `gs_model`, resume tiếp, so sánh model cũ và mới
- nếu muốn resume tiếp, chỉ cần lấy `gs_model` mới và chạy lại notebook 2

## Bước 1. Chọn đúng notebook

- bật `GPU`
- bật `Internet`
- nếu muốn train `30000`: dùng `kaggle_round1_train_30k.ipynb`
- nếu đã có `gs_model` và muốn train tiếp: dùng `kaggle_round1_resume_from_drive.ipynb`

## Bước 2. Điền token và dataset

Trong notebook:

- điền `GITHUB_TOKEN` nếu repo private
- nếu dataset đã mount sẵn trên Kaggle thì không cần điền thêm gì
- nếu dataset chưa có trên Kaggle thì mới dùng link dataset

Hai notebook đều tự dò `Dataset/VAI_NVS_DATA/phase1/public_set`.

## Bước 3. Chạy notebook 1 để lấy `gs_model`

Với [pipeline/kaggle_round1_train_30k.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_train_30k.ipynb):

- bạn chỉ cần chọn `SCENE`
- notebook sẽ:
  - train `30000`
  - render `test_poses.csv`
  - chấm điểm với GT ở `test/images`
  - nhả ra `gs_model`

## Bước 4. Giữ đúng cấu trúc `gs_model/`

Để resume ở notebook 2, giữ nguyên thư mục:

```text
gs_model/
  cfg_args
  pipeline_train_flags.json
  chkpnt30000.pth
  point_cloud/
    iteration_30000/
      point_cloud.ply
```

Rồi upload nguyên thư mục này lên Google Drive.

Không zip cũng được.
Không chỉ lấy riêng `point_cloud.ply`.

`iteration_7000/` và `iteration_15000/` có thể giữ nếu muốn, nhưng không bắt buộc cho workflow resume hiện tại.

## Bước 5. Chạy notebook 2 để resume và so sánh

Với [pipeline/kaggle_round1_resume_from_drive.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_resume_from_drive.ipynb):

- dán `CHECKPOINT_DRIVE_LINK`
- chọn `SCENE`
- đặt `TARGET_ITERATIONS`, ví dụ `60000`

Notebook sẽ:

- tải `gs_model` cũ
- render và chấm điểm model cũ
- tìm `chkpnt*.pth` lớn nhất
- train tiếp tới `TARGET_ITERATIONS`
- render và chấm điểm model mới
- in chênh lệch score

## Chốt nhanh

- Không nhắc tới workflow khác ngoài `round1`
- Notebook 1 để lấy `gs_model`
- Notebook 2 để resume vô hạn từ link Google Drive của `gs_model`
- Muốn tăng iteration sau này thì checkpoint phải có `.pth`
