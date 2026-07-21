# 5 Bước Chạy Kaggle

Notebook:

- Round1 test score thật: [pipeline/kaggle_round1_public_resume.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_public_resume.ipynb)
- Round2 train/resume: [pipeline/kaggle_train_resume.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_train_resume.ipynb)

Mục tiêu:

- với `round1`: bấm `Run All` để tự chạy `30000 -> chấm điểm -> 60000 -> chấm lại`
- với `round2`: train từ đầu hoặc resume thủ công bằng checkpoint Drive
- luôn giữ nguyên thư mục `gs_model/` nếu muốn train tiếp lần sau

## Bước 1. Chọn đúng notebook

- bật `GPU`
- bật `Internet`
- nếu muốn kiểm chứng score trước: dùng `kaggle_round1_public_resume.ipynb`
- nếu muốn chạy round2: dùng `kaggle_train_resume.ipynb`

## Bước 2. Điền token và dataset

Trong notebook:

- điền `GITHUB_TOKEN` nếu repo private
- nếu dataset đã mount sẵn trên Kaggle thì không cần điền thêm gì
- nếu dataset chưa có trên Kaggle thì mới dùng link dataset

Riêng với `round1`, notebook one-click sẽ tự dò `Dataset/VAI_NVS_DATA/phase1/public_set`.

## Bước 3. Chạy round1 one-click nếu muốn kiểm chứng score

Với [pipeline/kaggle_round1_public_resume.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_public_resume.ipynb):

- bạn chỉ cần chọn `SCENE`
- notebook tự chạy:
  - train `30000`
  - render `test_poses.csv`
  - chấm điểm với GT ở `test/images`
  - resume lên `60000`
  - render lại
  - chấm lại và in chênh lệch score

## Bước 4. Nếu muốn resume ở round2 hoặc lần sau, giữ đúng cấu trúc `gs_model/`

Nếu muốn resume ở lần sau, phải lấy nguyên thư mục:

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

## Bước 5. Chạy round2 khi đã xác nhận round1 có cải thiện

Với [pipeline/kaggle_train_resume.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_train_resume.ipynb):

- lần đầu:
  - `MODE = "train"`
  - `ITERATIONS = 30000`
- lần sau:
  - `MODE = "resume"`
  - dán `CHECKPOINT_DRIVE_LINK`
  - tăng `ITERATIONS` lên `60000`

Notebook sẽ tải `gs_model/`, tìm `chkpnt30000.pth`, rồi train tiếp.

## Chốt nhanh

- `round1 public_set` bây giờ là notebook one-click, không cần tự đổi `MODE=train/resume`
- nếu `round1` cho thấy score tăng sau resume, mới đem workflow đó sang `round2`
- Không dùng các nhánh thử nghiệm cũ
- Muốn tăng iteration sau này thì checkpoint phải có `.pth`
