# BTS Digital Twin — Novel View Synthesis (Viettel AI Race 2026, Vòng 1)

Sinh ảnh RGB tại các pose (góc nhìn) mục tiêu cho 8 scene trạm BTS, từ ảnh drone
train của mỗi scene — pipeline: COLMAP (ước lượng pose camera) → 3D Gaussian
Splatting (dựng scene 3D, train riêng từng scene) → render đúng pose yêu cầu.

## Cấu trúc repo

| File/thư mục | Nội dung |
|---|---|
| [`Đề bài.md`](./Đề%20bài.md) | Đề bài gốc + tóm tắt quy định vòng 1 từ BTC |
| [`Hướng đi.md`](./Hướng%20đi.md) | Kế hoạch triển khai + phân tích điểm chấm thật + thứ tự ưu tiên hướng cải tiến (baseline, checklist nộp bài, Mip-Splatting, depth prior, antenna-focus...) |
| [`Dataset/README.md`](./Dataset/README.md) | Phân tích cấu trúc dataset thật (public_set/private_set1, format `test_poses.csv`, các bất thường phát hiện được) — **dataset thật (`Dataset/VAI_NVS_DATA/`) không nằm trong repo** (quá nặng, ~1GB), tải riêng qua Google Drive |
| [`pipeline/`](./pipeline) | Toàn bộ code: COLMAP, train/render 3D Gaussian Splatting, tính PSNR/SSIM/LPIPS, đóng gói submission |
| [`pipeline/README.md`](./pipeline/README.md) | Hướng dẫn cài đặt + thứ tự chạy từng script |
| [`pipeline/kaggle_pipeline.ipynb`](./pipeline/kaggle_pipeline.ipynb) | Notebook chạy toàn bộ pipeline trên Kaggle (GPU free) — clone repo này + tải dataset từ Google Drive |

## Chạy nhanh

Cách dễ nhất: mở `pipeline/kaggle_pipeline.ipynb` trên Kaggle (bật GPU + Internet),
điền `REPO_URL` (link repo này) và `GDRIVE_URL` (link dataset), chạy tuần tự từng
cell theo hướng dẫn trong chính notebook. Chi tiết đầy đủ ở `pipeline/README.md`.

**Sanity-check hệ toạ độ (`02_validate_frame.py`)** không còn bắt buộc — đã xác nhận
sparse COLMAP của BTC khớp `test_poses.csv` (xem `Hướng đi.md` mục 1).

## Trạng thái

Đang ở giai đoạn Vòng 1 (Sơ loại) — deadline nộp `submission_round1.zip`: 30/07/2026.
