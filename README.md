# BTS Digital Twin — Novel View Synthesis (Viettel AI Race 2026, Vòng 1)

Sinh ảnh RGB tại các pose (góc nhìn) mục tiêu cho 8 scene trạm BTS, từ ảnh drone
train của mỗi scene — pipeline: COLMAP (ước lượng pose camera) → 3D Gaussian
Splatting (dựng scene 3D, train riêng từng scene) → render đúng pose yêu cầu.

**Nhánh này (`feature/mip-splatting`)**: hướng cải tiến sau khi có điểm chấm thật
(baseline vanilla 3DGS đạt 58.67/100, xem `Kết quả/Hướng đi.md`) — bật
`--antialiasing` (EWA Filter của Mip-Splatting, đã tích hợp sẵn trong repo Inria
từ 10/2024) + depth regularization (Depth Anything V2) + exposure compensation,
không cần đổi codebase 3DGS. Chi tiết: `Kết quả/Hướng đi.md`, `pipeline/README.md`.

## Cấu trúc repo

| File/thư mục | Nội dung |
|---|---|
| [`Đề bài.md`](./Đề%20bài.md) | Đề bài gốc + tóm tắt quy định vòng 1 từ BTC |
| [`KE_HOACH_VONG1.md`](./KE_HOACH_VONG1.md) | Kế hoạch triển khai chi tiết: baseline, rủi ro kỹ thuật, mốc thời gian, checklist nộp bài |
| [`Kết quả/Hướng đi.md`](./Kết%20quả/Hướng%20đi.md) | Phân tích điểm chấm thật + thứ tự ưu tiên hướng cải tiến (Mip-Splatting, depth prior, antenna-focus...) |
| [`Dataset/README.md`](./Dataset/README.md) | Phân tích cấu trúc dataset thật (public_set/private_set1, format `test_poses.csv`, các bất thường phát hiện được) — **dataset thật (`Dataset/VAI_NVS_DATA/`) không nằm trong repo** (quá nặng, ~1GB), tải riêng qua Google Drive |
| [`pipeline/`](./pipeline) | Toàn bộ code: COLMAP, train/render 3D Gaussian Splatting, depth prior, tính PSNR/SSIM/LPIPS, đóng gói submission |
| [`pipeline/README.md`](./pipeline/README.md) | Hướng dẫn cài đặt + thứ tự chạy từng script |
| [`pipeline/kaggle_public.ipynb`](./pipeline/kaggle_public.ipynb) / [`bts-digital-twin-public.ipynb`](./bts-digital-twin-public.ipynb) | Notebook train+eval 1 scene `public_set`/phiên Kaggle (2 file nội dung giống nhau — root là bản để upload/chạy trực tiếp trên Kaggle, đồng bộ thủ công từ `pipeline/`) |
| [`pipeline/kaggle_private.ipynb`](./pipeline/kaggle_private.ipynb) | Notebook train+render 1 scene `private_set1`/phiên Kaggle (không eval — không có ảnh GT) |
| [`pipeline/kaggle_submission.ipynb`](./pipeline/kaggle_submission.ipynb) | Notebook tải checkpoint đã train + render lại + đóng gói `submission_round1.zip` |

## Chạy nhanh

Cách dễ nhất: mở `pipeline/kaggle_public.ipynb` (hoặc bản đồng bộ ở root
`bts-digital-twin-public.ipynb`) trên Kaggle (bật GPU + Internet), điền `REPO_URL`
(link repo này, nhớ `REPO_BRANCH = "feature/mip-splatting"`) và `GDRIVE_URL` (link
dataset), chạy tuần tự từng cell theo hướng dẫn trong chính notebook. Chi tiết đầy
đủ ở `pipeline/README.md`. Vòng 1 dùng 3 notebook riêng (`kaggle_public` /
`kaggle_private` / `kaggle_submission`) vì tổng thời gian train 13 scene không vừa
1 phiên Kaggle — xem `KE_HOACH_VONG1.md` mục 8.

**Sanity-check hệ toạ độ (`02_validate_frame.py`)** không còn bắt buộc — đã xác
nhận sparse COLMAP của BTC khớp `test_poses.csv` (xem `KE_HOACH_VONG1.md` mục 2 và 4).

## Trạng thái

Đang ở giai đoạn Vòng 1 (Sơ loại) — deadline nộp `submission_round1.zip`: 30/07/2026.
Đã nộp baseline vanilla 3DGS (Score 58.67320/100, xem `Kết quả/Kết quả chấm.png`) —
đang triển khai cải tiến trên nhánh `feature/mip-splatting`, chưa chạy GPU thật.
