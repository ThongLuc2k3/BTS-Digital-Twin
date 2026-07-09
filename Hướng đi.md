# Kế hoạch & Hướng đi kỹ thuật — BTS Digital Twin (Viettel AI Race 2026, Vòng 1)

> Gộp `KE_HOACH_VONG1.md` (kế hoạch ban đầu) + `Hướng đi.md` cũ (phân tích sau khi có
> điểm chấm thật) vào 1 file, đã bỏ phần trùng lặp/lỗi thời (mốc thời gian tuần đã qua,
> câu hỏi đã chốt xong không cần nhắc lại chi tiết).

## 0. Tóm tắt đề bài & tiến độ hiện tại

- Nhiệm vụ: từ 150–300 ảnh drone/scene (`train/images/`), render ảnh RGB tại các pose
  BTC chỉ định (`test_poses.csv`) cho **8 scene** `private_set1`. Quy trình chuẩn NVS:
  COLMAP (pose + point cloud thưa) → train 3D Gaussian Splatting → render đúng pose.
  Không nộp model 3D, chỉ nộp PNG đúng format `submission_round1.zip/<scene>/<image>.png`.
- Deadline **30/07/2026**. Giới hạn nộp **5 lần/ngày, cách nhau ≥600s**, hệ thống chỉ giữ
  **bản nộp cuối cùng** — không nộp bản thử/lỗi ngay trước deadline, và nên nộp đè sớm
  ngay khi có bản tốt hơn thay vì chờ tới cuối.
- **Đã nộp thật, đã chấm** trên `private_set1` (8/8 scene khớp): **Score 58.67320**
  (PSNR 19.47 dB, SSIM 0.5637, LPIPS 0.2480) — Top 1 hiện ~74.9, cần lội ngược ~16 điểm.
  Đây là **vanilla 3DGS gốc, không sửa gì**, dùng thẳng sparse COLMAP của BTC (hệ toạ độ
  đã xác nhận đúng, không phải lỗi pose).

## 1. Dữ liệu & pipeline đã xác nhận (không cần làm lại)

- ✅ `sparse/0/` hợp lệ ở **13/13 scene** → dùng thẳng, không cần tự chạy lại COLMAP.
- ✅ Hệ toạ độ COLMAP khớp `test_poses.csv` — xác nhận bằng chạy thật `hcm0031` (train đủ
  30000 iter, PSNR mean 21.689, số liệu hợp lý).
- Ảnh đã downscale sẵn 1320×989, không EXIF/GPS. `public_set` (5 scene) có ảnh test thật
  (dùng tự chấm điểm); `private_set1` (8 scene) không có ảnh test, chỉ có `test_poses.csv`
  — đây là tập phải nộp.
- Máy local (GTX 1650 4GB) không đủ train 3DGS chất lượng — dùng GPU thuê/free
  (Kaggle/Colab). Từng gặp CUDA OOM ở scene nhiều chi tiết mảnh (khung thép/dây cáp khiến
  densify sinh rất nhiều Gaussian) — `03_train_3dgs.sh` đã có cờ giảm tải (`SH_DEGREE`,
  `DENSIFY_GRAD_THRESHOLD`, `RESOLUTION`) + tự lưu checkpoint giữa chừng.
- Quy ước đã chốt với BTC (đọc kỹ `Đề bài.md`, không suy đoán): tên file PNG giữ nguyên
  `.JPG` như `image_name` gốc (không đổi đuôi); tên thư mục scene trong zip dùng **tên
  thật** (`HCM0249`...), `scene_001` trong đề chỉ là ví dụ minh hoạ định dạng; được dùng
  AI hỗ trợ code tự do.
- **Công thức Score chính thức** (Đề bài.md mục 8.4):
  `Score = 0.4×(1−LPIPS) + 0.3×SSIM + 0.3×PSNR_norm`, `PSNR_norm = clamp(PSNR/PSNR_max, 0, 1)`,
  và **"điểm trên bảng xếp hạng là điểm trung bình của toàn bộ các scene"** — trung bình
  cộng Score TỪNG SCENE, không gộp ảnh của nhiều scene lại tính chung.

## 2. Phát hiện quan trọng sau điểm chấm thật

- Giải ngược công thức Score từ điểm chấm thật (58.67320) → **`PSNR_max` thực tế ≈ 50**
  (không phải 30 như giả định ban đầu), sai số <0.001%. Đã cập nhật mặc định này vào
  `pipeline/scripts/05_eval_metrics.py` (cùng lúc sửa lại cách gộp điểm nhiều scene cho
  đúng luật ở mục 1).
- Hệ quả: PSNR chia cho 50 nên đóng góp vào điểm rất "nhẹ tay" — **LPIPS (trọng số 0.4)
  và SSIM (0.3) mới là đòn bẩy điểm số thật**, tức cải thiện *chất lượng cảm quan + cấu
  trúc hình học* quan trọng hơn nhiều so với cày PSNR thuần bằng tăng iteration. Đây là
  lý do quyết định thứ tự ưu tiên ở mục 3.
- Nguyên nhân 58.67 bị kẹt: **(a)** ăng-ten/dây cáp là cấu trúc mảnh, Gaussian ellipsoid
  phải phình to để phủ → mờ/răng cưa; **(b)** pose test ở góc không có ảnh train → dễ sinh
  floaters vùng khuyết; **(c)** nền trời/mây/cây chiếm phần lớn khung hình, loss hiện tại
  chia đều cho cả nền lẫn ăng-ten nên Gaussian bị lãng phí vào vẽ nền.

## 3. Thứ tự ưu tiên hướng cải thiện

| # | Hướng đi | Đánh vào nguyên nhân | Khả năng lội ngược dòng | Effort/rủi ro |
|---|---|---|---|---|
| 1 | **Antenna-region-focus** (đã có sẵn, chỉ cần chạy thật) | (c) nền nhiễu | Trung bình | Rất thấp — đã xây ~90% |
| 2 | **Mip-Splatting** (đổi lõi chống alias) | (a) cấu trúc mảnh | **Cao** | Thấp |
| 3 | **Depth Anything V2** (`L_depth`) | (b) floaters vùng khuyết | **Cao** | Trung bình |
| 4 | **Edge loss (Sobel/Canny)** | (a) biên cạnh mờ | Thấp-Trung bình | Rất thấp |
| 5 | **Appearance embedding** | sai lệch màu/sáng giữa ảnh | Thấp | Thấp |
| 6 | *(dự phòng, chỉ nếu dư thời gian)* 2DGS / Scaffold-GS / NeRF distillation / diffusion view completion | (a)+(b) triệt để hơn | Cao nhưng rủi ro cao | Cao |

Làm theo đúng thứ tự 1→5, test xong trên `public_set` (có ảnh thật so PSNR/SSIM/LPIPS)
rồi mới áp dụng `private_set1` + nộp đè. #6 chỉ quay lại nếu #1-5 xong sớm và còn dư
ngày/GPU — NeRF distillation riêng còn tốn gấp đôi compute (đã ~37.5 GPU-giờ chỉ 1 lượt
vanilla 3DGS/13 scene, xem mục 5).

### #1 — Antenna-region-focus

`07_build_antenna_weights.py` + `apply_antenna_patch.py`: từ 1 khung pixel quanh ăng-ten,
tự chiếu mask/độ phủ lên toàn bộ ảnh train qua sparse COLMAP, vá `train.py` tăng trọng số
L1 vùng ăng-ten + ưu tiên lấy mẫu camera thấy rõ ăng-ten. ⚠️ Nhánh gốc
`feature/antenna-region-focus` đã đổi tên thành `feature/mip-splatting` (đã merge
`main` mới nhất) để dồn lực làm #2/#3 trước — `apply_antenna_patch.py` viết cho bản
`train.py` cũ hơn commit đã pin ở #2, **chưa kiểm chứng lại** có áp sạch lên bản mới
không. Việc cần làm: kiểm tra patch áp sạch → chạy thật 2-3 scene public → so
PSNR/SSIM/LPIPS với baseline → áp dụng private set nếu tốt hơn.

### #2 — Mip-Splatting ✅ đã tích hợp code (nhánh `feature/mip-splatting`)

Repo gốc `graphdeco-inria/gaussian-splatting`, từ bản 10/2024, đã tích hợp sẵn "EWA
Filter" của Mip-Splatting làm cờ `--antialiasing` — không cần clone riêng
`autonomousvision/mip-splatting`. Cùng bản có sẵn depth regularization (`--depths`, #3)
và exposure compensation (`--train_test_exp`, #5) — cả 3 hướng giờ chỉ còn là bật cờ.
Đã pin commit `54c035f7834b564019656c3e3fcc3646292f727d`, thêm cờ
`ANTIALIASING`/`DEPTH_PRIOR`/`EXPOSURE_COMP` vào `03_train_3dgs.sh` (mặc định
`ANTIALIASING=1`), sửa `04_render_test_poses.py` tự đọc `cfg_args` để khớp cấu hình
train/render, cập nhật 3 notebook Kaggle. Việc còn lại: chạy thật 2-3 scene public
(Kaggle/GPU thuê — máy local không đủ VRAM), so với baseline đã nộp (58.67), rồi áp
dụng 13 scene.

### #3 — Depth Anything V2 ✅ đã tích hợp code (nhánh `feature/depth-anything-v2`)

`08_generate_depth_priors.py` gọi thẳng `DepthAnythingV2.infer_image()` (KHÔNG dùng
`run.py` gốc — script đó lưu depth 8-bit trong khi `make_depth_scale.py` của repo Inria
cần đọc 16-bit, dùng nhầm mất độ chính xác âm thầm), tự chuẩn hoá 16-bit rồi gọi
`make_depth_scale.py` ra `depth_params.json`. Nhánh riêng `feature/depth-anything-v2`
(từ `feature/mip-splatting`) đã bật `USE_DEPTH_PRIOR=True` mặc định trong 3 notebook
train + pin commit Depth-Anything-V2
(`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`). Việc còn lại: chạy thật trên GPU, so
PSNR/SSIM/LPIPS với Mip-Splatting thuần.

### #4 — Edge loss (Sobel/Canny)

Thêm loss phạt sai lệch biên cạnh (Sobel/Canny so khớp biên render vs train), ép thanh
thép/dây cáp thẳng và sắc nét hơn. Làm sau #2/#3 để đo tác động riêng, tránh nhiễu kết
quả do đổi nhiều thứ cùng lúc. Việc cần làm: thêm 1 hàm loss nhỏ, bật/tắt bằng cờ so
sánh trên public_set.

### #5 — Appearance/exposure embedding ✅ đã có sẵn (cờ `--train_test_exp`)

Affine transform học theo từng ảnh train, tách lệch sáng/màu khỏi hình học cố định — có
sẵn trong repo Inria đã pin ở #2. Đã đối chiếu source: lúc render pose MỚI (test_poses),
pipeline KHÔNG áp exposure (an toàn — không pose nào chưa thấy có exposure đã học), chỉ
giúp train hội tụ tốt hơn. Chỉ đáng bật nếu quan sát rõ lệch màu/sáng giữa ảnh train của
1 scene cụ thể, không phải vấn đề chắc chắn ở mọi scene.

### #6 — Dự phòng

- **2DGS/Scaffold-GS**: trần điểm cao hơn nhưng phải đổi rasterizer + viết lại phần đọc
  sparse/pose, tốn nhiều ngày hơn.
- **NeRF distillation → 3DGS**: train NeRF trước sinh pseudo-GT cho vùng khuyết, tốn gấp
  đôi compute.
- **Diffusion-based view completion (ReconFusion/CAT3D)**: tiềm năng cao nhất cho góc
  khuyết cực đoan nhưng rủi ro hallucinate sai chi tiết cơ khí, codebase phức tạp nhất.

## 4. Luật thi cần tuân thủ

Đề bài chỉ cấm dùng **dữ liệu** (ảnh/video/3D) chứa cùng đối tượng/scene của cuộc thi —
dùng trọng số pretrained của foundation model tổng quát (Mip-Splatting, Depth Anything
V2...) hợp lệ (đề bài mục 3: không giới hạn số model, fine-tune tự do). Mọi repo mới
clone thêm phải **pin commit hash/version** ngay từ đầu, để đáp ứng yêu cầu tái lập nếu
BTC yêu cầu (đề bài mục 10.3) — cả `gaussian-splatting` và `Depth-Anything-V2` đã pin.

## 5. Vận hành thực tế (bài học, không phải hướng cải thiện)

- **Workflow 3-notebook** (`kaggle_public`/`kaggle_private`/`kaggle_submission`, thay
  `kaggle_pipeline.ipynb` cũ) — mỗi notebook train đúng 1 scene/phiên Kaggle vì tổng thời
  gian train 13 scene (~37.5 GPU-giờ, ~2.88 giờ/scene đo từ log thật) không vừa 1 phiên
  Kaggle (~12h). `checkpoints/` (git-ignore) là nơi stage cục bộ các `gs_model/` tải về.
- **Bài học hết đĩa**: chạy gộp nhiều scene 1 lần từng crash thật ("No space left on
  device") ở scene thứ 4 do không dọn `colmap/dense/images/` giữa các scene — đã sửa:
  `CLEANUP_DENSE_IMAGES` (mặc định bật) tự xoá thư mục nặng sau mỗi scene + kiểm tra
  dung lượng đĩa trước khi train (báo lỗi sớm nếu <5GB).
- `06_package_submission.py` tự kiểm tra đủ 8 thư mục/đúng số ảnh/kích thước theo
  `test_poses.csv` (số ảnh **không đều nhau** giữa các scene, xem `Dataset/README.md`),
  có `--check_only` để verify lại 1 zip đã đóng.

## 6. Checklist trước mỗi lần nộp

- [ ] Đủ đúng 8 thư mục scene (tên thật, không phải `scene_001`).
- [ ] Mỗi scene: số ảnh = số dòng `test_poses.csv` scene đó (không đều nhau giữa các scene).
- [ ] Mỗi ảnh đúng kích thước `width×height` đọc từ CSV (không hard-code).
- [ ] Tên file giữ nguyên đuôi `.JPG` như quy ước đã chốt (mục 1).
- [ ] Zip không chứa thư mục rác (`__MACOSX/`...).
- [ ] Test giải nén lại ở máy khác/thư mục sạch để chắc cấu trúc đúng yêu cầu BTC.

## 7. Việc cần làm ngay

- [x] Tích hợp code Mip-Splatting + depth prior + exposure compensation (nhánh
      `feature/mip-splatting` → `feature/depth-anything-v2`).
- [x] Pin commit Depth-Anything-V2.
- [x] Sửa `05_eval_metrics.py` trên `main`: `PSNR_max` mặc định 50 (giải ngược từ điểm
      chấm thật) + gộp điểm đúng luật (trung bình theo scene, không gộp ảnh).
- [ ] Chạy thật trên GPU cho 2-3 scene `public_set`: baseline vanilla vs
      `ANTIALIASING=1` vs `+ DEPTH_PRIOR=1`, so với baseline đã nộp (58.67).
- [ ] Kiểm tra `apply_antenna_patch.py` còn áp sạch lên commit mới pin không, trước khi
      bật `ANTENNA_FOCUS=1` cùng lúc với Mip-Splatting.
- [ ] Nếu tốt hơn baseline → áp dụng private set → nộp đè bản mới (đừng để 58.67 là bản
      cuối cùng).
- [ ] Mọi kỹ thuật mới test trên `public_set` trước, chỉ roll-out private set khi đã
      thấy cải thiện rõ ràng.
