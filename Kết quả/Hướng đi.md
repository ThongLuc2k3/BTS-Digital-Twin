# Hướng đi kỹ thuật — BTS Digital Twin (chốt sau kết quả chấm thật Vòng 1)

## 0. Bối cảnh (ý chính)

- Đã nộp, đã chấm thật trên **private_set1 (8/8 scene khớp)**: **Score 58.67320**
  (PSNR 19.47 dB, SSIM 0.5637, LPIPS 0.2480) — Top 1 hiện ~74.9. Cần lội ngược ~16 điểm.
- Đây là **vanilla 3DGS gốc, không sửa gì**, dùng thẳng sparse COLMAP của BTC (đã xác nhận
  đúng hệ toạ độ — không phải lỗi pose).
- Còn **~22 ngày** tới deadline (30/07), đội 3 người, pipeline COLMAP→train→render→eval→đóng
  gói đã chạy được đầu-cuối. Hệ thống chỉ giữ **bản nộp cuối cùng** → nên nộp đè sớm ngay khi
  có bản tốt hơn, không chờ tới cuối mới nộp.

## 1. Phát hiện quan trọng (ý chính)

- Giải ngược công thức Score từ điểm chấm thật → **`PSNR_max` thực tế ≈ 50** (không phải 30
  như giả định cũ trong `KE_HOACH_VONG1.md`), sai số <0.001%.
- Hệ quả: PSNR bị chia cho 50 nên đóng góp vào điểm rất "nhẹ tay"; **LPIPS (trọng số 0.4) và
  SSIM (0.3) mới là đòn bẩy điểm số thật** — tức là cải thiện *chất lượng cảm quan + cấu
  trúc hình học* quan trọng hơn nhiều so với việc cày PSNR thuần bằng cách tăng iteration.
  → Điều này quyết định thứ tự ưu tiên ở mục 2: mọi hướng ảnh hưởng trực tiếp đến LPIPS/SSIM
  (kiến trúc chống alias, masking, depth) được xếp trên các việc chỉ ảnh hưởng PSNR.
- Nguyên nhân 58.67 bị kẹt: **(a)** ăng-ten/dây cáp là cấu trúc mảnh, Gaussian ellipsoid phải
  phình to để phủ → mờ/răng cưa; **(b)** pose test nằm ở góc không có ảnh train → dễ sinh
  floaters ở vùng khuyết; **(c)** nền trời/mây/cây chiếm phần lớn khung hình, loss hiện tại
  chia đều cho cả nền lẫn ăng-ten nên Gaussian bị lãng phí vào vẽ nền.

## 2. Thứ tự đề xuất — xếp theo khả năng lội ngược dòng thực tế

| # | Hướng đi | Đánh vào nguyên nhân nào (mục 1) | Khả năng lội ngược dòng | Effort/rủi ro |
|---|---|---|---|---|
| 1 | **Antenna-region-focus** (nhánh đã có sẵn, chỉ cần chạy thật) | (c) nền nhiễu | Trung bình | Rất thấp — đã xây ~90% |
| 2 | **Mip-Splatting** (đổi lõi chống alias) | (a) cấu trúc mảnh | **Cao** | Thấp |
| 3 | **Depth Anything V2** (`L_depth`) | (b) floaters vùng khuyết | **Cao** | Trung bình |
| 4 | **Edge loss (Sobel/Canny)** | (a) biên cạnh mờ | Thấp-Trung bình | Rất thấp |
| 5 | **Appearance embedding** | sai lệch màu/sáng giữa ảnh | Thấp | Thấp |
| 6 | *(dự phòng, chỉ làm nếu dư thời gian)* 2DGS / Scaffold-GS / NeRF distillation / diffusion view completion | (a)+(b) triệt để hơn | Cao nhưng rủi ro cao | Cao |

**Cách đọc bảng:** làm theo đúng thứ tự 1→5, mỗi việc test xong trên `public_set` (có ảnh
thật để so PSNR/SSIM/LPIPS) rồi mới áp dụng cho `private_set1` và nộp đè. Việc #6 không nằm
trong kế hoạch chính — chỉ quay lại nếu #1-#5 xong sớm và còn dư ngày/GPU.

## 3. Chi tiết từng hướng

### #1 — Antenna-region-focus (đã có sẵn trong repo)

- Là gì: `pipeline/scripts/07_build_antenna_weights.py` + `apply_antenna_patch.py` — cho 1
  khung pixel bao quanh ăng-ten trên 1 ảnh, tự chiếu ra mask/độ phủ trên toàn bộ ảnh train
  qua sparse COLMAP, rồi vá `train.py` gốc để tăng trọng số L1 trong vùng ăng-ten + ưu tiên
  lấy mẫu camera thấy rõ ăng-ten.
- Vì sao xếp đầu: hạ tầng xong ~90%, **chưa từng chạy GPU thật** — chỉ cần bật cờ
  `ANTENNA_FOCUS`, chạy trên 2-3 scene public, so PSNR/SSIM/LPIPS với baseline.
- ⚠️ **Cập nhật trạng thái nhánh**: nhánh git từng tên `feature/antenna-region-focus` đã được
  **đổi tên thành `feature/mip-splatting`** và merge cập nhật mới nhất từ `main`, để dồn
  vào xây dựng #2/#3 (Mip-Splatting + depth prior) trước. 2 file antenna ở trên vẫn còn
  nguyên trong nhánh này, nhưng `apply_antenna_patch.py` viết cho 1 bản `train.py` CŨ hơn
  commit vừa pin ở #2 — **chưa kiểm chứng lại** có áp sạch lên bản mới hay không. Muốn làm
  #1 cùng lúc với #2/#3, tự chạy thử `apply_antenna_patch.py --gs_repo "$GS_REPO"` trước.
- Việc cần làm: kiểm tra patch còn áp được sạch không → chạy thật → nếu tốt hơn thì áp dụng
  cho private set, nộp đè bản mới ngay.

### #2 — Mip-Splatting ✅ đã tích hợp code xong (nhánh `feature/mip-splatting`)

- **Phát hiện quan trọng** (đối chiếu trực tiếp source thật, không suy đoán): repo GỐC
  `graphdeco-inria/gaussian-splatting`, kể từ bản cập nhật 10/2024, đã **tích hợp sẵn** đúng
  "EWA Filter" của Mip-Splatting làm cờ `--antialiasing` — KHÔNG cần clone riêng
  `autonomousvision/mip-splatting` hay đổi rasterizer như dự tính ban đầu. Cùng bản đó có
  luôn depth regularization (`--depths`, xem #3) và exposure compensation
  (`--train_test_exp`, xem #5) — cả 3 hướng ưu tiên trong bảng trên giờ chỉ còn là bật cờ.
- Vì sao xếp cao: đánh thẳng vào nguyên nhân (a) — răng cưa/mờ ở cấu trúc mảnh — và cải
  thiện trực tiếp LPIPS/SSIM (đòn bẩy điểm số thật theo mục 1), với effort thấp hơn cả dự
  tính ban đầu (không phải "drop-in codebase mới", mà là 1 cờ dòng lệnh trên đúng pipeline
  đang chạy). 2DGS/Scaffold-GS vẫn có trần điểm cao hơn nhưng cần đổi rasterizer thật —
  vẫn xếp ở mục #6 dự phòng.
- **Đã làm**: pin commit `54c035f7834b564019656c3e3fcc3646292f727d` (bản đầu tiên xác nhận
  có đủ antialiasing/depth-reg/exposure), thêm cờ `ANTIALIASING`/`DEPTH_PRIOR`/`EXPOSURE_COMP`
  vào `pipeline/scripts/03_train_3dgs.sh` (mặc định `ANTIALIASING=1`), sửa
  `04_render_test_poses.py` tự đọc `cfg_args` để không lệch cấu hình train/render, cập nhật
  3 notebook Kaggle. Chi tiết: `pipeline/README.md`, `KE_HOACH_VONG1.md` mục 7.
- Việc còn lại: chạy thật trên 2-3 scene `public_set` (Kaggle/GPU thuê — máy local không đủ
  VRAM), so PSNR/SSIM/LPIPS với baseline vanilla đã nộp (58.67), rồi áp dụng 13 scene.

### #3 — Depth Anything V2 (depth prior) ✅ script đã viết xong

- Là gì: dùng model SOTA monocular depth để ước lượng bản đồ chiều sâu cho ảnh train, dùng
  đúng cơ chế depth regularization (`--depths`) đã có sẵn trong repo Inria (không cần tự
  viết `L_depth` — chỉ cần chuẩn bị đúng định dạng depth map + `depth_params.json`).
- Vì sao xếp cao: đánh thẳng vào nguyên nhân (b) — floaters ở vùng pose test không có ảnh
  train trực tiếp — đây chính là bản chất khó nhất của bài toán NVS này.
- **Đã làm**: `pipeline/scripts/08_generate_depth_priors.py` — gọi thẳng
  `DepthAnythingV2.infer_image()` (KHÔNG dùng `Depth-Anything-V2/run.py` gốc, vì script đó
  lưu depth 8-bit trong khi `make_depth_scale.py` của repo Inria cần đọc 16-bit — dùng nhầm
  sẽ mất độ chính xác âm thầm, đã kiểm chứng bằng cách đọc source cả 2 repo), tự chuẩn hoá
  16-bit đúng chuẩn rồi gọi `make_depth_scale.py` để ra `depth_params.json`.
- Việc còn lại: chạy thật trên GPU (tải checkpoint `depth_anything_v2_vitl.pth`), rồi bật
  `DEPTH_PRIOR=1` khi train (làm trên nền Mip-Splatting #2, không cần chờ #2 xong hết 13
  scene mới bắt đầu — 2 việc gối đầu được).

### #4 — Edge loss (Sobel/Canny)

- Là gì: thêm nhánh loss phạt sai lệch biên cạnh (dùng bộ lọc Sobel/Canny so khớp biên giữa
  ảnh render và ảnh train), ép các thanh thép/dây cáp thẳng và sắc nét hơn.
- Vì sao xếp sau #2/#3: chi phí thấp nhưng chỉ nên đo tác động sau khi kiến trúc + depth đã
  ổn định, tránh nhiễu kết quả do đổi nhiều thứ cùng lúc.
- Việc cần làm: thêm 1 hàm loss nhỏ, bật/tắt bằng cờ để so sánh trước/sau trên public_set.

### #5 — Appearance/exposure embedding ✅ đã có sẵn (cờ `--train_test_exp`)

- Là gì: 1 affine transform học được theo từng ảnh train (tách "ánh sáng/màu sắc thay đổi
  giữa các lần chụp" khỏi "hình học cố định") — cũng đã có sẵn trong repo Inria đã pin ở #2,
  không cần tự viết. Đã đối chiếu source (`gaussian_renderer/__init__.py`,
  `scene/gaussian_model.py`): khi render pose MỚI (không có trong tập train, đúng trường hợp
  `test_poses.csv`), pipeline vẫn render KHÔNG áp exposure (đúng/an toàn — không có exposure
  nào đã học cho 1 pose chưa từng thấy) — chỉ ảnh hưởng đến việc train hội tụ tốt hơn, không
  ảnh hưởng cách render ảnh nộp bài.
- Vì sao xếp cuối: chỉ đáng bật nếu quan sát thấy rõ hiện tượng lệch màu/sáng giữa các ảnh
  train trên 1 scene cụ thể — không phải vấn đề chắc chắn xảy ra ở mọi scene.
- Việc cần làm: bật `EXPOSURE_COMP=1` (đã có cờ trong `03_train_3dgs.sh`) cho scene nào thực
  sự thấy artifact màu/sáng khi review render, so PSNR/SSIM/LPIPS trước/sau trên `public_set`.

### #6 — Dự phòng (chỉ cân nhắc nếu #1-#5 xong sớm và dư thời gian/GPU)

- **2DGS / Scaffold-GS**: trần điểm cao hơn Mip-Splatting nhưng phải đổi rasterizer + viết
  lại phần đọc sparse/pose — tốn nhiều ngày hơn.
- **NeRF distillation → 3DGS**: train NeRF trước để sinh pseudo-GT cho vùng khuyết, tốn gấp
  đôi compute (đã tốn ~37.5 GPU-giờ chỉ riêng 1 lượt vanilla 3DGS/13 scene).
- **Diffusion-based view completion (ReconFusion/CAT3D)**: tiềm năng cao nhất cho góc khuyết
  cực đoan nhưng rủi ro hallucinate sai chi tiết cơ khí, codebase phức tạp nhất.

## 4. Lưu ý tuân thủ luật thi

Đề bài chỉ cấm dùng **dữ liệu** (ảnh/video/3D) chứa cùng đối tượng/scene của cuộc thi — dùng
trọng số pretrained của foundation model tổng quát (Mip-Splatting, Depth Anything V2...) là
hợp lệ (mục 3 đề bài: "không giới hạn số lượng model, fine-tune tự do"). Cần **pin commit
hash/version** của mọi repo mới clone thêm ngay từ đầu, để đáp ứng yêu cầu tái lập nếu BTC
yêu cầu (mục 10.3 đề bài).

## 5. Việc cần làm ngay

- [x] Tích hợp code Mip-Splatting (`--antialiasing`) + depth prior
      (`08_generate_depth_priors.py`) + exposure compensation — xong trên nhánh
      `feature/mip-splatting`, chưa chạy GPU thật.
- [ ] Chạy thật trên GPU (Kaggle/thuê) cho 2-3 scene `public_set`: baseline vanilla vs
      `ANTIALIASING=1` vs `+ DEPTH_PRIOR=1`, so PSNR/SSIM/LPIPS với baseline đã nộp (58.67).
- [ ] Kiểm tra `apply_antenna_patch.py` còn áp sạch lên commit mới pin không, trước khi bật
      `ANTENNA_FOCUS=1` cùng lúc với Mip-Splatting.
- [ ] Nếu tốt hơn baseline → áp dụng cho private set → nộp đè bản mới (đừng để 58.67 là bản
      cuối cùng).
- [x] Pin commit cho Depth-Anything-V2 (`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`) — đã
      thêm `git checkout` sau clone ở `08_generate_depth_priors.py` + 3 notebook train,
      trên nhánh `feature/depth-anything-v2`.
- [ ] Mọi kỹ thuật mới đều test trên `public_set` trước, chỉ roll-out private set khi đã thấy
      cải thiện rõ ràng.
