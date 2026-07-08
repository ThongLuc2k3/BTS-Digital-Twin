# Kế hoạch Vòng 1 — VIETTEL AI RACE 2026: BTS Digital Twin (Novel View Synthesis)

> Góc nhìn: mục tiêu vòng 1 **không phải làm SOTA**, mà là **nộp đúng format, đủ số lượng, chất lượng chấp nhận được**, trong 24 đội tốt nhất được chọn tiếp vào Vòng 2. Ưu tiên pipeline đơn giản – chạy được – lặp lại nhanh, hơn là ensemble nhiều model phức tạp. Được nộp lại nhiều lần (5 lần/ngày) nên chiến lược đúng là: **có bài nộp hợp lệ càng sớm càng tốt, rồi cải thiện dần**.

## 1. Tóm tắt đề bài (theo cách hiểu thống nhất)

- Nhiệm vụ: từ 150–300 ảnh drone quanh 1 trạm BTS (`train/images/`), tái dựng 3D và **render ảnh RGB 2D** tại 40–70 pose (góc nhìn) mới được BTC chỉ định trong `test_poses.csv`, cho **8 scene** của `private_set1`.
- Quy trình 2 tầng chuẩn của NVS: (1) SfM/Feature matching để suy ra pose camera của ảnh train + point cloud thưa (COLMAP), (2) train model biểu diễn 3D (3D Gaussian Splatting / NeRF) rồi render tại pose bất kỳ.
- Không cần nộp model 3D — chỉ cần **ảnh PNG** đúng kích thước, đúng tên file, đủ số lượng, đóng gói theo cấu trúc `submission_round1.zip / <scene>/<image>.png`.
- Không giới hạn số model/kỹ thuật, được xử lý thủ công từng bước.
- Hạ tầng train tự lo; hạ tầng chấm (inference) của BTC tương đương 1×A4000 20GB — không liên quan đến việc ta train, chỉ là tham chiếu nếu BTC có tự render lại.
- Deadline: **30/07/2026**. Giới hạn nộp: **5 lần/ngày**, cách nhau **≥600 giây**. Hệ thống giữ **bản nộp cuối cùng**, không phải bản điểm cao nhất — cực kỳ quan trọng: **không được nộp bản thử nghiệm/lỗi ngay trước deadline**.

## 2. Hiện trạng sau khi kiểm tra dữ liệu thật (đã verify, không suy đoán)

Đã kiểm tra toàn bộ `Dataset/VAI_NVS_DATA/phase1/` (chi tiết đầy đủ ở `Dataset/README.md`). Điểm mấu chốt cần biết trước khi code:

1. **`public_set` (5 scene) có ảnh test thật** → dùng để tự luyện + tự chấm PSNR/SSIM/LPIPS trước khi đụng vào private set.
2. **`private_set1` (8 scene) không có ảnh test** — chỉ có `test_poses.csv`. Đây là tập phải nộp bài.
3. ✅ **`sparse/0/` hợp lệ ở CẢ 13/13 scene** → **Pipeline dùng THẲNG sparse có sẵn cho mọi scene**, không cần tự chạy lại COLMAP — tiết kiệm rất nhiều thời gian GPU và giảm hẳn rủi ro lỗi (OOM, đăng ký thiếu ảnh...) ở bước SfM.
4. Ảnh đã downscale sẵn về **1320×989**, không có EXIF/GPS.
5. ✅ **Đã xác nhận hệ toạ độ khớp** — chạy thật scene `hcm0031` (public_set): COLMAP 200/200 ảnh + 211262 điểm 3D, train 3DGS đủ 30000 iteration, render 50 pose test, eval ra PSNR mean=21.689 (min 19.260), SSIM mean=0.6823, LPIPS mean=0.1535 — số liệu hợp lý (không phải ảnh nhiễu loạn), xác nhận sparse của BTC và `test_poses.csv` cùng hệ toạ độ, yên tâm áp dụng cho private set (xem thêm mục 4, câu #1).
6. Local machine hiện tại: GPU GTX 1650 4GB — **không đủ** để train 3DGS chất lượng tốt cho scene lớn. Cần thuê GPU ngoài (Colab Pro/Kaggle/RunPod/Vast.ai — 1×3090/4090/A4000 trở lên) cho bước train thật. Lưu ý: train 3DGS có thể bị **CUDA out of memory** ở scene nhiều chi tiết mảnh (khung thép/dây cáp BTS khiến densify sinh rất nhiều Gaussian) — `pipeline/scripts/03_train_3dgs.sh` đã có sẵn cờ giảm tải (`SH_DEGREE`, `DENSIFY_GRAD_THRESHOLD`, `RESOLUTION`) và tự lưu checkpoint giữa chừng.

## 3. Baseline được đề xuất (đơn giản, đủ để qua vòng 1)

Không cần ensemble nhiều model, không cần DroneSplat/LOBE-GS chuyên biệt ngay từ đầu. Baseline 1 pipeline thống nhất áp dụng cho cả 13 scene:

```
train/images/ + sparse/0/ có sẵn ──► undistort sang PINHOLE sạch (pycolmap, rất nhanh)
                     │  (chỉ tự chạy lại feature extract+match+mapper nếu 1 scene nào đó
                     │   thiếu/hỏng sparse — không còn là bước mặc định nữa)
                     ▼
                 3D Gaussian Splatting  (dùng repo có sẵn, KHÔNG tự viết lại từ đầu:
                                          gsplat / nerfstudio "splatfacto", hoặc repo gốc
                                          inria graphdeco-inria/gaussian-splatting)
                     │  → scene representation đã train
                     ▼
        render tại từng pose trong test_poses.csv (dùng đúng fx,fy,cx,cy,width,height)
                     ▼
        so khớp convention quaternion/translation COLMAP (world-to-cam) khi tạo camera-to-world
                     ▼
              PNG output → đóng gói submission_round1.zip
```

Vì sao chọn 3DGS thay vì NeRF/Nerfacto: train nhanh hơn nhiều (phút thay vì giờ), chất lượng ảnh RGB thường tốt hơn ở scene ngoài trời nhiều chi tiết mảnh (khung thép, dây cáp) — phù hợp thời gian eo hẹp và hạ tầng tự túc. Chỉ cân nhắc ensemble/model khác **sau khi** đã có 1 bài nộp hợp lệ chạy trọn vẹn.

## 4. Câu hỏi/rủi ro cần chốt SỚM (trước khi đổ công sức train hàng loạt)

| # | Vấn đề | Cách xử lý |
|---|---|---|
| 1 | Hệ toạ độ sparse có sẵn có khớp với `test_poses.csv` không? | ✅ **Đã xác nhận** — xem mục 2, item 5 (render thật `hcm0031`, PSNR mean 21.689, hợp lý). |
| 2 | Tên file PNG nộp bài: giữ nguyên `image_name` (có đuôi `.JPG`) hay đổi đuôi `.png`? | ✅ **Đã chốt: dùng `.JPG`** (giữ nguyên chuỗi `image_name`, không đổi đuôi) — khớp sẵn với `--filename_mode literal` (mặc định) của `06_package_submission.py`, không cần sửa code. |
| 3 | Tên thư mục scene trong zip nộp: `scene_001` (ví dụ minh hoạ trong đề) hay tên thật `HCM0249`, `HNI0131`...? | ✅ **Đã xác nhận: dùng tên thật** — đọc toàn văn đề bài chính thức (`Đề bài.md`), `scene_001`/`scene_002` chỉ là ví dụ minh hoạ định dạng, không phải yêu cầu đặt tên theo nghĩa đen. Đã code theo tên thật (`06_package_submission.py`). |
| 4 | Công thức Score chính thức đã biết (`Score = 0.4×(1−LPIPS) + 0.3×SSIM + 0.3×PSNR_norm`, `PSNR_norm = clamp(PSNR/PSNR_max, 0, 1)`) — nhưng giá trị `PSNR_max` là bao nhiêu? | ✅ **Đã suy ra được, không còn là ẩn số**: từ kết quả chấm thật trên `private_set1` (Score=58.67320, PSNR=19.471466, SSIM=0.563734, LPIPS=0.248042 — xem `Kết quả/Kết quả chấm.png`), giải ngược công thức ra `PSNR_max ≈ 50.0` (sai số <0.001%). `05_eval_metrics.py` đã đổi mặc định từ `--psnr_max 30.0` sang `--psnr_max 50.0` (chi tiết ở `Kết quả/Hướng đi.md` mục 1). |
| 5 | Có được dùng AI hỗ trợ code (Claude Code, Copilot...) không? | ✅ **Đã xác nhận: được dùng.** |

**Nguyên tắc:** không có câu hỏi nào thực sự chặn tiến độ (blocking) — có thể vừa hỏi BTC vừa tiếp tục code song song với giả định mặc định.

## 5. Kế hoạch theo tuần (04/07 → 30/07/2026)

### Tuần 0 — Dựng khung + sanity-check hệ toạ độ (04/07 – 06/07)

✅ Đã xong toàn bộ — môi trường, COLMAP, sanity-check train+render+eval trên `hcm0031`
(xem mục 2 item 5), và cả 3 câu hỏi liên quan cũng đã được chốt (mục 4, câu #2/#3/#5).

### Tuần 1 — Baseline đầy đủ trên public_set (07/07 – 12/07)

- [ ] Chạy `pipeline/kaggle_public.ipynb` cho từng scene còn lại của `public_set` (đổi
      biến `SCENE`, 1 phiên Kaggle/scene) để có đủ PSNR/SSIM/LPIPS trên cả 5 scene —
      hiện mới xong `hcm0031`, còn `hcm0034`, `HCM0181`, `HCM0193`, `HCM0204`.
- [ ] 12/07: theo dõi livestream giải thích thể lệ — cập nhật ngay nếu có thay đổi format/luật.

### Tuần 2 — Áp dụng cho private_set1 + nộp bài đầu tiên càng sớm càng tốt (13/07 – 19/07)

- [ ] Chạy `pipeline/kaggle_private.ipynb` cho từng scene của `private_set1` (đổi biến
      `SCENE`, 1 phiên Kaggle/scene, không có eval vì private không có ground-truth —
      chỉ xem render trực quan) — có thể chạy song song 2 phiên Kaggle để rút ngắn thời gian.
- [ ] Dùng `pipeline/kaggle_submission.ipynb` để tải 8 checkpoint (qua Google Drive hoặc
      `checkpoints/` cục bộ nếu đã stage sẵn), re-render, rồi đóng gói bằng
      `pipeline/scripts/06_package_submission.py` (đã tự kiểm tra đủ 8 thư mục/đúng số
      ảnh/kích thước theo `test_poses.csv` — dùng `--check_only` để verify lại 1 zip đã
      đóng). **Nộp thử sớm nhất có thể** trong tuần này, nhớ giới hạn 5 lần/ngày + chờ
      600s giữa các lần.

### Tuần 3 — Tinh chỉnh chất lượng (20/07 – 27/07)

- [ ] Dựa trên kết quả tuần 2 (nếu có leaderboard/feedback điểm), cải thiện: tăng iteration train, lọc điểm nhiễu point cloud, xử lý riêng vật thể mảnh (dây cáp/khung thép) nếu quan sát thấy artifact rõ trên `public_set`.
- [ ] Nếu baseline ổn định và còn dư thời gian/compute: thử nghiệm thêm 1 phương án so sánh (ví dụ Nerfacto hoặc tăng số Gaussian) **chỉ trên public_set trước**, không đổi pipeline chính cho private set nếu chưa chứng minh tốt hơn baseline hiện tại.
- [ ] Nộp lại định kỳ (vài ngày/lần) để luôn có 1 bản nộp hợp lệ mới nhất trên hệ thống — vì hệ thống chỉ giữ **bản cuối cùng**.

### Tuần 4 — Chốt bài (28/07 – 30/07)

- [ ] Chạy lại `check_submission.py` lần cuối trên bản nộp dự kiến là bản cuối.
- [ ] Nộp bản cuối **sớm hơn deadline ít nhất nửa ngày** (không nộp phút 89) để có thời gian xử lý nếu hệ thống lỗi/timeout/nghẽn do nhiều đội nộp cùng lúc.
- [ ] Sau khi nộp bản cuối, **không nộp thêm gì nữa** trừ khi chắc chắn bản mới tốt hơn — vì bản cuối cùng trước deadline mới được tính, nộp nhầm bản lỗi/dở sẽ ghi đè bản tốt.

## 6. Phân công (rút gọn, thực tế cho 3 thành viên)

| Vai trò | Việc chính | Sản phẩm bàn giao |
|---|---|---|
| CV core | Train 3DGS từng scene, render pose test, tối ưu chất lượng | Model weights + ảnh render/scene |
| 3D/Graphics | Kiểm tra chất lượng sparse có sẵn từng scene, chuẩn hoá input, chạy `02_validate_frame.py`/đối chiếu khi nghi ngờ | Xác nhận sparse dùng được/scene + báo cáo nếu phát hiện bất thường |
| Tổng quát/IT | `eval_metrics.py`, `check_submission.py`, đóng gói zip, theo dõi giới hạn nộp bài (5 lần/ngày, 600s), liên hệ BTC các câu hỏi mở | Script chấm điểm nội bộ + zip nộp bài hợp lệ |

## 7. Checklist trước mỗi lần nộp

- [ ] Đủ đúng 8 thư mục scene (đúng tên thật, không phải `scene_001`).
- [ ] Mỗi scene: số ảnh = số dòng dữ liệu trong `test_poses.csv` scene đó (26/50/52/56/60 tuỳ scene — **không đều nhau**, xem bảng ở `Dataset/README.md`).
- [ ] Mỗi ảnh đúng kích thước `width×height` ghi trong CSV (đọc từ CSV, không hard-code 1320×989 dù hiện tại luôn đúng).
- [ ] Tên file đúng theo quy ước đã chốt với BTC (mục 4, câu hỏi #2).
- [ ] Zip không chứa thư mục rác kiểu `__MACOSX/` (nếu nén trên máy Mac, cẩn thận zip mặc định của Finder).
- [ ] Test giải nén lại zip ở máy khác/thư mục sạch để chắc chắn cấu trúc đúng như BTC yêu cầu.

## 8. Tiến độ thực tế & hướng cải thiện thêm (tham chiếu)

1. **Workflow 3-notebook** thay `kaggle_pipeline.ipynb` cũ (đã xoá): `pipeline/kaggle_public.ipynb`
   / `pipeline/kaggle_private.ipynb` / `pipeline/kaggle_submission.ipynb` — mỗi notebook
   train đúng 1 scene/phiên Kaggle (đổi biến `SCENE`), vì tổng thời gian train 13 scene
   (~37.5 GPU-giờ, xem item 2 dưới) không vừa 1 phiên Kaggle (~12h). `checkpoints/`
   (thư mục gốc, git-ignore) là nơi stage cục bộ các `gs_model/` tải về từ Kaggle Output.
2. **Thời gian train thật đo được**: ~2.88 giờ/scene (30000 iteration, đo từ log thật
   3 scene đã train xong + 1 scene ngoại suy) → tổng ước tính ~37.5 GPU-giờ cho 13 scene.
3. **Bài học sự cố hết đĩa**: chạy gộp nhiều scene 1 lần từng crash thật
   ("OSError: No space left on device") ở scene thứ 4, ngay tại checkpoint iteration
   15000, do không dọn `colmap/dense/images/` giữa các scene. Đã sửa trong
   `pipeline/scripts/03_train_3dgs.sh`: `CLEANUP_DENSE_IMAGES` (mặc định bật) tự xoá
   thư mục nặng sau mỗi scene, cộng thêm kiểm tra dung lượng đĩa trước khi train mỗi
   scene (báo lỗi sớm nếu còn dưới 5GB).
4. `pipeline/scripts/06_package_submission.py` đã thay thế vai trò `check_submission.py`
   dự kiến ban đầu ở mục 5 (Tuần 2) — tự kiểm tra đủ 8 thư mục/đúng số ảnh/kích thước,
   có `--check_only` để verify lại 1 zip đã đóng.
5. ✅ **Khả năng tái lập — ĐÃ pin commit** cho `graphdeco-inria/gaussian-splatting`:
   `54c035f7834b564019656c3e3fcc3646292f727d` (xem mục 7 bên dưới — commit này còn
   được chọn vì nó là bản ĐẦU TIÊN có antialiasing/depth-reg/exposure, không phải
   chọn tuỳ ý). `03_train_3dgs.sh`, `pipeline/README.md`, `pipeline/requirements.txt`
   và cả 3 notebook Kaggle đều đã cập nhật lệnh `git checkout <commit>` sau khi
   clone. Depth-Anything-V2 (dùng cho depth prior, mục 7) hiện CHƯA pin commit
   (repo ít thay đổi hơn, rủi ro thấp hơn) — có thể pin thêm sau nếu cần chắc chắn
   tuyệt đối.
6. **Nhánh `feature/antenna-region-focus`** — hướng cải thiện cho artifact mờ ở
   ăn-ten/vật thể mảnh quan sát được trên render thật của `hcm0031`:
   - `pipeline/scripts/07_build_antenna_weights.py` — từ 1 ảnh + khung pixel người dùng
     khoanh quanh ăn-ten, suy ra bbox 3D từ điểm sparse tương ứng, chiếu ra bbox 2D +
     độ phủ trên từng ảnh train.
   - `pipeline/scripts/apply_antenna_patch.py` — vá `train.py` gốc (8 chỗ, idempotent):
     thêm L1 loss có trọng số trong vùng ăn-ten + lấy mẫu camera có trọng số ưu tiên
     view thấy rõ ăn-ten.
   - Cờ `ANTENNA_FOCUS`/`ENABLE_ANTENNA_FOCUS` mặc định **tắt** trong `03_train_3dgs.sh`
     và 2 notebook train — không ảnh hưởng hành vi mặc định nếu không bật.
   - **Trạng thái: mới kiểm thử bằng mock/logic (chưa chạy GPU/Kaggle thật), chưa merge
     vào `main`.** ⚠️ Nhánh đó đã được **đổi tên thành `feature/mip-splatting`** và
     pivot hẳn sang hướng Mip-Splatting (mục 7) — `apply_antenna_patch.py` viết cho
     1 bản `train.py` CŨ hơn commit vừa pin ở mục 5, **chưa được kiểm chứng lại** có
     áp sạch lên bản mới hay không. Muốn tiếp tục hướng antenna-focus, hãy tự chạy
     `apply_antenna_patch.py --gs_repo "$GS_REPO"` để kiểm tra trước khi tin nó vẫn đúng.
7. **Hướng đi Mip-Splatting** (`Kết quả/Hướng đi.md` mục 2, hạng #2 — nhánh
   `feature/mip-splatting`, trước đây là `feature/antenna-region-focus`): phát hiện
   quan trọng — repo GỐC `graphdeco-inria/gaussian-splatting` (bản cập nhật 10/2024,
   đã đối chiếu trực tiếp source: `README.md`, `gaussian_renderer/__init__.py`,
   `arguments/__init__.py`, `train.py`, `utils/make_depth_scale.py`) **đã tích hợp
   sẵn** đúng "EWA Filter" của Mip-Splatting (`--antialiasing`), depth regularization
   (`--depths`, dùng Depth Anything V2) và exposure compensation (`--train_test_exp`)
   — KHÔNG cần clone riêng `autonomousvision/mip-splatting` hay đổi rasterizer, chỉ
   cần bật cờ + (cho depth) chuẩn bị depth map trước. Đã pin commit
   `54c035f7834b564019656c3e3fcc3646292f727d` (đầu tiên có đủ 3 tính năng), thêm
   `pipeline/scripts/08_generate_depth_priors.py` (sinh depth map 16-bit đúng chuẩn
   `make_depth_scale.py` cần — **không** dùng thẳng `Depth-Anything-V2/run.py` vì
   script đó lưu depth 8-bit, sai định dạng 16-bit mà `make_depth_scale.py` cần,
   gây mất độ chính xác âm thầm nếu dùng nhầm). Chi tiết cờ mới: xem
   `pipeline/scripts/03_train_3dgs.sh` (biến `ANTIALIASING`/`DEPTH_PRIOR`/`EXPOSURE_COMP`)
   và `pipeline/README.md`.
