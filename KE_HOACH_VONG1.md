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
3. ⚠️ **Thư mục `sparse/0/` (COLMAP có sẵn) chỉ thực sự dùng được ở đúng 1/8 scene private (`HCM0249`)**. 2 scene khác có file 0-byte (rác), 3 scene không có thư mục sparse. → **Không thiết kế pipeline phụ thuộc sparse có sẵn** — phải tự chạy COLMAP cho gần như toàn bộ 13 scene.
4. Ảnh đã downscale sẵn về **1320×989**, không có EXIF/GPS → SfM phải chạy thuần thị giác.
5. **Rủi ro kỹ thuật lớn nhất của cả vòng thi**: pose trong `test_poses.csv` phải nằm **cùng hệ toạ độ (gauge)** với model 3D mà ta tự dựng từ `train/images/`. Nếu tự chạy COLMAP riêng cho ảnh train và hệ toạ độ ra khác với hệ mà BTC dùng để sinh `test_poses.csv`, ảnh render sẽ sai hoàn toàn (nhìn "đúng scene, sai góc") dù model 3D có đẹp đến đâu. Đây là điều **bắt buộc phải kiểm chứng trước**, không phải giả định.
6. Local machine hiện tại: GPU GTX 1650 4GB — **không đủ** để train 3DGS chất lượng tốt cho scene lớn. Cần thuê GPU ngoài (Colab Pro/Kaggle/RunPod/Vast.ai — 1×3090/4090/A4000 trở lên) cho bước train thật.

## 3. Baseline được đề xuất (đơn giản, đủ để qua vòng 1)

Không cần ensemble nhiều model, không cần DroneSplat/LOBE-GS chuyên biệt ngay từ đầu. Baseline 1 pipeline thống nhất áp dụng cho cả 13 scene:

```
train/images/  ──► COLMAP (feature extract + exhaustive/sequential match + mapper)
                     │  → poses + sparse point cloud (TỰ CHẠY, không dùng sparse có sẵn)
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
| 1 | Hệ toạ độ COLMAP tự chạy có khớp với `test_poses.csv` không? | **Thực nghiệm bắt buộc** (chi tiết Phase 0 bên dưới), dùng scene `HCM0249` (có sparse thật để đối chiếu) + `public_set` (có ảnh test thật để tính PSNR). |
| 2 | 7/8 scene private thiếu sparse hợp lệ — có phải lỗi đóng gói của BTC? | Báo qua kênh hỗ trợ chính thức càng sớm càng tốt (không chờ), kèm bằng chứng (kích thước file 0 byte). Có thể BTC sẽ phát hành bản data vá lỗi. |
| 3 | Tên file PNG nộp bài: giữ nguyên `image_name` (có đuôi `.JPG`) hay đổi đuôi `.png`? | Đề bài viết "tên file theo `image_name`" nhưng ví dụ cấu trúc lại dùng `0001.png`. Hỏi BTC. Mặc định an toàn: dùng **nguyên văn chuỗi `image_name`** làm tên file (kể cả đuôi `.JPG`) vì đó là câu chữ literal của đề bài; làm script cấu hình được để đổi nhanh nếu BTC trả lời khác. |
| 4 | Tên thư mục scene trong zip nộp: `scene_001` (ví dụ minh hoạ trong đề) hay tên thật `HCM0249`, `HNI0131`...? | Dữ liệu thật không hề có tên `scene_001`. Gần như chắc chắn phải dùng **tên thư mục thật của từng scene** trong private_set1. Hỏi BTC nếu còn nghi ngờ, nhưng cứ code theo tên thật trước. |
| 5 | PSNR/SSIM/LPIPS tính trên toàn ảnh hay loại trừ nền/sky? | Không ảnh hưởng cách nộp bài, chỉ ảnh hưởng cách ta tự đánh giá nội bộ trên `public_set`. Cứ tính trên toàn ảnh trước, không cần chờ trả lời mới bắt đầu. |
| 6 | Có được dùng AI hỗ trợ code (Claude Code, Copilot...) không? | Hỏi BTC qua kênh chính thức, làm song song trong lúc chờ trả lời (không phải rủi ro chặn tiến độ). |

**Nguyên tắc:** chỉ có câu hỏi #1 là chặn tiến độ thật sự (blocking) — phải trả lời bằng thực nghiệm trước khi render hàng loạt cho private set. Các câu còn lại có thể vừa hỏi BTC vừa tiếp tục code song song với giả định mặc định.

## 5. Kế hoạch theo tuần (04/07 → 30/07/2026)

### Tuần 0 — Phase 0: Dựng khung + kiểm định giả thuyết hệ toạ độ (04/07 – 06/07)

- [ ] Cài môi trường: COLMAP (CLI hoặc pycolmap), PyTorch + CUDA, `gsplat` hoặc `nerfstudio` (splatfacto), `lpips`/`scikit-image` để tự tính metric.
- [ ] Viết script đọc `test_poses.csv` → build camera intrinsics/extrinsics (chú ý quy ước quaternion COLMAP là world-to-camera: R,t biến điểm world → camera; cần R^T, -R^T·t nếu công cụ NVS cần camera-to-world).
- [ ] **Thực nghiệm quyết định (ưu tiên #1)**: lấy `private_set1/HCM0249/train/images/` (240 ảnh, có sparse thật đi kèm) → tự chạy COLMAP từ đầu trên đúng tập ảnh này → so sánh pose từng ảnh train ước lượng được với pose gốc trong `sparse/0/images.bin` có sẵn (so sau khi align bằng phép biến đổi Sim3/Umeyama giữa 2 tập camera centers).
  - Nếu residual nhỏ (2 hệ toạ độ trùng khớp, hoặc chỉ lệch một phép Sim3 cố định dễ ước lượng) → **an toàn**, tự chạy COLMAP cho mọi scene còn lại, tin tưởng `test_poses.csv` cùng hệ.
  - Nếu lệch nhiều/vô định hình → cần dùng chính phép Sim3 ước lượng được để **align** kết quả COLMAP tự chạy của các scene khác vào "hệ chuẩn" trước khi render (rủi ro cao hơn, cần bàn thêm/hỏi BTC).
- [ ] Thực nghiệm chéo kiểm tra thêm trên `public_set` (1 scene, ví dụ `hcm0031` vì nhỏ nhất – 200 ảnh): chạy COLMAP + 3DGS nhanh (ít iteration) chỉ trên `train/images/`, render tại pose trong `test_poses.csv`, so PSNR với ảnh thật trong `test/images/`. PSNR hợp lý (không phải ảnh nhiễu loạn hoàn toàn) = xác nhận thêm giả thuyết hệ toạ độ đúng.
- [ ] Gửi câu hỏi #2, #3, #4, #6 (mục 4) cho BTC qua kênh hỗ trợ chính thức ngay trong tuần này.

### Tuần 1 — Baseline đầy đủ trên public_set (07/07 – 12/07)

- [ ] Đóng gói pipeline thành script lặp qua từng scene: `colmap_run.py`, `train_3dgs.py`, `render_poses.py`, `eval_metrics.py`.
- [ ] Chạy full baseline (đủ iteration, không rút gọn) trên cả 5 scene `public_set`, đo PSNR/SSIM/LPIPS trên `test/images/` thật.
- [ ] Đo thời gian chạy thực tế/scene (COLMAP + train + render) → nhân với 8 scene private để ước lượng tổng thời gian cần, đối chiếu deadline 30/07 và tốc độ GPU thuê được. Đây là input để quyết định số iteration/độ phân giải dùng cho private set.
- [ ] 12/07: theo dõi livestream giải thích thể lệ — cập nhật ngay nếu có thay đổi format/luật.

### Tuần 2 — Áp dụng cho private_set1 + nộp bài đầu tiên càng sớm càng tốt (13/07 – 19/07)

- [ ] Chạy pipeline cho toàn bộ 8 scene của `private_set1` (dùng sparse có sẵn cho `HCM0249` nếu thực nghiệm Phase 0 xác nhận an toàn/nhanh hơn; 7 scene còn lại tự chạy COLMAP từ đầu).
- [ ] Viết `check_submission.py`: kiểm tra tự động — đủ 8 thư mục scene, đủ số ảnh đúng bằng số dòng trong `test_poses.csv` từng scene, đúng kích thước width×height từng ảnh, đúng tên file, không thiếu/thừa.
- [ ] Đóng gói `submission_round1.zip` và **nộp thử sớm nhất có thể** trong tuần này (không đợi tới hạn) để xác nhận: hệ thống chấm chấp nhận format, không lỗi gì bất ngờ. Nhớ giới hạn 5 lần/ngày + chờ 600s giữa các lần.

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
| 3D/Graphics | Chạy & kiểm chứng COLMAP, thực nghiệm Phase 0 (kiểm định hệ toạ độ), chuẩn hoá input | Sparse reconstruction đã verify/scene + báo cáo thực nghiệm hệ toạ độ |
| Tổng quát/IT | `eval_metrics.py`, `check_submission.py`, đóng gói zip, theo dõi giới hạn nộp bài (5 lần/ngày, 600s), liên hệ BTC các câu hỏi mở | Script chấm điểm nội bộ + zip nộp bài hợp lệ |

## 7. Checklist trước mỗi lần nộp

- [ ] Đủ đúng 8 thư mục scene (đúng tên thật, không phải `scene_001`).
- [ ] Mỗi scene: số ảnh = số dòng dữ liệu trong `test_poses.csv` scene đó (26/50/52/56/60 tuỳ scene — **không đều nhau**, xem bảng ở `Dataset/README.md`).
- [ ] Mỗi ảnh đúng kích thước `width×height` ghi trong CSV (đọc từ CSV, không hard-code 1320×989 dù hiện tại luôn đúng).
- [ ] Tên file đúng theo quy ước đã chốt với BTC (mục 4, câu hỏi #3).
- [ ] Zip không chứa thư mục rác kiểu `__MACOSX/` (nếu nén trên máy Mac, cẩn thận zip mặc định của Finder).
- [ ] Test giải nén lại zip ở máy khác/thư mục sạch để chắc chắn cấu trúc đúng như BTC yêu cầu.
