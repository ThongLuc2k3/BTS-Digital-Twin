# Dataset README — VAI_NVS_DATA (VIETTEL AI RACE 2026, Vòng 1: BTS Digital Twin / NVS)

Tài liệu này mô tả **chính xác những gì thực sự nằm trong thư mục `Dataset/`** (đã kiểm tra trực tiếp từng file, không suy đoán), phục vụ cho việc xây dựng pipeline Novel View Synthesis (NVS).

## 0. Tóm tắt nhanh

- Tổng dung lượng: **~3.2 GB** (`VAI_NVS_DATA/`), gồm 2 tập: `public_set` (5 scene) và `private_set1` (8 scene) → **13 scene** tất cả.
- Ảnh đã bị **downscale 1/4 so với ảnh gốc** (ghi rõ trong mỗi `README.txt`), độ phân giải quan sát được là **1320×989** cho toàn bộ scene đã kiểm tra (train lẫn test).
- **EXIF đã bị xóa** khỏi ảnh (không có GPS, không có thông tin máy bay).
- ✅ **`sparse/0/` (kết quả COLMAP) có dữ liệu hợp lệ ở CẢ 13/13 scene** (cả `public_set` lẫn `private_set1`). Pipeline nên **dùng thẳng sparse có sẵn**, không cần tự chạy lại COLMAP/feature-matching/bundle-adjustment nữa — tiết kiệm rất nhiều thời gian và giảm hẳn rủi ro lỗi/OOM ở bước đó.

## 1. Cây thư mục

```
Dataset/
└── VAI_NVS_DATA/
    └── phase1/
        ├── public_set/           ← tập công khai: có ĐẦY ĐỦ ảnh train + ảnh test thật (ground-truth) để tự luyện & tự chấm
        │   ├── HCM0181/
        │   ├── HCM0193/
        │   ├── HCM0204/
        │   ├── hcm0031/
        │   └── hcm0034/
        └── private_set1/         ← tập nộp bài chính thức vòng 1: CHỈ có ảnh train, KHÔNG có ảnh test thật (phải tự sinh ảnh)
            ├── HCM0249/
            ├── HCM0254/
            ├── HCM0276/
            ├── HCM1439/
            ├── HNI0131/
            ├── HNI0265/
            ├── HNI0366/
            └── HNI0437/
```

Ghi chú tên scene: tiền tố `HCM` = TP. Hồ Chí Minh, `HNI` = Hà Nội (địa điểm trạm BTS thật). Lưu ý **2 scene trong `public_set` đặt tên chữ thường** (`hcm0031`, `hcm0034`) trong khi các scene còn lại viết hoa (`HCM0181`...) — không nhất quán, cần giữ nguyên chính xác tên thư mục khi code (phân biệt hoa/thường trên Linux).

## 2. Cấu trúc bên trong 1 scene

Mỗi scene (ví dụ `public_set/HCM0181/`) có dạng:

```
HCM0181/
├── README.txt              ← mô tả định dạng, số ảnh, scale factor (giống nhau ở mọi scene, chỉ khác số liệu)
├── train/
│   ├── images/              ← ảnh RGB dùng để dựng lại scene (.JPG, tên gốc kiểu DJI_<timestamp>_<index>_V.JPG)
│   └── sparse/0/             ← sparse reconstruction từ COLMAP: cameras.bin, images.bin, points3D.bin (+ points3D.ply, rigs.bin, frames.bin ở phiên bản COLMAP mới)
└── test/
    ├── images/               ← [CHỈ CÓ Ở public_set] ảnh thật tại các pose mục tiêu — dùng để tự tính PSNR/SSIM/LPIPS
    └── test_poses.csv        ← danh sách pose (góc nhìn) cần render, ở CẢ public_set và private_set1
```

### `test_poses.csv` — cột dữ liệu

```
image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
```

| Cột | Ý nghĩa |
|---|---|
| `image_name` | Tên file ảnh gốc tương ứng (có đuôi `.JPG`, vd `DJI_20241229103827_0207_V.JPG`) — **đã kiểm chứng trùng khớp 100%** với tên file trong `test/images/` ở public_set |
| `qw,qx,qy,qz` | Quaternion xoay camera (world→camera theo quy ước COLMAP) |
| `tx,ty,tz` | Vị trí/tịnh tiến camera |
| `fx,fy,cx,cy` | Nội tham số camera (pinhole). Đã kiểm tra: `fx == fy` luôn đúng ở mọi hàng đã xem → camera vuông, không méo/không cần distortion model phức tạp |
| `width,height` | Độ phân giải ảnh cần render cho đúng pose đó |

Đã kiểm tra toàn bộ 13 scene: **width×height luôn là 1320×989** — đề bài cảnh báo "mỗi scene/pose có thể khác kích thước" nhưng thực tế dữ liệu hiện có đồng nhất. Vẫn nên code tổng quát (đọc width/height từ CSV, không hard-code) để an toàn nếu private test #2 hoặc dữ liệu bổ sung sau này khác đi.

## 3. Bảng số liệu đầy đủ (đã đếm trực tiếp từng scene)

| Scene | Tập | Ảnh train | Ảnh test (GT) | Số pose cần sinh | Sparse COLMAP hợp lệ? |
|---|---|---|---|---|---|
| HCM0181 | public | 240 | 60 | 60 | ✅ |
| HCM0193 | public | 240 | 60 | 60 | ✅ |
| HCM0204 | public | 240 | 60 | 60 | ✅ |
| hcm0031 | public | 200 | 50 | 50 | ✅ |
| hcm0034 | public | 240 | 60 | 60 | ✅ |
| HCM0249 | private | 240 | — | 60 | ✅ |
| HCM0254 | private | 240 | — | 60 | ✅ |
| HCM0276 | private | 240 | — | 60 | ✅ |
| HCM1439 | private | 103 | — | 26 | ✅ |
| HNI0131 | private | 240 | — | 60 | ✅ |
| HNI0265 | private | 205 | — | 52 | ✅ |
| HNI0366 | private | 240 | — | 60 | ✅ |
| HNI0437 | private | 224 | — | 56 | ✅ |

Ghi chú: 2 scene có số ảnh/pose **thấp hơn** khoảng công bố trong đề bài (150–300 ảnh / 40–70 pose): `HCM1439` (103 ảnh / 26 pose) và `HNI0265` (205 ảnh / 52 pose) — không phải lỗi, chỉ là ngoại lệ nhỏ so với con số trung bình BTC nêu.

## 4. Định dạng `sparse/0/`

Đây là output chuẩn của COLMAP, nhưng dùng **định dạng mới** (COLMAP ≥ 3.10, có hỗ trợ multi-rig):

- `cameras.bin` — danh sách camera (ở đây chỉ 1 camera dùng chung cho cả scene, model pinhole/simple, khớp fx trong `test_poses.csv`)
- `images.bin` — pose (quaternion + translation) của từng ảnh train, gắn với camera_id
- `points3D.bin` / `points3D.ply` — point cloud thưa (sparse) tam giác hoá từ feature matching (`points3D.ply` không có ở 1 số scene private, không sao — `gaussian-splatting` tự sinh lại từ `points3D.bin` nếu thiếu)
- `frames.bin`, `rigs.bin` — mở rộng mới của COLMAP cho multi-camera rig (ở đây rig chỉ có 1 camera nên gần như rỗng/tối giản)

Đọc bằng Python: dùng `pycolmap` (khuyến nghị, hỗ trợ format mới) — xem `pipeline/common/colmap_runner.py`.

## 5. Điểm cần lưu ý khi xây dựng pipeline

1. **Ưu tiên dùng thẳng sparse có sẵn** cho mọi scene (cả 13/13 đều hợp lệ) — chỉ cần bước undistort sang PINHOLE sạch trước khi đưa vào 3DGS, KHÔNG cần tự chạy lại feature extraction/matching/bundle-adjustment (xem `pipeline/common/colmap_runner.py::use_provided_sparse`). Việc tự chạy COLMAP từ đầu chỉ cần thiết nếu nghi ngờ chất lượng sparse của 1 scene cụ thể nào đó, hoặc muốn đối chiếu (dùng `--force_own_colmap`).
2. Vì dùng thẳng sparse do chính BTC tạo ra (không tự dựng lại), pose trong `test_poses.csv` gần như chắc chắn cùng hệ toạ độ với sparse đó (cùng 1 lần chạy COLMAP của BTC). Vẫn nên kiểm chứng bằng cách render thử 1 scene `public_set` rồi so PSNR với ảnh thật (xem `pipeline/scripts/05_eval_metrics.py`) trước khi tin tưởng hoàn toàn.
3. Tên file ảnh test (`image_name` trong CSV) giữ nguyên đuôi `.JPG` gốc — cần làm rõ với BTC việc file PNG nộp bài có phải đặt tên **y hệt chuỗi này** (kể cả đuôi `.JPG`) hay phải đổi đuôi thành `.png` (xem mục câu hỏi mở trong file kế hoạch).
