# Dataset README — VAI_NVS_DATA (VIETTEL AI RACE 2026, Vòng 1: BTS Digital Twin / NVS)

Tài liệu này mô tả **chính xác những gì thực sự nằm trong thư mục `Dataset/`** (đã kiểm tra trực tiếp từng file, không suy đoán), phục vụ cho việc xây dựng pipeline Novel View Synthesis (NVS).

## 0. Tóm tắt nhanh

- Tổng dung lượng: ~948 MB (`VAI_NVS_DATA/`), gồm 2 tập: `public_set` (5 scene) và `private_set1` (8 scene) → **13 scene** tất cả.
- Ảnh đã bị **downscale 1/4 so với ảnh gốc** (ghi rõ trong mỗi `README.txt`), độ phân giải quan sát được là **1320×989** cho toàn bộ scene đã kiểm tra (train lẫn test).
- **EXIF đã bị xóa** khỏi ảnh (không có GPS, không có thông tin máy bay) → không thể dùng GPS prior, bắt buộc phải Structure-from-Motion thuần thị giác.
- ⚠️ Phát hiện quan trọng: thư mục `sparse/0/` (kết quả COLMAP) mà BTC hứa cung cấp **chỉ thực sự có dữ liệu hợp lệ ở 1/8 scene của `private_set1`** (xem mục 4). Không nên thiết kế pipeline phụ thuộc vào sparse có sẵn.

## 1. Cây thư mục

```
Dataset/
└── VAI_NVS_DATA/
    ├── __MACOSX/                 ← rác do máy Mac tạo khi nén zip (file "._*"), KHÔNG chứa dữ liệu thật, có thể xoá an toàn (908 KB)
    └── phase1/
        ├── .DS_Store             ← rác macOS, có thể xoá
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
│   └── sparse/0/             ← (nếu có) sparse reconstruction từ COLMAP: cameras.bin, images.bin, points3D.bin (+ points3D.ply, rigs.bin, frames.bin ở phiên bản COLMAP mới)
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
| HCM0181 | public | 240 | 60 | 60 | ✗ (thư mục rỗng) |
| HCM0193 | public | 240 | 60 | 60 | ✗ |
| HCM0204 | public | 240 | 60 | 60 | ✗ |
| hcm0031 | public | 200 | 50 | 50 | ✗ |
| hcm0034 | public | 240 | 60 | 60 | ✗ |
| HCM0249 | private | 240 | — | 60 | ✅ **có dữ liệu thật** |
| HCM0254 | private | 240 | — | 60 | ✗ (6 file, toàn bộ 0 byte) |
| HCM0276 | private | 240 | — | 60 | ✗ (6 file, toàn bộ 0 byte) |
| HCM1439 | private | 103 | — | 26 | ✗ (không có thư mục sparse) |
| HNI0131 | private | 240 | — | 60 | ✗ (5 file, toàn bộ 0 byte) |
| HNI0265 | private | 205 | — | 52 | ✗ (không có thư mục sparse) |
| HNI0366 | private | 240 | — | 60 | ✗ (không có thư mục sparse) |
| HNI0437 | private | 224 | — | 56 | ✗ (5 file, toàn bộ 0 byte) |

Nhận xét:
- `public_set` **không hề có sparse** ở bất kỳ scene nào → phải tự chạy COLMAP hoàn toàn khi luyện tập, đúng như thực tế sẽ gặp ở phần lớn `private_set1`.
- Trong `private_set1`, chỉ **HCM0249** có sparse thật (COLMAP nhị phân hợp lệ: `cameras.bin` 64B đọc được header, `points3D.ply` là PLY binary hợp lệ với 165.726 điểm). Còn lại là file rỗng (0 byte, gần như chắc chắn lỗi đóng gói dữ liệu của BTC — nên báo qua kênh hỗ trợ chính thức) hoặc thiếu hẳn thư mục.
- 2 scene có số ảnh/pose **thấp hơn** khoảng công bố trong đề bài (150–300 ảnh / 40–70 pose): `HCM1439` (103 ảnh / 26 pose) và `HNI0265` (205 ảnh / 52 pose). Không phải lỗi của bạn — cứ xử lý bình thường, chỉ là ngoại lệ nhỏ so với con số trung bình BTC nêu.

## 4. Định dạng `sparse/0/` (khi có dữ liệu — case HCM0249)

Đây là output chuẩn của COLMAP, nhưng dùng **định dạng mới** (COLMAP ≥ 3.10, có hỗ trợ multi-rig):

- `cameras.bin` — danh sách camera (ở đây chỉ 1 camera dùng chung cho cả scene, model pinhole/simple, khớp fx trong `test_poses.csv`)
- `images.bin` — pose (quaternion + translation) của từng ảnh train, gắn với camera_id
- `points3D.bin` / `points3D.ply` — point cloud thưa (sparse) tam giác hoá từ feature matching
- `frames.bin`, `rigs.bin` — mở rộng mới của COLMAP cho multi-camera rig (ở đây rig chỉ có 1 camera nên gần như rỗng/tối giản)

Đọc bằng Python: dùng `pycolmap` (khuyến nghị, hỗ trợ format mới) hoặc script `read_write_model.py` chính thức của COLMAP (cần bản cập nhật hỗ trợ rigs/frames nếu dùng scene HCM0249).

## 5. Điểm cần lưu ý khi xây dựng pipeline (rút ra từ việc khảo sát dữ liệu)

1. **Không phụ thuộc vào `sparse/` có sẵn** — chỉ 1/13 scene có dữ liệu dùng được. Pipeline phải tự chạy COLMAP (hoặc SfM khác) cho **mọi scene**, kể cả khi thư mục `sparse/0/` tồn tại nhưng rỗng.
2. **Rủi ro lớn nhất: hệ quy chiếu (coordinate frame).** Vì mỗi lần chạy SfM sẽ ra một hệ toạ độ/scale riêng (gauge freedom của SfM đơn mắt), câu hỏi sống còn là: pose trong `test_poses.csv` có nằm **cùng hệ toạ độ** với sparse reconstruction mà ta tự dựng từ `train/images/` hay không? Nếu không, ảnh render ra sẽ sai hoàn toàn dù model 3D dựng đúng. `public_set` (có ảnh test thật) và scene `HCM0249` (có sparse thật để đối chiếu) là 2 nơi duy nhất có thể **kiểm chứng giả thuyết này trước khi chạy toàn bộ 13 scene** — xem kế hoạch kiểm định ở file kế hoạch (`KE_HOACH_VONG1.md`).
3. **`__MACOSX/` và mọi `.DS_Store`** đều là rác, không phải dữ liệu thi — bỏ qua hoặc xoá khi cần gọn thư mục.
4. **Không có ground-truth train pose độc lập** để tự kiểm chứng SfM (ngoại trừ HCM0249) — nghĩa là chất lượng SfM tự chạy phải được đánh giá gián tiếp qua chất lượng render trên `public_set`.
5. Tên file ảnh test (`image_name` trong CSV) giữ nguyên đuôi `.JPG` gốc — cần làm rõ với BTC việc file PNG nộp bài có phải đặt tên **y hệt chuỗi này** (kể cả đuôi `.JPG`) hay phải đổi đuôi thành `.png` (xem mục câu hỏi mở trong file kế hoạch).
