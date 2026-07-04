# Pipeline — BTS Digital Twin (NVS) Vòng 1

Code thực thi cho kế hoạch ở `../KE_HOACH_VONG1.md`. Toàn bộ script đã được viết
dựa trên việc đối chiếu trực tiếp source thật của COLMAP (`colmap/colmap`, nhánh
`main`, thư mục `src/pycolmap/`) và của `graphdeco-inria/gaussian-splatting`
(fetch trực tiếp từ GitHub, không đoán từ trí nhớ) — xem chú thích đầu mỗi file.

**Chưa chạy được trên máy hiện tại**: máy này không có `colmap`/`pycolmap`/`torch`
cài sẵn, và GPU local (GTX 1650 4GB) không đủ để train 3DGS. Toàn bộ script dưới
đây cần chạy trên máy có GPU CUDA đủ mạnh (thuê Colab Pro/Kaggle/RunPod/Vast.ai,
≥ 8GB VRAM, khuyến nghị ≥ 16GB để thoải mái). **Hãy tự sanity-check từng bước**
(script đã có nhiều assertion/cảnh báo tự động, nhưng vẫn nên nhìn qua output).

## 0. Cài đặt (làm 1 lần trên máy GPU thuê)

```bash
# PyTorch: cài đúng bản khớp CUDA của máy, xem https://pytorch.org/get-started/locally/
pip install -r requirements.txt   # pycolmap, numpy, Pillow, scikit-image, lpips, plyfile, tqdm

# Repo train/render 3DGS — KHÔNG cài qua pip
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
cd ..
export GS_REPO=$(pwd)/gaussian-splatting   # cần export lại mỗi lần mở terminal mới
```

## 1. Thứ tự chạy (đúng theo `KE_HOACH_VONG1.md`)

Sparse `sparse/0/` hợp lệ ở cả 13/13 scene — `01_run_colmap.py` mặc định **dùng
thẳng sparse có sẵn** (chỉ undistort, rất nhanh), không tự chạy lại COLMAP.

### Bước 0 (tuỳ chọn) — Sanity-check hệ toạ độ nếu còn nghi ngờ

```bash
cd scripts
python 02_validate_frame.py
```

So sánh COLMAP tự chạy vs sparse có sẵn của `HCM0249` — chỉ cần chạy nếu muốn
đối chiếu thêm; không còn là điều kiện bắt buộc trước khi làm tiếp (xem
`KE_HOACH_VONG1.md` mục 2, điểm 5).

### Bước 1 — Chuẩn bị dữ liệu COLMAP cho từng scene (dùng sparse có sẵn)

```bash
python 01_run_colmap.py --scene HCM0181            # thử 1 scene public trước
python 01_run_colmap.py --all --split public        # cả 5 scene public
python 01_run_colmap.py --all --split private        # cả 8 scene private
```

Mặc định tự nhận diện sparse hợp lệ và chỉ undistort (vài giây tới vài chục
giây/scene). Chỉ khi 1 scene cụ thể thiếu/hỏng sparse (hoặc muốn ép chạy lại để
đối chiếu) mới cần thêm `--force_own_colmap` (khi đó có thể thêm `--matching
exhaustive` nếu tỉ lệ đăng ký ảnh thấp).

Output: `pipeline/work/<scene>/colmap/dense/{images/,sparse/0/}`.

### Bước 2 — Train 3D Gaussian Splatting

```bash
bash 03_train_3dgs.sh HCM0181
bash 03_train_3dgs.sh HCM0193 HCM0204 hcm0031 hcm0034   # nốt các scene public
```

### Bước 3 — Render + tự chấm điểm trên public_set (làm trước khi đụng private)

```bash
python 04_render_test_poses.py --scene HCM0181
python 05_eval_metrics.py --scene HCM0181
```

Nếu PSNR/SSIM hợp lý (không phải ảnh nhiễu loạn ngẫu nhiên) trên vài scene public
→ pipeline ổn, chuyển sang private set. Nếu tệ bất thường → quay lại Bước 0/1
(khả năng cao là vấn đề hệ toạ độ hoặc COLMAP đăng ký thiếu ảnh).

### Bước 4 — Áp dụng cho 8 scene private_set1

```bash
python 01_run_colmap.py --all --split private
bash 03_train_3dgs.sh HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
    python 04_render_test_poses.py --scene $s
done
```

### Bước 5 — Đóng gói & kiểm tra submission

```bash
python 06_package_submission.py --out ../../submission_round1.zip
```

Script tự kiểm tra đủ 8 scene / đủ ảnh / đúng kích thước trước khi nén, và verify
lại chính file zip vừa tạo. Nếu báo lỗi, KHÔNG nộp — sửa xong chạy lại.

## 2. Log chi tiết nằm ở đâu (console chỉ in tóm tắt, tránh spam khi chạy 13 scene)

Tên file log luôn trùng với tên script sinh ra nó, dễ tra cứu:

| File log | Script sinh ra | Ghi gì |
|---|---|---|
| `work/<scene>/01_run_colmap.log` | `01_run_colmap.py` | Từng bước COLMAP hoặc undistort sparse có sẵn |
| `work/<scene>/02_validate_frame.log` | `02_validate_frame.py` | Từng bước COLMAP khi tự chạy lại để đối chiếu |
| `work/<scene>/03_train_3dgs.log` | `03_train_3dgs.sh` | Toàn bộ output của `train.py` (loss/iteration...) |
| `work/<scene>/04_render_test_poses.log` | `04_render_test_poses.py` | Từng ảnh đã render (tên file, thứ tự) |
| `work/<scene>/colmap/pycolmap_internal_logs/` | (COLMAP nội bộ) | Log rất chi tiết của chính thư viện COLMAP (glog) |

Console/notebook chỉ hiện 1-2 dòng tóm tắt mỗi scene (số ảnh đăng ký, số điểm 3D,
đường dẫn log) — cần xem chi tiết thì mở đúng file log tương ứng ở trên. Nếu
`03_train_3dgs.sh` báo lỗi, nó tự in 50 dòng cuối của `03_train_3dgs.log` ra
console để không phải mò file khi có sự cố.

## 3. Ghi chú quan trọng nằm rải trong code (đọc trước khi chạy)

- `common/poses.py`: quy ước quaternion/translation world→camera, công thức FOV —
  đối chiếu byte-for-byte với `scene/dataset_readers.py` của gaussian-splatting.
- `common/colmap_runner.py`: dùng camera model `SIMPLE_RADIAL` (không phải
  `PINHOLE`) vì giải mã trực tiếp `cameras.bin` gốc của BTC (scene `HCM0249`) cho
  thấy đó là model_id=2=SIMPLE_RADIAL (4 tham số f,cx,cy,k) — khớp với việc mọi
  pose trong `test_poses.csv` luôn có `fx==fy`. Sau mapping, script tự
  `undistort_images` sang PINHOLE sạch trước khi đưa vào 3DGS.
- `scripts/04_render_test_poses.py`: PNG lưu theo `<stem>.png` trong thư mục làm
  việc nội bộ; tên file CUỐI CÙNG trong zip nộp bài do `06_package_submission.py
  --filename_mode` quyết định (mặc định giữ nguyên `image_name`, kể cả đuôi
  `.JPG` gốc — xem `KE_HOACH_VONG1.md` mục 4 câu hỏi #2, chưa có xác nhận BTC).
