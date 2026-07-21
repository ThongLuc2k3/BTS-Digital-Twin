# Hướng xử lý scene BTS trên `round1 public_set`

File này thay cho transcript tranh luận dài. Mục tiêu là chốt hướng kỹ thuật, trạng thái hiện tại, và việc phải làm tiếp.

## Kết luận đã chốt

Hướng xử lý ưu tiên hiện tại là pipeline một model nhiều bước:

`P0 -> M0 -> B2 -> C/F -> A`

Trong đó:

- `P0`: xác minh và khôi phục dữ liệu benchmark `round1 public_set` đủ sạch để đo.
- `M0`: đo trên `round1 public` bằng `full-image`, `tower-crop`, `skyline-crop`.
- `B2`: `COLMAP dense-stereo + depth regularization` để xử lý `background xa`, `skyline`, `floater`.
- `C/F`: refinement hoặc fine-tune cục bộ cho `trụ`, `anten`, `dây`.
- `A`: tuning cuối và chọn checkpoint theo `full-image score`.

Các nhánh `E` (2 model / 2-stage) và `G` (đổi backbone) chỉ mở nếu pipeline trên không thắng rõ baseline trên `round1 public`.

## Vì sao chọn hướng này

- `A` đơn lẻ không đủ mạnh vì lỗi chính không phải chỉ là thiếu hội tụ.
- `B2` đánh đúng bệnh `global geometry yếu ở vùng xa`.
- `C/F` đánh đúng bệnh `thin structure` của trụ BTS.
- Pipeline một model nhiều bước rẻ và an toàn hơn `E` hoặc `G`, nhưng vẫn mạnh hơn tune thuần.
- `round1 public` có GT thật nên là nơi quyết định đúng sai trước khi mang bất kỳ kỹ thuật nào sang `round2`.

## Bằng chứng cần giữ lại

### 1. Baseline `hcm0031`

Theo các CSV đang có trong [pipeline/work/hcm0031](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/work/hcm0031):

- `full-image`: `PSNR 21.6938`, `SSIM 0.6819`, `LPIPS 0.1542`, `Score 0.6731`
- `tower-crop`: `PSNR 23.2912`, `SSIM 0.7287`, `LPIPS 0.1298`, `Score 0.7064`
- `skyline-crop`: `PSNR 20.4286`, `SSIM 0.6298`, `LPIPS 0.1829`, `Score 0.6384`

Diễn giải ngắn:

- `skyline-crop` tệ hơn rõ so với `full-image`, khớp với chẩn đoán lỗi nền xa.
- `tower-crop` hiện vẫn là crop theo hộp chiếu, chưa phải mask pixel thật của trụ; dùng để dẫn đường, không dùng thay cho tiêu chí chốt.

### 2. Bằng chứng thực nghiệm cũ để loại `B1` và hạ ưu tiên `C` kiểu cũ

Các commit đã có:

- `9383e23`: chốt Stage 1 cho `HCM0421`
- `8e88bc0`: kết quả thật `DEPTH_PRIOR=1`
- `b696ff3`: sửa bug `antenna-focus`

Các số cần nhớ:

- baseline `A = 0.6616`
- `depth-prior = 0.644`
- `antenna-focus = 0.6611`

Lưu ý:

- `depth-prior` bị confound vì phải hạ `SH_DEGREE 3 -> 2` để tránh OOM.
- Dù vậy, đây vẫn là bằng chứng thật duy nhất đang có để tránh đề xuất lại `B1` hoặc thổi phồng `antenna-focus` như thể chưa từng thử.

## Trạng thái hiện tại trong repo

### `P0` cho `hcm0031`

Trong workspace hiện tại, `P0` cho `hcm0031` đã PASS:

```bash
python3 pipeline/scripts/verify_round1_public_restore.py \
  --dataset_root "Dataset/VAI_NVS_DATA/phase1/public_set" \
  --out_csv "pipeline/work/p0_round1_public_verify_current.csv"
```

Kết quả kiểm tra mới nhất:

- `hcm0031/train/images`: `count=200`, `unique=200`, `suspicious=no`
- `hcm0031/test/images`: `count=50`, `unique=50`, `suspicious=no`

File audit/verify đang có:

- [pipeline/work/p0_round1_public_audit_current.csv](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/work/p0_round1_public_audit_current.csv)
- [pipeline/work/p0_round1_public_verify_current.csv](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/work/p0_round1_public_verify_current.csv)

### Phạm vi benchmark local hiện tại

Dataset local hiện chỉ có:

- `Dataset/VAI_NVS_DATA/phase1/public_set/hcm0031`

Nghĩa là:

- `P0` đã xong cho scene đang benchmark là `hcm0031`
- nhưng chưa có đủ 5 scene `public_set` để gọi là benchmark đầy đủ toàn bộ `round1 public`

### `M0` cho `hcm0031`

`M0` đã có code và đã có số liệu thật:

- [pipeline/scripts/eval_round1_metrics.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/eval_round1_metrics.py)
- [pipeline/scripts/estimate_object_bbox3d.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/estimate_object_bbox3d.py)
- [pipeline/work/hcm0031/eval_metrics_m0.csv](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/work/hcm0031/eval_metrics_m0.csv)
- [pipeline/work/hcm0031/eval_metrics_m0_tower_crop.csv](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/work/hcm0031/eval_metrics_m0_tower_crop.csv)
- [pipeline/work/hcm0031/eval_metrics_m0_skyline_crop.csv](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/work/hcm0031/eval_metrics_m0_skyline_crop.csv)

## Bản hành động ngắn gọn

### Trường hợp 1: tiếp tục tối ưu trên `hcm0031` ngay

1. Giữ `hcm0031` làm benchmark chính cục bộ.
2. Chạy pilot `B2` trên `hcm0031`:
   - dựng `dense stereo`
   - đo thời gian `patch_match_stereo` và `stereo_fusion`
   - xác nhận artifact/chi phí trước khi full run
3. Tích hợp `depth regularization` vào training hoặc fine-tune từ baseline đang có.
4. Chấm lại bằng đủ `M0`:
   - `full-image` là tiêu chí chốt
   - `tower-crop` và `skyline-crop` chỉ để giải thích kỹ thuật đang giúp phần nào
5. Nếu `B2` giúp nền xa nhưng trụ vẫn yếu, chuyển sang `C/F`.
6. Chỉ sau `B2 -> C/F` mới làm `A`.

### Trường hợp 2: muốn khôi phục benchmark rộng hơn `hcm0031`

1. Đồng bộ thêm 4 scene còn thiếu của `round1 public_set` từ nguồn sạch.
2. Chạy lại:

```bash
python3 pipeline/scripts/audit_round1_public_images.py \
  --dataset_root "Dataset/VAI_NVS_DATA/phase1/public_set" \
  --out_csv "pipeline/work/p0_round1_public_audit.csv"
```

3. Chạy lại:

```bash
python3 pipeline/scripts/verify_round1_public_restore.py \
  --dataset_root "Dataset/VAI_NVS_DATA/phase1/public_set" \
  --out_csv "pipeline/work/p0_round1_public_verify.csv"
```

4. Chỉ khi các scene mới không còn pattern `constant-size truncation` mới mở rộng `M0` sang chúng.

## Kế hoạch thực thi `P0` trong repo hiện tại

### Cái đã có sẵn

- [P0_ROUND1_PUBLIC.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/P0_ROUND1_PUBLIC.md): mô tả ngắn `P0`
- [P0_RESTORE_CHECKLIST.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/P0_RESTORE_CHECKLIST.md): checklist restore/verify
- [pipeline/scripts/audit_round1_public_images.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/audit_round1_public_images.py): audit file size
- [pipeline/scripts/verify_round1_public_restore.py](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/scripts/verify_round1_public_restore.py): PASS/FAIL sau restore
- [pipeline/kaggle_round1_p0_restore_verify.ipynb](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/pipeline/kaggle_round1_p0_restore_verify.ipynb): notebook kiểm tra `P0` trên Kaggle

### Việc cần làm thêm nếu muốn benchmark đủ `round1 public`

1. Xác định nguồn sạch chứa đủ 5 scene `public_set`.
2. Đồng bộ về đúng cấu trúc:
   - `Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/train/images`
   - `Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/train/sparse/0`
   - `Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/test/images`
   - `Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/test/test_poses.csv`
3. Chạy `audit`.
4. Chạy `verify`.
5. Chỉ sau đó mới coi `P0` hoàn tất cho benchmark toàn bộ `round1 public`.

## Tiêu chí quyết định bước tiếp theo

- Nếu `B2` tăng `full-image score` rõ rệt trên `hcm0031`, tiếp tục hoàn thiện `B2`.
- Nếu `B2` chỉ giúp `skyline-crop` nhưng không giúp `full-image`, cân nhắc giảm phạm vi hoặc đổi cách regularize.
- Nếu `B2` xong mà `tower-crop` vẫn là bottleneck, mở `C/F`.
- Nếu `B2 + C/F + A` vẫn không thắng rõ baseline, chuyển nhánh sang `E` hoặc `G`.

## Trạng thái ngắn gọn

- Hướng kỹ thuật đã chốt: `P0 -> M0 -> B2 -> C/F -> A`
- `P0` local cho `hcm0031`: đã PASS
- `M0` cho `hcm0031`: đã có số liệu
- Benchmark local toàn bộ `round1 public`: chưa đủ scene
- Việc hợp lý nhất ngay bây giờ: chạy pilot `B2` trên `hcm0031`, đồng thời chỉ mở rộng `P0` nếu thật sự cần benchmark trên nhiều scene hơn
