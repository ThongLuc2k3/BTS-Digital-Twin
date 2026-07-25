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

## Tổng hợp tranh luận 2026-07-24

Phần này cập nhật lại hướng đi sau khi đã đối chiếu:

- [trick/README.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/trick/README.md)
- [trick/hcm0031/experiment_matrix.csv](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/trick/hcm0031/experiment_matrix.csv)
- [.ai-debate/01_codex_proposal.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/.ai-debate/01_codex_proposal.md)
- [.ai-debate/02_claude_review.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/.ai-debate/02_claude_review.md)
- [.ai-debate/03_codex_response.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/.ai-debate/03_codex_response.md)
- [.ai-debate/04_claude_final_review.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/.ai-debate/04_claude_final_review.md)

### 1. Kết luận tranh luận Codex–Claude

Các điểm đã chốt:

- `full-image 0.6731` trên `hcm0031` là baseline thật và là mốc đúng để so.
- baseline local hiện hành không phải `raw sparse`, mà đã train từ `prepared dense source`:
  - `pipeline/work/hcm0031/gs_model/cfg_args` ghi `source_path=.../colmap/dense`
- baseline hiện hành chưa dùng `true depth supervision`:
  - `pipeline/work/hcm0031/03_train_3dgs.log` chỉ có `Depth Loss=0.0000000`
- repo đã có failed run thật của nhánh `B2`/`prepared`, với score rơi về khoảng `0.4027`:
  - `downloads/B2_done.ipynb`
  - `downloads/B2_done 2.ipynb`
  - `downloads/B2_done 2 safe.ipynb`
- dense stereo từng chạy xong ở một môi trường khác:
  - `colab_b2_results_drop/04_colmap_dense_summary.txt` ghi `depth_map_files=400`
  - `fused.ply` tồn tại và có kích thước `121495051` bytes
- các kết luận cũ hạ ưu tiên `depth-prior` và `antenna-focus` đến từ `HCM0421` Round 2 holdout, không phải bằng chứng trực tiếp cho `hcm0031` Round 1 public.

Kết luận tổng hợp:

- không còn hợp lý để coi `B2` là nhánh "chưa thử"
- cũng chưa đủ bằng chứng để kết luận `B2` chết hẳn về mặt ý tưởng
- tuning nhẹ gần như chắc chắn không đủ để tự tin nhắm `0.85`
- cần coi `0.85` là target, không phải kỳ vọng mặc định

### 2. Những giả định cũ bị bác bỏ hoặc cần kiểm chứng lại

Những giả định cũ đã bị bác bỏ:

- `baseline_ref` trong [trick/hcm0031/experiment_matrix.csv](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/trick/hcm0031/experiment_matrix.csv) không còn được hiểu là `raw baseline`; nó đang bị gắn nhãn sai so với artifact thật.
- không thể tiếp tục mô tả `B2` như bước "pilot cần chạy để biết có hiệu quả hay không", vì repo đã có negative result thật.
- không thể dùng kết quả `HCM0421` để đóng cửa dứt khoát các hướng `depth-prior` hoặc `tower/local focus` trên `hcm0031`.

Những giả định cần kiểm chứng lại:

- failed run `0.4027` là do bản thân `prepared/depth` hay do confound cấu hình huấn luyện
- train/render parity hiện tại có đúng hay không, vì local `gs_model` thiếu:
  - `pipeline_train_flags.json`
  - `chkpnt30000.pth`
  - `iteration_30000/point_cloud.ply`
- trần khả thi của dữ liệu này có đủ cao để theo đuổi `0.85` hay không
- tower có thật sự là bottleneck chính hay bbox crop đang kéo metric lên sai

### 3. Danh mục phương án

#### 3.1. Cải tiến pipeline hiện tại

- rebuild `gold baseline` đầy đủ artifact và parity
- cách ly failed run `B2` bằng control rõ ràng:
  - `prepared`, `LOW_VRAM_PROFILE=0`, `RESOLUTION=-1`
  - so với control âm tái hiện cấu hình low-VRAM cũ
- thử `prepared + true depths` chỉ sau khi confound cấu hình đã được khóa
- thêm `LPIPS-aware loss` hoặc score-oriented loss hợp lệ trên train GT
- refine nhẹ về pose / exposure / color chỉ sau khi baseline và checkpoint selection đã sạch

#### 3.2. Multi-run và search

- checkpoint sweep
- multi-seed để đo variance thật
- search hẹp trên:
  - save/checkpoint schedule
  - densify schedule
  - `sh_degree`
  - `densify_grad_threshold`
- mọi search phải báo song song:
  - checkpoint cuối
  - best checkpoint sau sweep

#### 3.3. Ensemble / two-stage

- global/background model + local/tower specialist
- render-space ensemble hợp lệ
- compositing chỉ được dựa vào:
  - geometry
  - train-time masks
  - uncertainty từ render
- không dùng GT test để chọn trọng số blend hay checkpoint thắng

#### 3.4. Thay model hoặc representation

- thử sớm các branch đã có sẵn trong repo:
  - `feature/mip-splatting`
  - `feature/gsplat-mcmc`
  - `compact/compact-gaussian`
  - `feature/depth-anything-v2`
- chỉ khi các branch này không mở thêm headroom mới cân nhắc đổi representation xa hơn

#### 3.5. Crazy ideas và trick hợp lệ

- geometry-assisted oracle warp/blend để đo trần
- sky/far-field masking hoặc sky prior cho skyline
- local tower model độ phân giải cao
- uncertainty-weighted ensemble giữa nhiều seed
- xin hoặc phục hồi dữ liệu nếu có thể mở rộng từ sparse gốc `388` frame

### 4. Ma trận Expected Gain / Cost / Risk / Evidence

| Hướng | Expected gain | Cost | Risk | Evidence hiện có |
|---|---|---:|---:|---|
| Rebuild gold baseline | Trung bình | Thấp | Thấp | artifact local hiện thiếu, parity chưa chắc đúng |
| B2 failure isolation | Cao | Trung bình | Thấp | đã có failed run `0.4027`, nhưng nguyên nhân chưa khóa |
| Checkpoint sweep | Trung bình | Thấp | Thấp | repo đã có script sweep, chưa có run matrix thật |
| Multi-seed | Trung bình | Trung bình | Thấp | chưa có seed variance cho `hcm0031` |
| Masked tower eval | Trung bình | Thấp | Thấp | bbox tower đang có thể lạc quan giả |
| Prepared + true depths | Không rõ | Trung bình-cao | Trung bình | baseline chưa dùng depth thật, nhưng prepared branch từng fail nặng |
| LPIPS-aware loss | Trung bình | Trung bình | Trung bình | hợp lệ, chưa thấy repo thử trực tiếp |
| `feature/mip-splatting` | Trung bình-cao | Trung bình | Trung bình | branch có sẵn, đánh đúng bệnh skyline/far-field |
| Ensemble / two-stage | Trung bình-cao | Cao | Cao | hợp lý nếu oracle và tower mask cho thấy headroom vùng |
| Đổi representation triệt để | Không rõ nhưng có thể lớn | Rất cao | Cao | chỉ nên mở khi oracle cho thấy 3DGS-family thiếu trần |

### 5. Oracle experiments và upper-bound diagnostics

Các oracle bắt buộc nên có trước khi tiêu tốn nhiều GPU:

1. `Geometry-assisted warp/blend oracle`
   - dùng nhiều neighbor views
   - có xử lý occlusion
   - phải sanity-check trên một tập train held-out nhỏ trước khi tin số trên test

2. `Blend oracle`
   - so nhiều candidate render hợp lệ
   - không dùng GT test để chọn trọng số thật khi đưa vào pipeline chính
   - chỉ dùng để biết headroom của ensemble/compositing

3. `Dense value oracle`
   - so công bằng:
     - baseline control
     - `prepared`
     - `prepared + depths`

Upper-bound diagnostics cần giữ:

- `diagnose_distance.csv` hiện có giá trị tham khảo, nhưng yếu vì script gốc đã bị xoá
- phải ưu tiên oracle có thể tái tạo được hơn là bám quá chặt vào một CSV mồ côi

### 6. Kế hoạch 24 giờ, 72 giờ và full experiment

#### 24 giờ

1. Khóa chuẩn đo lường:
   - mọi score phải ghi rõ `psnr_max=50.0`
2. Rebuild `gold baseline`
3. Chạy `B2 failure isolation`
4. Xác nhận artifact bắt buộc:
   - `pipeline_train_flags.json`
   - `chkpnt30000.pth`
   - `iteration_30000/point_cloud.ply`

#### 72 giờ

1. Checkpoint sweep đầy đủ
2. 3-seed baseline
3. Masked tower eval
4. Oracle ceiling
5. Quyết định:
   - tiếp tục 3DGS-family
   - hay mở sớm representation / 2-stage

#### Full experiment

1. Nếu `B2` được cứu bởi failure isolation:
   - thử `prepared + depths`
   - thử LPIPS-aware loss
   - thử refine nhẹ pose/exposure
2. Nếu oracle thấp hoặc `B2` vẫn đỏ:
   - mở `feature/mip-splatting`
   - cân nhắc `feature/gsplat-mcmc`
3. Nếu tower mask thật sự yếu:
   - mở local/tower specialist hoặc two-stage compositing

### 7. GO/STOP thresholds dựa trên full-image score

Mọi threshold trong phần này mặc định nói về `full-image score`, không dùng crop metric thay thế.

- `GO baseline parity`:
  - baseline rebuild tái hiện được score gần `0.6731`
- `GO B2 isolation`:
  - run `prepared` control phục hồi về gần baseline
- `STOP B2 branch`:
  - sau khi đã cô lập confound cấu hình, `prepared/current implementation` vẫn nằm quá xa baseline
- `GO tuning branch`:
  - có ít nhất một run vượt control đủ rõ, không phải chỉ dao động trong nhiễu
- `STOP tuning branch`:
  - sau một vòng ngắn có kiểm soát, không có run nào thắng control đủ rõ
- `GO representation branch`:
  - oracle cho thấy còn headroom, hoặc tuning branch bị khóa
- `STOP representation branch`:
  - branch mới không thắng control và không cải thiện đúng vùng lỗi mục tiêu

Ngưỡng số cụ thể phải được hiệu chỉnh sau khi có:

- variance từ multi-seed
- chất lượng của oracle

Không nên hard-code ngưỡng tuyệt đối kiểu `±0.005` hoặc `+0.01` trước khi biết nhiễu thực tế của pipeline hiện hành.

### 8. Điều kiện chuyển sang E/G

Chuyển sớm sang `E/G` khi xảy ra một trong các điều kiện:

- oracle ceiling thấp, cho thấy `0.85` khó với 3DGS-family hiện tại
- `B2 failure isolation` không cứu được branch `prepared`
- sau một vòng ngắn 3DGS-family có kiểm soát, không có run nào vượt baseline đủ rõ
- masked tower eval xác nhận tower là bottleneck riêng, cần specialist model

Diễn giải:

- `E` không còn là "xa xỉ cuối roadmap"
- `G` không còn phải chờ đến khi mọi tuning nhỏ đã cạn sạch
- cả hai nên được mở ngay khi dữ kiện cho thấy trần của đường hiện tại không đủ

### 9. Rủi ro leakage và kiểm soát tính hợp lệ

Các rủi ro cần kiểm soát:

- chọn checkpoint tốt nhất bằng cách lặp eval trên `test/images` GT thật
- chọn blend weight hoặc compositing rule theo GT test
- dùng crop metric để hợp thức hóa run không thắng `full-image`

Kiểm soát tối thiểu:

- báo song song:
  - checkpoint cuối
  - best checkpoint sau sweep
- không so `best-sweep ứng viên` với `checkpoint-cuối baseline`
- mọi rule blend/compositing phải log rõ nguồn tín hiệu:
  - geometry
  - train-time masks
  - uncertainty
- không đọc GT test để tối ưu trọng số
- crop metrics chỉ dùng để chẩn đoán, không thay `full-image score` làm tiêu chí chốt

## Trạng thái chiến lược sau tổng hợp

- `0.85` là target, không phải kỳ vọng mặc định
- chưa có bằng chứng nào đủ mạnh để tuyên bố chắc chắn một hướng cụ thể sẽ đạt `0.85`
- hướng đúng hiện tại là:
  - khóa baseline và đo lường
  - cách ly failed run `B2`
  - đo oracle ceiling
  - rồi quyết định tiếp tục 3DGS-family hay chuyển sang `E/G`
