# 05 Codex Synthesis

## 0. Phạm vi và nguyên tắc

File này tổng hợp lại các kết luận sau khi đã đọc và đối chiếu:

- `trao đổi.md`
- toàn bộ `trick/`
- `.ai-debate/01_codex_proposal.md`
- `.ai-debate/02_claude_review.md`
- `.ai-debate/03_codex_response.md`
- `.ai-debate/04_claude_final_review.md`

Nguyên tắc của bản synthesis này:

- giữ nguyên các dữ kiện thực nghiệm đã được xác minh
- không tuyên bố chắc chắn đạt `0.85`
- coi `0.85` là target cần được chứng minh bằng bằng chứng thực nghiệm và oracle
- dùng `full-image score` làm tiêu chí chốt cuối cùng
- coi crop/region metrics là tín hiệu chẩn đoán, không thay thế metric chốt

## 1. Kết luận tranh luận Codex–Claude

### 1.1. Những điểm hai bên đã hội tụ

Các điểm hiện đã đủ chắc để xem là nền tảng chung:

- baseline `hcm0031` là thật:
  - `full-image`: `PSNR 21.6938`, `SSIM 0.6819`, `LPIPS 0.1542`, `Score 0.6731`
  - `tower-crop`: `Score 0.7064`
  - `skyline-crop`: `Score 0.6384`
- baseline local hiện tại được train từ `prepared dense source`, không phải raw sparse thuần:
  - `pipeline/work/hcm0031/gs_model/cfg_args` ghi `source_path=.../colmap/dense`
- baseline local hiện tại chưa dùng `true depth supervision`:
  - `03_train_3dgs.log` chỉ có `Depth Loss=0.0000000`
- local artifact hiện không đủ sạch để tự verify trọn vẹn baseline:
  - thiếu `pipeline_train_flags.json`
  - thiếu `chkpnt30000.pth`
  - thiếu `point_cloud/iteration_30000/point_cloud.ply`
- repo đã có failed run thật của nhánh `B2/prepared`, rơi về khoảng `Score 0.4027`
- dense stereo từng chạy xong thật ở môi trường khác:
  - `colab_b2_results_drop/04_colmap_dense_summary.txt` ghi `depth_map_files=400`
  - `fused.ply` tồn tại
- các kết quả `HCM0421` về `depth-prior` / `antenna-focus` là evidence cross-scene, cross-round, cross-protocol, không được xem là bằng chứng trực tiếp cho `hcm0031`

### 1.2. Những điểm Codex ban đầu sai hoặc thiếu

- bỏ sót bằng chứng lớn trong:
  - `downloads/B2_done.ipynb`
  - `downloads/B2_done 2.ipynb`
  - `downloads/B2_done 2 safe.ipynb`
  - `colab_b2_results_drop/`
- vì vậy đã xếp `prepared/depth` như nhánh "chưa thử thật" trong khi repo đã có negative result
- đã cho trọng số hơi cao với một số kết luận cũ từ `HCM0421`

### 1.3. Những điểm Claude đi quá xa bằng chứng

- chưa đủ bằng chứng để kết luận bản chất của `B2` là sai; hiện mới đủ bằng chứng để kết luận:
  - nhánh này từng fail nặng
  - failed run đó chưa được cách ly nguyên nhân sạch
- chưa đủ bằng chứng để bác bỏ dứt khoát target `0.85`; điều cần làm là đo ceiling bằng oracle trước

### 1.4. Kết luận tổng hợp

Kết luận chiến lược tốt nhất hiện tại là:

- không tiếp tục tin rằng tuning nhẹ sẽ tự đưa score đến gần `0.85`
- không tiếp tục mô tả `B2` như một hướng chưa có dữ kiện
- ưu tiên hàng đầu là khóa lại baseline, parity đo lường và nguyên nhân failed run
- sau đó dùng oracle ceiling để quyết định:
  - tiếp tục 3DGS-family
  - hay chuyển sớm sang `E/G`, ensemble, hoặc representation khác

## 2. Những giả định cũ bị bác bỏ hoặc cần kiểm chứng lại

### 2.1. Giả định đã bị bác bỏ

- `baseline_ref` không được gọi là `raw baseline` nữa; baseline hiện hành là `prepared-no-depth`
- không thể tiếp tục coi `B2` là bước "pilot cần chạy để biết có hiệu quả không"
- không thể dùng `HCM0421 Round 2 holdout` để đóng cửa trực tiếp các hướng trên `hcm0031 Round 1 public`
- không thể coi checkpoint cuối mặc định là đại diện đúng nhất nếu chưa sweep checkpoint

### 2.2. Giả định cần kiểm chứng lại

- failed run `0.4027` là do:
  - bản thân `prepared/depth`
  - hay confound cấu hình như `LOW_VRAM_PROFILE`, `resolution`, densify schedule, parity train/render
- tower có thật sự là bottleneck lớn độc lập không, hay bbox crop đang lạc quan giả
- trần khả thi của scene này nằm ở đâu nếu hình học/interpolation được làm tốt hơn
- các branch representation hiện có trong repo có mở thêm headroom thật hay không

## 3. Danh mục phương án

### 3.1. Cải tiến pipeline hiện tại

- rebuild `gold baseline` với artifact đầy đủ
- khóa `Measurement Lock`:
  - `psnr_max=50.0`
  - audit mọi notebook/script in score
- chạy `B2 failure isolation`
- thử `prepared + true depths` chỉ sau khi failure isolation xong
- thử score-oriented loss:
  - LPIPS-aware loss
- refine nhẹ:
  - pose
  - exposure
  - color

### 3.2. Multi-run và search

- checkpoint sweep
- multi-seed
- search hẹp trên:
  - `sh_degree`
  - `densify_grad_threshold`
  - save/checkpoint schedule
  - densify schedule
- mọi so sánh phải báo song song:
  - checkpoint cuối
  - best checkpoint sau sweep

### 3.3. Ensemble / two-stage

- global/background model + local/tower model
- render-space ensemble giữa multiple seeds hoặc multiple branches
- compositing theo:
  - geometry
  - train-time mask
  - uncertainty từ render

### 3.4. Thay model hoặc representation

- thử sớm các branch đã có sẵn:
  - `feature/mip-splatting`
  - `feature/gsplat-mcmc`
  - `compact/compact-gaussian`
  - `feature/depth-anything-v2`
- chỉ sau khi các branch này không cho thêm headroom mới cân nhắc đổi representation xa hơn

### 3.5. Crazy ideas và trick hợp lệ

- geometry-assisted warp/blend oracle
- sky/far-field prior cho skyline
- local tower model độ phân giải cao
- uncertainty-weighted ensemble
- data-side recovery nếu có nguồn bổ sung từ sparse gốc `388` frame

## 4. Ma trận Expected Gain / Cost / Risk / Evidence

| Hướng | Expected gain | Cost | Risk | Evidence |
|---|---|---:|---:|---|
| Gold baseline rebuild | Trung bình | Thấp | Thấp | artifact local đang thiếu |
| Measurement lock | Trung bình | Thấp | Thấp | đã có bug `PSNR_MAX=30.0` trong notebook |
| B2 failure isolation | Cao | Trung bình | Thấp | đã có failed run `0.4027`, chưa khóa nguyên nhân |
| Checkpoint sweep | Trung bình | Thấp | Thấp | script có sẵn, chưa có matrix thật |
| Multi-seed | Trung bình | Trung bình | Thấp | chưa đo variance trên `hcm0031` |
| Masked tower eval | Trung bình | Thấp | Thấp | bbox tower có thể lạc quan |
| Prepared + true depths | Không rõ | Trung bình-cao | Trung bình | baseline chưa dùng depth thật, nhưng branch prepared từng fail |
| LPIPS-aware loss | Trung bình | Trung bình | Trung bình | hợp lệ, chưa thấy repo thử trực tiếp |
| `feature/mip-splatting` | Trung bình-cao | Trung bình | Trung bình | branch có sẵn, đánh vào skyline/far-field |
| `feature/gsplat-mcmc` / compact-gaussian | Không rõ | Trung bình-cao | Trung bình | branch có sẵn nhưng chưa có số liệu trên `hcm0031` |
| Ensemble / two-stage | Trung bình-cao | Cao | Cao | hợp lý nếu oracle và masked tower cho thấy headroom vùng |
| Data recovery từ sparse gốc | Có thể rất cao | Không rõ | Cao | có evidence thiếu `138/388` frame cục bộ |

## 5. Oracle experiments và upper-bound diagnostics

### 5.1. Geometry-assisted oracle

Mục tiêu:

- đo trần khả thi của dữ liệu nếu hình học/view synthesis được cải thiện tốt

Yêu cầu tối thiểu:

- nhiều neighbor views
- blend có xử lý occlusion
- sanity-check trên train held-out nhỏ trước khi tin số trên test

### 5.2. Blend oracle

Mục tiêu:

- đo headroom của ensemble/compositing

Yêu cầu:

- không dùng GT test để chọn trọng số của pipeline chính
- có thể dùng như diagnostic để biết đa mô hình có đáng đầu tư hay không

### 5.3. Dense value oracle

Mục tiêu:

- tách giá trị của:
  - baseline control
  - `prepared`
  - `prepared + depths`

### 5.4. Upper-bound diagnostics phụ trợ

- `diagnose_distance.csv` có giá trị tham khảo, nhưng yếu vì script gốc đã bị xóa
- artifact `01_run_colmap.log` cho biết sparse gốc từng có `388` ảnh, trong khi cục bộ chỉ có:
  - `200 train`
  - `50 test`
  - còn thiếu `138` frame không nằm ở train hay test cục bộ

## 6. Kế hoạch 24 giờ, 72 giờ và full experiment

### 6.1. Trong 24 giờ

1. Khóa đo lường:
   - chuẩn `psnr_max=50.0`
2. Rebuild `gold baseline`
3. Chạy `B2 failure isolation`
4. Xác nhận artifact baseline đầy đủ

### 6.2. Trong 72 giờ

1. Checkpoint sweep
2. 3-seed baseline
3. Masked tower eval
4. Oracle ceiling
5. Chọn nhánh tiếp theo:
   - tiếp tục 3DGS-family
   - hoặc mở representation / two-stage sớm

### 6.3. Full experiment

Nếu failure isolation cứu được `prepared`:

- thử `prepared + depths`
- thử LPIPS-aware loss
- thử refine nhẹ pose/exposure

Nếu oracle thấp hoặc `prepared` vẫn đỏ:

- mở `feature/mip-splatting`
- sau đó cân nhắc `feature/gsplat-mcmc`

Nếu masked tower xác nhận deficit riêng:

- mở local/tower specialist hoặc two-stage compositing

## 7. GO/STOP thresholds dựa trên full-image score

Mọi threshold trong file này mặc định là `full-image score`.

### 7.1. GO

- `GO baseline parity`:
  - baseline rebuild tái hiện được score gần baseline tham chiếu
- `GO B2 isolation`:
  - control `prepared` phục hồi rõ so với failed run `0.4027`
- `GO tuning branch`:
  - có run thắng control đủ rõ, không chỉ trong vùng nhiễu
- `GO representation branch`:
  - oracle cho thấy còn headroom hoặc tuning branch bị khóa

### 7.2. STOP

- `STOP B2 branch`:
  - sau khi đã cô lập confound chính, branch prepared/current implementation vẫn nằm quá xa baseline
- `STOP tuning branch`:
  - sau một vòng ngắn có kiểm soát, không có run nào thắng control đủ rõ
- `STOP representation branch`:
  - branch mới không thắng control và không cải thiện đúng vùng lỗi mục tiêu

### 7.3. Lưu ý về ngưỡng số

Không nên hard-code sớm các ngưỡng kiểu:

- `±0.005`
- `+0.01`

trước khi có:

- seed variance thật
- chất lượng oracle thật

Các ngưỡng số cụ thể phải được hiệu chỉnh sau khi đo được nhiễu thực tế.

## 8. Điều kiện chuyển sang E/G

Chuyển sang `E/G` sớm nếu xảy ra một trong các trường hợp:

- oracle ceiling thấp, cho thấy `0.85` khó với 3DGS-family hiện tại
- `B2 failure isolation` không cứu được branch `prepared`
- sau một vòng ngắn 3DGS-family có kiểm soát, không có run nào thắng baseline đủ rõ
- masked tower eval xác nhận tower là deficit riêng, cần specialist model

Diễn giải:

- `E` không còn là nhánh chỉ mở cuối cùng
- `G` không còn phải chờ cạn mọi tuning nhỏ
- cả hai được phép mở sớm khi evidence cho thấy trần của đường hiện tại không đủ

## 9. Rủi ro leakage và kiểm soát tính hợp lệ

Các rủi ro chính:

- chọn checkpoint tốt nhất bằng cách lặp eval trên `test/images` GT thật
- chọn blend/compositing weight theo GT test
- dùng crop metric để hợp thức hóa run không thắng `full-image`

Kiểm soát tối thiểu:

- luôn báo song song:
  - checkpoint cuối
  - best checkpoint sau sweep
- không so `best-sweep ứng viên` với `checkpoint-cuối baseline`
- mọi blend/compositing rule phải log rõ nguồn tín hiệu:
  - geometry
  - train-time mask
  - uncertainty
- không đọc GT test để tối ưu trọng số
- crop metrics chỉ dùng để chẩn đoán, không thay thế `full-image score`

## 10. Trạng thái cuối cùng của chiến lược

- `0.85` là target, không phải kỳ vọng mặc định
- hiện chưa có bằng chứng nào đủ mạnh để tuyên bố chắc chắn một hướng cụ thể sẽ đạt `0.85`
- bằng chứng hiện có chỉ cho phép chốt:
  - phải khóa baseline và đo lường trước
  - phải cách ly failed run `B2`
  - phải đo oracle ceiling
  - rồi mới quyết định tiếp tục 3DGS-family hay chuyển sang `E/G`, ensemble, hoặc representation khác

## 11. Bằng chứng cần có để chứng minh từng hướng đủ khả năng nhắm `0.85`

### 11.1. Pipeline hiện tại / 3DGS-family

Cần có:

- baseline sạch, reproducible
- failure isolation cứu được prepared branch
- oracle ceiling đủ cao
- ít nhất một run 3DGS-family thắng control rõ rệt sau khi đã sweep checkpoint và tính variance

### 11.2. Multi-run và search

Cần có:

- seed variance đủ thấp để gain không bị chìm trong nhiễu
- checkpoint sweep cho thấy có headroom thật
- ít nhất một config search thắng control rõ rệt, không phải chỉ do best-of-many bias

### 11.3. Ensemble / two-stage

Cần có:

- oracle blend cho thấy headroom thật
- masked tower hoặc skyline diagnostics chứng minh deficit theo vùng
- compositing rule hợp lệ, không dùng GT test

### 11.4. Representation mới

Cần có:

- branch representation mới thắng control trên `full-image score`
- tốt hơn nữa nếu thắng đúng vùng lỗi mục tiêu:
  - skyline
  - far-field
  - thin structure

### 11.5. Crazy ideas

Cần có:

- ít nhất một oracle hoặc diagnostic cho thấy idea đó không chỉ là cảm giác
- bằng chứng rằng gain tiềm năng đủ lớn để đáng đổi lấy chi phí và rủi ro
