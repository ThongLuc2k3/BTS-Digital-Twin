# 03 Codex Response

## Executive Summary

Sau khi kiểm chứng `01_codex_proposal.md`, `02_claude_review.md`, code, dữ liệu hiện có, notebook lưu kết quả, và git history, kết luận chính là:

- Claude đúng ở điểm quan trọng nhất mà bản `01` đã bỏ sót: repo này đã có bằng chứng `B2`/`prepared` branch từng chạy thật và rơi về khoảng `Score 0.4027`, không còn là trạng thái "chưa biết".
- Claude cũng đúng khi chỉ ra rằng các kết luận hạ ưu tiên `depth-prior` và `antenna-focus` dựa trên `HCM0421` là ngoại suy từ `Round 2 holdout`, không thể coi là bằng chứng trực tiếp cho `hcm0031`.
- Tuy nhiên Claude có xu hướng đi xa hơn bằng chứng ở vài chỗ: `B2` chưa thể kết luận là thất bại bản chất, vì còn một confound cấu hình rất nặng; và target `0.85` chưa thể bác bỏ hoàn toàn nếu chưa có oracle ceiling.
- Chiến lược tổng hợp tốt nhất không phải bảo vệ proposal cũ, cũng không phải nhảy ngay sang representation mới. Bước đúng là:
  1. chốt lại baseline vàng và parity đo lường,
  2. cách ly confound làm sập `B2`,
  3. chạy oracle ceiling,
  4. rồi mới quyết định tiếp tục 3DGS-family hay chuyển sang 2-stage / ensemble / representation khác.

Kết luận cứng:

- `0.85` gần như chắc chắn **không nên giả định đạt được bằng tuning nhẹ**.
- `0.85` có thể vẫn còn cửa bằng pipeline mạnh hơn, nhưng cần oracle để xác nhận trước.
- Nếu oracle không lên gần `0.80+`, thì phải coi `2-stage model`, `ensemble hợp lệ`, hoặc `representation mới` là hướng bắt buộc, không còn là Tier 3 xa xỉ.

## 1. Adjudication Table

| Nhận xét của Claude | Verdict | Bằng chứng | Hành động tương ứng |
|---|---|---|---|
| `B2` đã được thử thật trong repo và fail nặng, không còn là trạng thái "chưa biết" | `ACCEPT` | `downloads/B2_done.ipynb`, `downloads/B2_done 2 safe.ipynb`, `downloads/B2_done 2.ipynb` đều lưu `PSNR mean : 10.5509`, `Score mean: 0.4027`; `colab_b2_results_drop/04_colmap_dense_summary.txt` xác nhận dense stereo từng chạy xong với `depth_map_files=400`, `fused_ply_bytes=121495051` | Sửa proposal cũ: không xếp `prepared/depth` như một ý tưởng chưa thử. Từ nay coi đây là nhánh đã có negative result cần root-cause analysis |
| Có confound cấu hình cụ thể cho vụ sập điểm: `LOW_VRAM_PROFILE` / giảm resolution / tắt densification | `PARTIAL` | Code trong [pipeline/scripts/03_train_3dgs.sh](../pipeline/scripts/03_train_3dgs.sh) đúng là tự ép `RESOLUTION=4`, `DENSIFY_UNTIL_ITER=0`, `DENSIFY_FROM_ITER=0`, `DENSIFICATION_INTERVAL=0`, `PERCENT_DENSE=0.0` khi `LOW_VRAM_PROFILE=1`; nhưng notebook fail được lưu trong repo hiện tại không show trực tiếp dòng output `LOW_VRAM_PROFILE=1` trong file text mà ta grep được. `B2_done 2.ipynb` và `B2_done 2 safe.ipynb` có cell override `LOW_VRAM_PROFILE=0`, nhưng kết quả cuối vẫn là `0.4027` | Không kết luận thủ phạm duy nhất là low-VRAM. Phải chạy thí nghiệm cách ly có kiểm soát: `prepared` với `LOW_VRAM_PROFILE=0`, `RESOLUTION=-1`, và schedule densify tường minh |
| Các kết luận hạ ưu tiên `depth-prior` / `antenna-focus` từ `HCM0421` không đủ sức áp sang `hcm0031` | `ACCEPT` | `git show 0f85a05:pipeline/common/scenes.py` xác nhận `HCM0421` thuộc Round 2 không có GT thật; commit `8e88bc0`, `9383e23` là kết quả trên `HCM0421` holdout; đây là khác scene, khác dataset regime, khác protocol | Hạ các claim đó từ `fact` xuống `cross-scene evidence`. Không dùng chúng để bác bỏ depth/tower tricks trên `hcm0031` nếu chưa có test trực tiếp |
| Baseline `0.6731` không phải raw baseline thuần mà đã train từ `prepared dense source` | `ACCEPT` | `pipeline/work/hcm0031/gs_model/cfg_args` có `source_path='/kaggle/working/pipeline/work/hcm0031/colmap/dense'`; `03_train_3dgs.log` có `Depth Loss=0.0000000` | Sửa control table. Đổi nhãn baseline trong `experiment_matrix.csv` và mọi tài liệu: baseline hiện hành là `prepared-no-depth`, không phải `raw` |
| Artifact cục bộ không đủ để tự verify lại baseline `0.6731` | `ACCEPT` | `pipeline/work/hcm0031/gs_model` hiện thiếu `pipeline_train_flags.json`, `chkpnt30000.pth`, `point_cloud/iteration_30000`; render baseline log cũ lại tham chiếu `iteration_30000` | Ưu tiên số 1 là rebuild một `gold baseline package` đầy đủ artifact trước khi kết luận về bất kỳ delta nhỏ nào |
| `P0 PASS` chỉ chứng minh dữ liệu cục bộ sạch, không chứng minh bộ dữ liệu phát hành là đầy đủ so với sparse gốc 388 ảnh | `ACCEPT` | `pipeline/work/hcm0031/01_run_colmap.log` ghi sparse gốc có `388 ảnh`; đối chiếu tên file cho ra `188 missing`, trong đó `50` nằm ở test, `138` không nằm ở train hay test cục bộ | Thêm một fact mới vào chiến lược: có ceiling data-side do missing frames. Mọi mục tiêu quá cao phải được đánh giá dưới ràng buộc này |
| Tương quan distance diagnostic đi ngược giả thuyết nền của `B2`, nên lý do ưu tiên `B2` yếu hơn tài liệu nói | `PARTIAL` | `pipeline/work/hcm0031/diagnose_distance.csv` cho corr `dist vs psnr = +0.1951`, `dist vs ssim = +0.3870`, `dist vs lpips = -0.3470`; nhưng script tạo CSV đã bị xóa (`git log --all -- '*09_diagnose_distance*'`) nên không audit lại được cách đo, và tương quan này còn yếu | Giữ cảnh báo, nhưng không coi đây là bằng chứng bác bỏ `B2`. Chỉ dùng nó để nâng ưu tiên cho oracle và phân tích theo view cluster, không dùng làm quyết định đơn lẻ |
| Repo có bug đo lường thật do `PSNR_MAX=30.0` trong nhánh re-eval notebook | `ACCEPT` | `downloads/B2_done*.ipynb` đều chứa `PSNR_MAX = 30.0`; cùng notebook lại in `Score mean: 0.4027` ở eval chính và khoảng `0.444x` ở re-eval summary | Chuẩn hóa toàn bộ score reporting về `psnr_max=50.0`; mọi score khác chuẩn phải ghi rõ và không dùng để so trực tiếp |
| Codex cũ đã anchor quá lâu vào pipeline tuần tự `B2 -> C/F -> A` | `PARTIAL` | Nhận định này đúng về tài liệu repo gốc. Với `01_codex_proposal.md`, tôi có mở `global/local`, `ensemble`, `representation swap`, nhưng xếp quá muộn vì bỏ sót negative result của `B2` | Cập nhật ladder: không bắt mọi nhánh chờ tuần tự hoàn toàn. Sau khi baseline và confound được khóa, có thể mở song song oracle + 1 nhánh representation |
| Mục tiêu `0.85` cần bị nghi ngờ nghiêm túc, khó đạt bằng 3DGS tuning thuần | `PARTIAL` | Phân tích công thức score là hợp lý; baseline `0.6731` lên `0.85` là một bước rất lớn; nhưng hiện chưa có oracle ceiling để biến nhận định này thành kết luận cứng | Không hứa `0.85` bằng tuning. Dùng oracle ceiling làm gate: nếu oracle không gần `0.80+`, chuyển sớm sang 2-stage / ensemble / representation mới |
| Cần thử sớm nhánh representation có sẵn như `feature/mip-splatting` | `ACCEPT` | `git branch -a` xác nhận có `remotes/origin/feature/mip-splatting`, `feature/gsplat-mcmc`, `compact/compact-gaussian`, `feature/depth-anything-v2` | Đưa representation branch lên sớm hơn proposal cũ, nhưng chỉ sau khi khóa baseline parity và chạy oracle |

## 2. Where The Old Proposal Was Weak

Các điểm yếu thật của `01_codex_proposal.md`:

1. Tôi đã dừng ở `pipeline/work/hcm0031/` quá sớm và bỏ sót bằng chứng quan trọng trong:
   - `downloads/B2_done.ipynb`
   - `downloads/B2_done 2 safe.ipynb`
   - `downloads/B2_done 2.ipynb`
   - `colab_b2_results_drop/`
2. Vì bỏ sót các file này, tôi xếp `prepared/depth` như một hướng "chưa thử thật", trong khi repo đã có failed run.
3. Tôi đã cho trọng số hơi cao với các kết quả `HCM0421` mà chưa tách bạch Round 1 vs Round 2.
4. Tôi chưa đưa yếu tố `138/388` missing frames thành một bottleneck data-side rõ ràng trong chiến lược.

Các điểm của proposal cũ vẫn giữ được:

1. baseline hiện hành thực sự là `prepared-no-depth`
2. artifact hygiene đang kém
3. render parity có rủi ro thật
4. masked tower eval vẫn là bước phân loại ROI tốt
5. cần oracle để đo ceiling trước khi đốt thêm GPU

## 3. Integrated Strategy

### 3.1 Facts we should now anchor on

- Baseline thật đang có là `0.6731`, train từ `prepared dense source`, không dùng depth loss thật.
- Repo đã có failed run của nhánh `B2`/`prepared` với score khoảng `0.4027`.
- Dense stereo từng chạy xong ở ít nhất một môi trường khác, tạo `400 depth maps` và `fused.ply`.
- Local artifact hiện không đủ để tự verify lại baseline 30k.
- Có mismatch measurement risk do `PSNR_MAX=30` ở một số notebook re-eval.
- Có evidence data-side rằng benchmark phát hành thiếu `138/388` frame gốc đã có pose trong sparse reconstruction.

### 3.2 What this means

- Không còn hợp lý để tiếp tục văn phong "đợi pilot B2".
- Nhưng cũng chưa hợp lý để kết luận "`B2` vô dụng" vì run failed hiện có vẫn bị confound bởi cấu hình huấn luyện chưa được cách ly sạch.
- Tuning đơn thuần không còn là ứng viên chính để lên `0.85`.
- Muốn tiếp tục 3DGS-family một cách nghiêm túc, phải trả lời hai câu hỏi trước:
  1. nhánh `prepared/depth` fail vì bản thân ý tưởng, hay vì config bug / low-VRAM / measurement mismatch?
  2. ceiling khả dĩ của dữ liệu này là bao nhiêu nếu hình học và interpolation gần tối ưu?

### 3.3 Revised view on target `0.85`

Phán quyết tích hợp:

- `0.85` **không thể được coi là mục tiêu khả dĩ bằng tuning nhẹ**. Điểm này tôi đồng ý với tinh thần của Claude.
- `0.85` **chưa thể bác bỏ hoàn toàn** vì chưa có oracle ceiling.
- Tuy nhiên, nếu oracle geometry-assisted / blend oracle không lên được vùng `0.80+`, thì từ đó trở đi phải coi:
  - `2-stage model`
  - `ensemble hợp lệ`
  - `representation mới`
  
  là hướng bắt buộc, không còn là "crazy optional branch".

## 4. Experiment Ladder

Nguyên tắc:

- Mỗi bậc phải có ngân sách rõ, output rõ, và ngưỡng dừng rõ.
- Các bậc 1 và 2 được phép chạy song song vì đều rẻ và mang tính quyết định.

### Ladder 0: Measurement Lock

Mục tiêu:

- khóa chuẩn score/reporting
- loại bỏ nhầm lẫn `psnr_max`

Việc làm:

- chuẩn hóa mọi eval về `psnr_max=50.0`
- audit mọi notebook/script đang in score theo chuẩn khác

Ngân sách:

- `0 GPU run`
- `0.5 ngày`

GO:

- tất cả score mới đều báo rõ `psnr_max=50.0`

STOP:

- không so trực tiếp bất kỳ score nào dùng `psnr_max != 50.0`

### Ladder 1: Gold Baseline Rebuild

Mục tiêu:

- tái tạo baseline chuẩn với artifact đầy đủ

Việc làm:

- train lại baseline control
- lưu đủ:
  - `pipeline_train_flags.json`
  - `chkpnt30000.pth`
  - `point_cloud/iteration_30000/point_cloud.ply`

Ngân sách:

- `1 full GPU run`

Đo:

- `full-image score`
- artifact completeness

GO:

- score nằm trong `±0.005` quanh `0.6731`

STOP:

- nếu không tái hiện được baseline, dừng mọi trick mới và sửa pipeline parity trước

### Ladder 2: B2 Failure Isolation

Mục tiêu:

- xác định failed run `0.4027` là do ý tưởng hay do config

Việc làm:

- chạy `prepared` control không depth
- ép tường minh:
  - `LOW_VRAM_PROFILE=0`
  - `RESOLUTION=-1`
  - densify schedule không bị auto tắt
- giữ các biến khác sát baseline nhất có thể

Ngân sách:

- `1 full GPU run`

Đo:

- `full-image score`
- `skyline-crop score`

GO:

- nếu score quay lại gần baseline, xác nhận failed run cũ bị config confound

STOP:

- nếu vẫn rơi vào vùng `<=0.50`, coi nhánh `prepared current implementation` là đỏ và không đầu tư thêm trước khi có root-cause sâu hơn

### Ladder 3: Checkpoint + Seed Hygiene

Mục tiêu:

- loại bỏ trần giả do checkpoint / seed

Việc làm:

- checkpoint sweep
- 3 seeds trên baseline control

Ngân sách:

- `1 render/eval sweep`
- `+2 full GPU runs`

Đo:

- best checkpoint score
- seed mean/std

GO:

- nếu best checkpoint thêm `>=0.01`, checkpoint sweep thành bắt buộc
- nếu seed std `>0.005`, mọi so sánh một-run bị hạ độ tin cậy

STOP:

- nếu gain checkpoint và gain seed đều nhỏ, không kỳ vọng tuning đơn thuần cứu được lớn

### Ladder 4: Oracle Ceiling

Mục tiêu:

- biết `0.85` còn trong vùng có thể đạt không

Việc làm:

- geometry-assisted warp/blend oracle
- optional render-space blend oracle trên các candidate hợp lệ

Ngân sách:

- `0-0.5 GPU run` tùy implementation

Đo:

- oracle `full-image score`

GO:

- nếu oracle `>=0.80`, còn đáng đầu tư mạnh cho 3DGS-family refinements

STOP:

- nếu oracle `<0.75`, không nên kỳ vọng `0.85` bằng tuning; mở sớm representation / 2-stage / ensemble

### Ladder 5: Region Diagnosis

Mục tiêu:

- xác định tower có còn là bottleneck độc lập không

Việc làm:

- bootstrap + manual mask
- masked tower eval

Ngân sách:

- `0 GPU run` hoặc rất thấp

Đo:

- masked tower score
- delta vs bbox tower score

GO:

- nếu masked tower thấp hơn bbox tower nhiều, mở local/tower branch

STOP:

- nếu masked tower không tệ đáng kể, không đầu tư sớm vào tower specialist

### Ladder 6A: 3DGS-Family Continuation

Điều kiện vào:

- Ladder 2 xác nhận failed run cũ chủ yếu do config
- Ladder 4 cho thấy ceiling đủ cao

Việc làm:

- true `prepared + depths`
- LPIPS-aware loss
- camera/exposure refinement nhẹ

Ngân sách:

- `2-4 GPU runs`

Ngưỡng dừng:

- nếu sau `3` run chất lượng không vượt best control ít nhất `+0.01`, dừng nhánh này

### Ladder 6B: Representation / Structured Branch

Điều kiện vào:

- Ladder 4 cho thấy 3DGS tuning khó tới trần mong muốn
hoặc
- Ladder 2 vẫn đỏ

Việc làm:

- thử `feature/mip-splatting` trước
- sau đó cân nhắc `feature/gsplat-mcmc` / `compact-gaussian`

Ngân sách:

- `1-3 GPU runs`

Ngưỡng dừng:

- nếu branch đầu tiên không thắng control trên `skyline/full-image`, chỉ mở tiếp branch khác nếu có lý do kỹ thuật rõ

### Ladder 7: Two-Stage / Ensemble

Điều kiện vào:

- masked tower thật sự yếu
hoặc
- oracle gợi ý blend/region specialization còn headroom

Việc làm:

- global/background model + local/tower specialist
- compositing hợp lệ, không dùng GT test
- multi-seed ensemble nếu có diversity thật

Ngân sách:

- `3-6 GPU runs`

Ngưỡng dừng:

- nếu complexity tăng nhưng score chỉ tăng `<0.01`, dừng

## 5. Practical Recommendation

Backlog thực tế, theo thứ tự:

1. `Measurement lock`
2. `Gold baseline rebuild`
3. `B2 failure isolation`
4. `Checkpoint sweep + 3 seeds`
5. `Oracle ceiling`
6. `Masked tower eval`
7. Chọn một trong hai nhánh:
   - `3DGS-family continuation`
   - `representation branch`
8. Chỉ sau đó mới mở `2-stage / ensemble`

## 6. Final Position

Tôi chấp nhận phần lớn phản biện quan trọng của Claude. Đặc biệt:

- bản `01` đã thiếu một mảng bằng chứng repo rất quan trọng
- không được tiếp tục nói về `B2` như một ý tưởng chưa thử
- không được dùng `HCM0421 Round 2 holdout` như bằng chứng trực tiếp để đóng cửa các hướng trên `hcm0031`

Nhưng tôi không đi đến kết luận cực đoan rằng:

- `B2` đã chết hẳn
hoặc
- `0.85` chắc chắn bất khả thi

Hai kết luận đó đều vượt quá bằng chứng hiện có.

Kết luận đúng hơn là:

- tuning nhẹ gần như chắc chắn không đủ cho `0.85`
- failed run `0.4027` buộc phải được cách ly nguyên nhân ngay
- oracle ceiling là gate quyết định cho toàn bộ roadmap
- nếu oracle thấp, phải chuyển sớm sang representation / 2-stage / ensemble
- nếu oracle cao và failed run được giải thích bằng config, vẫn còn lý do hợp lý để cứu nhánh 3DGS-family thêm một vòng ngắn, nhưng không quá 3-4 run trước khi stop
