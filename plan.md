# Kế hoạch duy nhất - BTS Digital Twin Round 1

## 1. Mục đích của file này

File này là bản chốt duy nhất để ra quyết định kỹ thuật cho giai đoạn còn lại của vòng 1.

Nó thay thế cách theo dõi bằng nhiều ghi chú rời rạc theo từng nhánh hoặc từng notebook.
Mục tiêu là:

- dừng vòng lặp "thử thêm một nhánh nữa"
- chốt một generator chính
- chốt một chuẩn benchmark chung
- chốt một lộ trình ngắn, thực dụng, làm được với thời gian và GPU hiện có

## 2. Bối cảnh đã xác minh

### 2.1 Bài toán và cách chấm

- Bài toán: từ ảnh drone train của mỗi scene, render ảnh RGB tại các pose trong `test_poses.csv`.
- Private round 1 có 8 scene cần nộp bài.
- Public set có 5 scene có ảnh ground-truth để tự chấm.
- Công thức chấm:

```text
Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)
```

- `PSNR_max` không được BTC công bố trong đề bài. Repo hiện đang dùng quy ước nội bộ `50` vì đã có tài liệu giải ngược từ một lần chấm thật. Giá trị này phù hợp để tự đánh giá nội bộ, nhưng vẫn phải coi là quy ước nội bộ, không phải tài liệu chính thức của BTC.

### 2.2 Dữ liệu

- Sparse COLMAP có sẵn hợp lệ cho 13/13 scene.
- Thiết kế pipeline hiện tại đúng theo hướng dùng thẳng sparse có sẵn, không ưu tiên tự chạy lại COLMAP.
- Độ phân giải đang thấy trong dữ liệu là `1320x989` và đồng nhất trên các scene đã kiểm.

### 2.3 Mốc baseline

- Mốc điểm đã chấm thật trên private bằng vanilla 3DGS: `58.67320`.
- Một mốc public đã được ghi nhận rõ trong notebook/tài liệu:
  - `hcm0031`: PSNR `21.689`, SSIM `0.6823`, LPIPS `0.1535`.

Kết luận quan trọng:

- mục tiêu thực sự không phải "ảnh đẹp hơn bằng mắt"
- mục tiêu là tăng điểm trung bình trên 8 scene private

## 3. Cấu trúc thư mục `Kết quả` đã được chuẩn hoá

Thư mục [Kết quả](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/Kết%20quả) đã được sắp xếp lại để dễ đọc và dễ cập nhật:

```text
Kết quả/
├── 01_main/
│   └── run_01_public_hcm0031/
├── 02_mip/
│   └── notebooks/
├── 03_depth/
├── 04_gsplat_mcmc/
│   └── run_01_public_hcm0031/
├── 05_compact/
└── 99_chua_phan_loai/
    ├── run_01_scene_50_views/
    ├── run_02_scene_60_views/
    ├── run_03_scene_50_views/
    └── run_04_scene_50_views/
```

### 3.1 Ý nghĩa từng khu

- `01_main`: kết quả đã gắn chắc với baseline `main`
- `02_mip`: notebook liên quan hướng Mip-Splatting, hiện mới có notebook, chưa có kết quả CSV đã gom chuẩn
- `03_depth`: notebook liên quan Depth Anything / depth prior
- `04_gsplat_mcmc`: kết quả đã gắn chắc với nhánh `feature/gsplat-mcmc`
- `05_compact`: notebook của hướng compact Gaussian
- `99_chua_phan_loai`: các lần chạy cũ chưa xác định được chắc chắn là thuộc nhánh nào

### 3.2 Những gì đọc được từ `eval_metrics.csv`

Các mốc hiện có:

- `01_main/run_01_public_hcm0031`
  - `n=50`
  - PSNR `21.687186`
  - SSIM `0.682331`
  - LPIPS `0.153909`
  - Score `0.673259`

- `04_gsplat_mcmc/run_01_public_hcm0031`
  - `n=50`
  - PSNR `21.694833`
  - SSIM `0.681849`
  - LPIPS `0.153849`
  - Score `0.673184`

- `99_chua_phan_loai/run_02_scene_60_views`
  - `n=60`
  - PSNR `20.369361`
  - SSIM `0.678227`
  - LPIPS `0.148724`
  - Score `0.666195`

- `99_chua_phan_loai/run_01_scene_50_views`
  - `n=50`
  - PSNR `21.682913`
  - SSIM `0.681791`
  - LPIPS `0.154166`
  - Score `0.759700`

- `99_chua_phan_loai/run_03_scene_50_views`
  - `n=50`
  - PSNR `21.684043`
  - SSIM `0.681875`
  - LPIPS `0.154107`
  - Score `0.759760`

- `99_chua_phan_loai/run_04_scene_50_views`
  - `n=50`
  - PSNR `21.688899`
  - SSIM `0.682256`
  - LPIPS `0.153506`
  - Score: không có trong file

### 3.3 Cách hiểu đúng các mốc trên

- Hai mốc đáng tin nhất để ra quyết định hiện tại là:
  - `01_main/run_01_public_hcm0031`
  - `04_gsplat_mcmc/run_01_public_hcm0031`

- Các run trong `99_chua_phan_loai` vẫn có giá trị tham khảo về PSNR/SSIM/LPIPS, nhưng chưa nên dùng để chốt nhánh vì:
  - chưa gắn chắc với nhánh/model
  - vài file score nhiều khả năng được tính bằng chuẩn `psnr_max` cũ, nên không so sánh trực tiếp với score mới

Kết luận từ số liệu hiện có:

- `gsplat_mcmc` hiện chưa cho thấy lợi thế rõ rệt so với `main` trên mốc `hcm0031`
- ít nhất với dữ liệu đang lưu trong `Kết quả`, chưa có bằng chứng rằng `gsplat_mcmc` vượt hẳn đường Inria 3DGS
- hiện cũng chưa có benchmark CSV sạch, gắn nhãn chắc chắn cho hướng `mip` / `depth-anything-v2` bên trong `Kết quả`; vì vậy các kết luận về hướng này hiện vẫn dựa nhiều vào cấu trúc code, commit, và mô tả kỹ thuật hơn là bằng chứng benchmark đã lưu trữ chuẩn hoá

### 3.4 Mức độ tin cậy của kết luận hiện tại

Đây là điểm quan trọng để tránh tự tin quá mức.

- Kết luận về `main`: độ tin cậy cao
  - vì đã có baseline thật, có file kết quả rõ ràng

- Kết luận về `feature/gsplat-mcmc`: độ tin cậy trung bình
  - vì đã có code chạy được và có một mốc kết quả gắn nhãn tương đối rõ
  - nhưng chưa có ưu thế rõ trên số liệu đang lưu

- Kết luận về `feature/mip-splatting` và `feature/depth-anything-v2`: độ tin cậy trung bình thấp
  - vì code và commit cho thấy đây là hướng hợp lý nhất để đi tiếp
  - nhưng thư mục `Kết quả` hiện chưa có benchmark CSV sạch, gắn nhãn chắc chắn để chứng minh mức tăng điểm của hướng này

Vì vậy, lựa chọn `feature/depth-anything-v2` ở thời điểm hiện tại là:

- **lựa chọn thực dụng nhất theo cấu trúc code hiện có**
- **chưa phải lựa chọn đã được xác nhận bằng benchmark lưu trữ sạch trong `Kết quả`**

## 4. Bản đồ các nhánh hiện có

### 4.1 `main`

Vai trò:

- baseline ổn định để đối chiếu
- có pipeline COLMAP -> train 3DGS repo Inria -> render -> eval -> package

Kết luận:

- giữ lại để đối chiếu
- không phải hướng để đi tiếp

### 4.2 `feature/mip-splatting`

Bản chất:

- vẫn là họ 3DGS repo Inria
- thêm antialiasing
- thêm sanity-check cho tính nhất quán train/render
- có code liên quan antenna-focus và depth prior

Điểm mạnh:

- là hướng nâng cấp trưởng thành nhất trong họ Inria 3DGS
- có script train/render/eval hoàn chỉnh
- đã có sửa bug lệch `antialiasing` giữa train và render

Rủi ro:

- tài liệu trong repo có chỗ ghi "đã tích hợp xong", nhưng phải tin vào kết quả chạy thật, không tin vào ghi chú
- antenna-focus mới ở mức có code, chưa có bằng chứng điểm số

Kết luận:

- đây là nền tảng đúng nhất để tiếp tục nếu vẫn đi theo họ Inria 3DGS

### 4.3 `feature/depth-anything-v2`

Bản chất:

- là nhánh tiến hoá trực tiếp từ hướng `feature/mip-splatting`
- có script sinh depth prior 16-bit đúng cho `make_depth_scale.py`
- commit mới nhất nhấn mạnh việc sửa mismatch antialiasing train/render

Điểm mạnh:

- giải quyết đúng một nút thắt có thể gặp trong private: pose test xa train, floaters, vùng khuyết
- không đổi họ mô hình, compute và pipeline vẫn trong tầm kiểm soát

Điểm yếu:

- depth prior từng gặp OOM
- chưa có bằng chứng điểm số sạch trên nhiều scene public

Kết luận:

- đây là nhánh **nên ưu tiên benchmark trước tiên**
- hiện tại vẫn có thể coi là **ứng viên generator chính tốt nhất theo mặt kỹ thuật**
- nhưng chỉ được xem là **generator chính đã chốt hoàn toàn** sau khi có benchmark sạch trên bộ scene dev

### 4.4 `feature/gsplat-mcmc`

Bản chất:

- không còn là trainer Inria mặc định
- đổi sang `gsplat` + chiến lược MCMC densification
- có render riêng cho checkpoint gsplat
- có fine-tune LPIPS riêng cho gsplat

Điểm mạnh:

- là hướng R&D nghiêm túc nhất nếu muốn đổi trainer để tránh OOM và kiểm soát budget Gaussian
- đã có train script, render script, fine-tune script riêng

Điểm yếu:

- độ phức tạp vận hành cao hơn
- không cùng format checkpoint với Inria
- cần thêm giờ test và benchmarking riêng
- dữ liệu hiện có trong `Kết quả` chưa chứng minh nó thắng rõ nhánh Inria đã fix bug

Kết luận:

- giữ làm nhánh R&D dự phòng
- không dùng làm trục chính ở thời điểm này
- với số liệu đang có trong `Kết quả`, chưa có lý do để nâng nó lên ngang hàng với hướng Inria đã sửa bug

### 4.5 `compact/compact-gaussian`

Bản chất:

- thêm cơ chế pruning/mask để nén Gaussian
- có bảo vệ vùng chi tiết qua detail-region/box

Điểm mạnh:

- ý tưởng rõ: cắt bớt Gaussian ở nền để dành ngân sách cho vùng BTS

Điểm yếu:

- đổi code khá sâu
- chưa có bằng chứng GPU thật
- cần thêm bước chuẩn bị detail region
- trong bối cảnh còn ít thời gian, đây là hướng dễ phát sinh thêm trạng thái cần debug

Kết luận:

- tạm bỏ khỏi roadmap chính
- chỉ giữ lại để tham khảo ý tưởng

## 5. Chuẩn đánh giá model từ giờ trở đi

Đây là quy tắc bắt buộc. Nếu không theo quy tắc này, mọi kết quả đều không đủ giá trị để ra quyết định.

### 5.1 Đơn vị so sánh

- đơn vị chính: điểm trung bình trên nhiều scene public
- không ra quyết định từ một scene duy nhất
- không ra quyết định từ ảnh đẹp bằng mắt

### 5.2 Bộ scene dev cố định

Cần chọn 3 scene public có tính chất khác nhau:

1. `hcm0031` - đã có mốc điểm, dùng làm scene chuẩn
2. một scene public có cấu trúc BTS phức tạp hơn
3. một scene public có artifact tệ nhất theo metric hoặc quan sát

Từ giờ, mọi A/B test tốn GPU phải chạy trên cùng 3 scene này.

### 5.3 Cấu hình benchmark tối thiểu

Mỗi hướng generator phải được đánh giá bằng:

- cùng số iteration hoặc cùng giao thức train
- cùng script render
- cùng script eval
- cùng `psnr_max` nội bộ
- lưu lại:
  - PSNR mean
  - SSIM mean
  - LPIPS mean
  - Score mean
  - thời gian train
  - peak VRAM nếu có
  - lỗi vận hành nếu có

### 5.4 Ngưỡng giữ hay bỏ một hướng

- Nếu một hướng không tăng được ít nhất `+0.5` điểm Score trung bình trên bộ scene dev, bỏ.
- Nếu một hướng tăng điểm nhưng làm pipeline phức tạp hơn rất nhiều, chỉ giữ nếu nó tăng rõ ràng và lặp lại được.
- Nếu một hướng chưa chạy sạch hết pipeline train -> render -> eval, chưa được tính là ứng viên.

## 6. Kết luận chốt về model

### 6.1 Generator chính

Chốt:

- **Ứng viên generator chính số 1 = nhánh `feature/depth-anything-v2`**

Lý do:

- nó thừa hưởng trực tiếp từ `feature/mip-splatting`
- vẫn nằm trong họ 3DGS Inria mà cả đội đã hiểu rõ nhất
- có fix bug train/render mismatch quan trọng
- có thêm depth prior, là hướng nâng cấp hợp lý nhất trước khi đổi trainer
- effort bổ sung để chạy sạch và benchmark nhỏ hơn pivot sang `gsplat-mcmc`

Điều kiện để chuyển từ "ứng viên số 1" thành "generator chính đã chốt":

- phải có benchmark sạch trên bộ scene dev
- phải có kết quả render -> eval -> package chạy trọn pipeline
- phải cho thấy ít nhất không thua cấu hình `AA only`

### 6.2 Generator dự phòng

Chốt:

- **Generator dự phòng / R&D = nhánh `feature/gsplat-mcmc`**

Chỉ kích hoạt nếu:

- benchmark đầy đủ trên nhánh chính cho thấy trần điểm quá sớm
- hoặc OOM/compute vẫn là khoá chặn chính
- hoặc nhánh này được chứng minh thắng nhánh chính trên bộ scene dev

### 6.3 Hướng tạm dừng

- `compact/compact-gaussian`: tạm dừng
- antenna-focus: chưa đủ để thành hướng ưu tiên độc lập

## 7. Kết luận chốt về ý tưởng "1 model sinh + 1 model làm sạch"

Ý tưởng này phải tách thành 2 loại.

### 7.1 Loại không nên làm

Không ưu tiên:

- diffusion img2img tổng quát
- super-resolution tổng quát
- enhancer chỉ dựa vào ảnh render để "vẽ lại"

Lý do:

- dễ làm ảnh đẹp bằng mắt hơn
- nhưng dễ sai chi tiết cơ khí BTS
- LPIPS/SSIM/PSNR có thể giảm dù ảnh nhìn sắc nét hơn

### 7.2 Loại nên làm nếu cần model thứ 2

Nên làm:

- **reference-guided post-refinement có ràng buộc hình học**

Cụ thể:

1. render ảnh từ generator chính
2. tìm 3-5 ảnh train gần pose nhất
3. dùng sparse/pose để project texture thật từ ảnh train lên vùng nhìn thấy được
4. blend theo góc nhìn, visibility, occlusion, confidence

Bản chất:

- đây không phải "AI vẽ lại tự do"
- đây là hậu xử lý tham chiếu có ràng buộc, rủi ro thấp hơn nhiều với metric

Kết luận:

- nếu cần "model thứ 2", không chọn enhancer tổng quát
- chọn **Tier-1 post-refinement bằng geometric reprojection / texture blending**

## 8. Roadmap thực thi

### Phase A - Dọn bối cảnh và đồng bộ tài liệu

Mục tiêu:

- ngừng tình trạng docs nói một đằng, code một nẻo

Việc cần làm:

1. đồng bộ lại README và các file hướng dẫn với trạng thái code thật
2. đánh dấu rõ script nào đã chạy thật, script nào mới ở mức ý tưởng
3. giữ `plan.md` này làm file tham chiếu chính

### Phase B - Benchmark generator chính

Mục tiêu:

- xác nhận ứng viên generator chính có thực sự là hướng đáng đi tiếp

Việc cần làm:

1. chuyển sang nhánh `feature/depth-anything-v2`
2. xác nhận lại toàn bộ:
   - train flags
   - render flags
   - depth prior input
   - package output
3. bổ sung benchmark đã gắn nhãn rõ vào thư mục `Kết quả`
4. chạy 3 scene dev bằng 2 config:
   - antialiasing only
   - antialiasing + depth prior
5. tổng hợp Score mean, PSNR, SSIM, LPIPS

Tiêu chí qua Phase B:

- nếu `+depth prior` thắng rõ ràng, đây sẽ là cấu hình generator chính
- nếu `+depth prior` không ăn hoặc rất bất ổn, giữ `antialiasing only` làm base
- nếu không tạo được benchmark sạch lưu vào `Kết quả`, không được coi là đã qua Phase B

### Phase C - Khoá generator chính và dừng thử nghiệm lung tung

Mục tiêu:

- chốt một pipeline train/render cho private

Việc cần làm:

1. sau benchmark, chốt một config generator duy nhất
2. dừng mở thêm branch model mới trong một khoảng thời gian cố định
3. chạy full 5 scene public nếu cần để có trung bình vững hơn
4. nếu score public còn cửa, render tiếp private

### Phase D - Hậu xử lý tham chiếu

Điều kiện vào:

- generator chính đã chốt
- đã có kết quả render ổn định
- vẫn cần thêm điểm

Việc cần làm:

1. thiết kế script hậu xử lý sau `04_render_test_poses.py`
2. input:
   - ảnh render
   - sparse / points
   - poses train
   - ảnh train
3. output:
   - ảnh render được gia cố texture thật ở vùng nhìn thấy
4. đánh giá trên public set như một bước hậu xử lý, không chen vào train

Đây là hướng "model thứ 2" được phép ưu tiên.

### Phase E - Nhánh dự phòng gsplat-MCMC

Điều kiện vào:

- nhánh chính không đủ tiến bộ
- hoặc có bằng chứng rõ ràng `gsplat-MCMC` ăn điểm trên bộ scene dev

Việc cần làm:

1. chuẩn hoá benchmark 3 scene dev giống hệt nhánh chính
2. chạy:
   - gsplat-MCMC có antialiased
   - nếu cần, fine-tune LPIPS
3. so sánh trực tiếp với generator chính

Tiêu chí giữ:

- chỉ giữ nếu nó thắng generator chính trên Score mean và không phá pipeline package

## 9. Việc phải làm ngay

### Ưu tiên 1

- benchmark ứng viên generator chính `feature/depth-anything-v2` trên bộ scene dev

Cần trả lời 3 câu hỏi bằng số liệu thật:

1. antialiasing only trung bình được bao nhiêu?
2. thêm depth prior tăng được bao nhiêu?
3. giá của depth prior là VRAM / thời gian / độ ổn định ra sao?

Trước khi trả lời 3 câu hỏi này, cần thêm 1 việc nền:

0. đưa các run benchmark mới vào `Kết quả` với tên rõ ràng, không để lẫn sang `99_chua_phan_loai`

### Ưu tiên 2

- sau khi khoá generator chính, viết script/refiner hậu xử lý tham chiếu

### Ưu tiên 3

- chỉ khi 2 bước trên không đủ mới quay sang `feature/gsplat-mcmc`

## 10. Việc không được làm nữa

- không mở thêm branch model mới khi các branch hiện có còn chưa benchmark xong
- không đánh giá bằng một scene đơn lẻ rồi suy rộng
- không ưu tiên enhancer tổng quát để "vẽ lại đẹp hơn"
- không tiếp tục đầu tư vào `compact/compact-gaussian` trừ khi tất cả hướng khác đã bị loại

## 11. Danh sách công việc cụ thể

### Nhóm 1 - Generator chính

- [ ] checkout `feature/depth-anything-v2`
- [ ] xác minh script train/render/package trên nhánh này
- [ ] tạo cấu trúc lưu benchmark rõ ràng trong `Kết quả/02_mip/` hoặc `Kết quả/03_depth/`
- [ ] chọn 3 scene dev cố định
- [ ] chạy config `AA only`
- [ ] chạy config `AA + depth prior`
- [ ] tổng hợp bảng kết quả
- [ ] chỉ sau đó mới chốt config generator

### Nhóm 2 - Hậu xử lý tham chiếu

- [ ] định nghĩa input/output của script post-refine
- [ ] chọn chiến lược tìm train views gần nhất
- [ ] chọn visibility/occlusion rule
- [ ] implement blend texture thật lên ảnh render
- [ ] eval trên public set

### Nhóm 3 - Dự phòng R&D

- [ ] giữ `feature/gsplat-mcmc` sạch và đọc được
- [ ] chỉ benchmark khi có mốc go/no-go rõ ràng

## 12. Kết luận cuối cùng

Chốt quyết định:

- không tiếp tục ở trạng thái "nhiều nhánh, nhiều ý tưởng, chưa có trục chính"
- trục chính từ giờ là:
  - **ứng viên generator chính số 1: `feature/depth-anything-v2`**
  - **hướng model thứ 2 nếu cần: reference-guided post-refinement có ràng buộc hình học**
  - **dự phòng duy nhất: `feature/gsplat-mcmc`**

Mọi mở rộng tiếp theo phải bám theo trục này, không quay lại kiểu mở thêm một nhánh model hoàn toàn mới chỉ vì chưa quyết được.

Điểm sửa quan trọng của bản kế hoạch này là:

- **không đổi chiến lược chính**
- **nhưng hạ mức khẳng định**
- nghĩa là `feature/depth-anything-v2` vẫn là hướng nên đi tiếp nhất, chỉ là hiện tại mới ở mức "ứng viên số 1", chưa phải "đã chứng minh xong bằng benchmark sạch lưu trong `Kết quả`"**
