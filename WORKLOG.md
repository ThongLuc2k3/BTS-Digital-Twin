# WORKLOG

## 2026-07-24 — Thí nghiệm A: cách ly confound `LOW_VRAM_PROFILE` cho `B2`

### Bối cảnh
Theo `.ai-debate/01-07` và `trao đổi.md` (mục "Tổng hợp tranh luận 2026-07-24"), nhánh `B2`/`prepared` đã có failed run thật (`Score ~0.40` so với baseline `0.6731`), và nghi phạm số 1 là `LOW_VRAM_PROFILE=auto` tự động bật trên Kaggle/Colab (`RESOLUTION=4`, tắt hoàn toàn densification). `.ai-debate/07_claude_audit.md` xác định rõ: phải chạy **2 run** (2a override tích cực + 2b control âm) mới đủ bằng chứng nhân quả — chỉ chạy 1 run rồi kết luận là không đủ.

### Đã làm (không cần GPU, chuẩn bị thí nghiệm)
1. `pipeline/scripts/03_train_3dgs.sh`: sửa 1 dòng để `MODEL_DIR` tôn trọng biến môi trường override thay vì hard-code theo `SCENE` — để các run isolation không ghi đè lên `gs_model` thật (baseline hiện đã thiếu artifact, không nên phá thêm).
2. Viết `pipeline/scripts/generate_b2_isolation_notebooks.py`, sinh ra 2 notebook từ `downloads/B2_done.ipynb`:
   - `downloads/B2_isolation_2a_low_vram_off.ipynb` — `LOW_VRAM_PROFILE=0` ép tường minh, `RESOLUTION=-1`, KHÔNG dùng `--depths`.
   - `downloads/B2_isolation_2b_low_vram_control.ipynb` — `LOW_VRAM_PROFILE=1` ép tường minh (tái lập đúng cấu hình cũ), cũng KHÔNG dùng `--depths`.
   - Cả hai redirect output sang `pipeline/work/hcm0031/trick_runs/b2_isolation_<id>/` (không đụng `gs_model` thật), và có cell tự động in kết luận GO/STOP so với baseline `0.6731` và vùng fail cũ `~0.40`.
3. Cập nhật `trick/hcm0031/experiment_matrix.csv`: sửa nhãn sai `baseline_ref.source_mode` từ `raw` → `prepared` (theo bằng chứng `cfg_args` thật), đánh dấu `prepared_train_template` là `superseded` (đã có kết quả thật), thêm 2 dòng `b2_isolation_2a/2b` trạng thái `queued_gpu`.

### Next steps (cần GPU — Kaggle/Colab)
1. Chạy `downloads/B2_isolation_2a_low_vram_off.ipynb` trên Kaggle/Colab GPU (full run từ đầu: dataset setup, COLMAP CUDA build, GS repo clone, rồi train+render+eval).
2. Chạy `downloads/B2_isolation_2b_low_vram_control.ipynb` tương tự.
3. Đối chiếu 2 kết quả:
   - Nếu 2a phục hồi gần `0.6731` VÀ 2b tái lập lại vùng `~0.40` → xác nhận `LOW_VRAM_PROFILE` là nguyên nhân chính, `B2`/`prepared` (không depth) chưa bị loại, có thể tiếp tục thử `prepared + true depths` sau đó.
   - Nếu 2a vẫn sập → `LOW_VRAM_PROFILE` không phải nguyên nhân duy nhất/chính, cần điều tra tiếp render-parity hoặc bản thân prepared source trước khi thử depth.
4. Cập nhật `trick/hcm0031/experiment_matrix.csv` (điền `full_image_score`, `status=done`) và `trao đổi.md` mục 7/8 sau khi có số liệu, theo đúng khuyến nghị của `.ai-debate/07_claude_audit.md`.
5. Song song (không phụ thuộc kết quả trên, chi phí thấp — xem `.ai-debate/02_claude_review.md` mục 7): Oracle trần khả thi (geometry-assisted warp/blend, không cần train) để biết `0.85` có khả thi với 3DGS thuần hay không.

## 2026-07-26 — Kết quả thật 2a/2b: LOW_VRAM_PROFILE KHÔNG phải nguyên nhân, tìm ra nghi phạm thật

### Kết quả run (25/07, Kaggle T4x2)
- `2a` (`LOW_VRAM_PROFILE=0` ép tắt): full-image `PSNR 10.98 / SSIM 0.443 / LPIPS 0.274 / Score 0.4891`
- `2b` (`LOW_VRAM_PROFILE=1` control): full-image `PSNR 10.57 / SSIM 0.363 / LPIPS 0.423 / Score 0.4032` (tái lập đúng vùng fail cũ)
- Cả hai đều train tốt: `train-set PSNR` nội bộ (log `train.py`) là `28.85` (2a) và `29.32` (2b) — tốt hơn cả baseline. Vậy training hội tụ bình thường; vấn đề nằm ở test-time eval, không phải training.
- **Kết luận: `LOW_VRAM_PROFILE` bị loại — 2a đã tắt nó mà vẫn sập gần như 2b.**

### Điều tra sâu hơn (không cần GPU, dùng artifact cục bộ)
1. Giả thuyết đầu tiên (sai, đã bác bỏ bằng dữ liệu thật): lệch camera intrinsics giữa `raw` (`train/sparse/0`, SIMPLE_RADIAL 1320x989) và `prepared` (`colmap/dense/sparse/0`, PINHOLE 1310x981, do `pycolmap.undistort_images` trong `prepare_round1_scene.py`). Bị bác bỏ vì `baseline_ref` (`0.6731`, đang hoạt động tốt) **cũng** train trên `source_path=.../colmap/dense` (xem `cfg_args` local) — nên lệch intrinsics không thể là biến phân biệt.
2. So sánh trực tiếp `cfg_args` baseline vs bản B2 pilot cũ đã tải về (`downloads/hcm0031_b2_ready/`):
   - baseline: `resolution=-1, train_test_exp=False`
   - B2 pilot cũ: `resolution=4, train_test_exp=True`, `pipeline_train_flags.json: {antialiasing:true, exposure_comp:true}`
   - Log train của **cả 2a và 2b** đều in `exposure=1` — tức `EXPOSURE_COMP` (mặc định `1` trong `03_train_3dgs.sh`, dòng 23) chưa từng bị tắt ở thí nghiệm cách ly, nên nó không phải là biến được kiểm soát trong Thí nghiệm A.
3. Kiểm chứng bằng ảnh thật (CPU, không cần GPU): so `render_b2_pilot` cũ (`downloads/hcm0031_b2_ready/bundles/renders_images/renders_b2_pilot/DJI_20241227155343_0023_V.png`) với GT test và với baseline render:
   - Phase-correlation shift = `(0,0)` cho cả baseline lẫn B2 → **không lệch hình học/pixel**.
   - Nhìn trực quan: B2 render đúng bố cục, đúng công trình, thẳng hàng với GT — nhưng toàn ảnh bị tối/xỉn màu và hơi mờ so với GT và so với baseline (baseline gần như trùng khớp GT).
   - PSNR đo trực tiếp trên đúng 1 ảnh: baseline vs GT = `21.95`, B2 vs GT = `9.86` — khớp đúng độ lớn với batch eval thật.

### Kết luận
- **Nghi phạm thật: `EXPOSURE_COMP=1` (`--train_test_exp` trong 3DGS train.py).** Baseline tắt cờ này; mọi run `prepared` (pilot cũ, 2a, 2b) đều bật mặc định. `render_round1_test_poses.py` không có logic áp dụng exposure compensation cho pose test mới (test pose chưa từng thấy khi train không có tham số exposure học được) → ảnh render đúng hình học nhưng sai tông màu toàn cục → PSNR sập trong khi SSIM/LPIPS chỉ giảm vừa phải (khớp chính xác với số liệu quan sát).
- Thí nghiệm A (2a/2b) đã vô tình giữ cố định đúng biến gây lỗi thật (`EXPOSURE_COMP`) trong khi chỉ đổi biến không liên quan (`LOW_VRAM_PROFILE`) → cả 2 run đều sập là điều tất yếu, không chứng minh được gì về bản chất `prepared`/`depth`.
- Đã **revert** một sửa đổi thử nghiệm sai ở `render_round1_test_poses.py` (dựa trên giả thuyết intrinsics đã bị bác bỏ) — không có thay đổi code nào được giữ lại từ phiên này.
- Đã thêm hàng `b2_isolation_2c_exposure_off` vào `trick/hcm0031/experiment_matrix.csv` (status `planned`).

### Next steps (cần GPU)
1. Chạy lại đúng config `prepared, LOW_VRAM_PROFILE=0, RESOLUTION=-1, EXPOSURE_COMP=0` (khớp chính xác baseline, chỉ khác nguồn depth/dense) — đây là `b2_isolation_2c_exposure_off` trong experiment_matrix.
2. Nếu score phục hồi gần `0.6731` → xác nhận `EXPOSURE_COMP` là nguyên nhân thật, mở lại nhánh `prepared + true depths` (mục tiêu gốc của B2).
3. Nếu vẫn sập → còn nghi phạm khác chưa lộ diện, cần lặp lại quy trình chẩn đoán bằng ảnh thật (train PSNR vs test PSNR, phase-correlation, so sánh cfg_args) trước khi thử thêm biến.
4. **Chưa nên** bắt đầu train antenna-focus / two-stage (nền + antenna) cho tới khi có tín hiệu `prepared`/depth sạch — nếu không, mọi cải thiện đo được trên vùng anten có thể chỉ là do đo lường sai (exposure), không phải do thay đổi thật.

### Đã chuẩn bị xong (không cần GPU)
- Viết `pipeline/scripts/generate_b2_isolation_2c_notebook.py`, sinh `downloads/B2_isolation_2c_exposure_off.ipynb` từ `downloads/B2_done.ipynb`. Đã kiểm tra cell config + cell train/render/eval patch đúng (`env['EXPOSURE_COMP']='0'`, `env['LOW_VRAM_PROFILE']='0'`, `env['RESOLUTION']='-1'`, output redirect sang `trick_runs/b2_isolation_2c_exposure_off/`, không đụng `gs_model` thật).
- Cập nhật `experiment_matrix.csv`: hàng `b2_isolation_2c_exposure_off` chuyển `planned` -> `queued_gpu`.

### Next action (cần GPU, việc kế tiếp thật sự)
- Chạy `downloads/B2_isolation_2c_exposure_off.ipynb` trên Kaggle/Colab GPU (full run: dataset setup, COLMAP CUDA build, GS clone, rồi train+render+eval — cell cuối tự in kết luận GO/STOP so với baseline `0.6731`).
- Dán lại output/score vào đây hoặc báo lại để điền `experiment_matrix.csv` và quyết định bước kế (mở lại `prepared+depth` nếu GO, hoặc điều tra nghi phạm khác nếu vẫn sập).

## 2026-07-26 (tiếp) — Deadline gấp: gộp 2c + depth thành 1 lần chạy submission-candidate

User báo gấp deadline nộp bài, yêu cầu tự quyết định và bắt đầu ngay, không chờ xác nhận thêm (bỏ qua bước tải render 2a để đối chiếu thêm — không bắt buộc, đã có đủ bằng chứng).

### Quyết định
Thay vì chạy `2c` (chỉ tắt `EXPOSURE_COMP`, không depth) rồi mới chạy thêm 1 lần depth riêng (tốn 2 chu kỳ GPU ~2-2.5h/lần), gộp thẳng thành **1 notebook**: `prepared` + `--depths` thật (dùng lại cơ chế `TRAIN_EXTRA_ARGS_RAW` đã chạy thành công ở `B2_done 1/2/2-safe.ipynb` cũ) + `EXPOSURE_COMP=0` + `LOW_VRAM_PROFILE=0` + `RESOLUTION=-1` (khớp mọi setting của `baseline_ref`).

**Đánh đổi đã biết và chấp nhận:** nếu run này vẫn sập điểm, sẽ không tách bạch được là do depth hay do nghi phạm khác còn sót — nhưng dưới áp lực deadline, gộp lại là lựa chọn hợp lý hơn tuần tự.

### Đã làm (không cần GPU)
- Viết `pipeline/scripts/generate_b2_final_candidate_notebook.py`, sinh `downloads/B2_final_candidate_depth_exposure_off.ipynb` từ `downloads/B2_done.ipynb`.
- Đã kiểm tra cấu trúc cell khớp đúng: cell vá `03_train_3dgs.sh` (thêm `TRAIN_EXTRA_ARGS_RAW`), cell config (assert `depth_maps` tồn tại từ pha `RUN_DENSE=1` đầu notebook, assert `GS_REPO` hỗ trợ `--depths`), cell train/render/eval (set `LOW_VRAM_PROFILE=0`, `RESOLUTION=-1`, `EXPOSURE_COMP=0`, `TRAIN_EXTRA_ARGS_RAW=--depths <dir>`), cell báo cáo kết quả so với baseline.
- **Khác với 2a/2b/2c**: run này ghi đè trực tiếp lên `gs_model` mặc định (không redirect `trick_runs/`) vì mục tiêu là ứng viên nộp bài, không phải run cách ly — nhưng lưu ý mọi thứ chạy trên VM ephemeral của Kaggle/Colab (clone repo riêng), không đụng tới file local của máy user.
- Cập nhật `experiment_matrix.csv`: `b2_isolation_2c_exposure_off` chuyển `superseded` (vẫn giữ notebook để dùng lại nếu cần tách biến sau), thêm hàng `b2_final_candidate_depth_exposure_off` (`queued_gpu`).

### Lưu ý bảo mật phát hiện được (chưa xử lý, không phải do tôi tạo ra)
`downloads/B2_done.ipynb` (và mọi notebook sinh từ nó, kể cả các notebook isolation trước) có **hardcode sẵn 1 GitHub personal access token** trong cell 1 (`GITHUB_TOKEN = 'ghp_...'`). Notebook này được tải lên Kaggle/Colab — nếu notebook từng bị share công khai hoặc dataset/kernel public, token này lộ. Nên thu hồi (revoke) token đó trên GitHub và thay bằng cách nhập token qua biến môi trường/secret của Kaggle thay vì hardcode, sau khi xong deadline.

### Next action (cần GPU, ưu tiên cao nhất hiện tại)
- Chạy `downloads/B2_final_candidate_depth_exposure_off.ipynb` trên Kaggle/Colab GPU.
- Nếu score >= baseline (`0.6731`) → dùng ngay làm bài nộp.
- Nếu thấp hơn nhưng gần → cân nhắc dùng tạm, đồng thời báo lại số liệu để phân tích thêm nếu còn thời gian.
- Nếu vẫn sập mạnh → quay lại `b2_isolation_2c_exposure_off.ipynb` (đã có sẵn, chỉ tắt exposure không depth) để tách bạch nguyên nhân, hoặc nộp tạm bằng `baseline_ref` (`0.6731`) nếu hết thời gian.

## 2026-07-26 (tiếp) — `B2_final_candidate_depth_exposure_off.ipynb` sập ngay ở train: `--depths` không tương thích với output COLMAP dense-stereo

### Triệu chứng
Run trên Colab, cell `b2-train-render-eval` fail ngay (~2s sau khi bắt đầu train):
```
Error: depth_params.json file not found at path '.../colmap/dense/sparse/0/depth_params.json'.
```
`RETURN CODE = 1`, notebook dừng ở exception. Không tốn GPU-time đáng kể (fail sớm), nhưng cell đã dừng và cần sửa trước khi chạy lại.

### Root cause (đã xác nhận bằng cách đọc source thật của `graphdeco-inria/gaussian-splatting` @ `main`)
Không chỉ thiếu 1 file — cơ chế `--depths` của repo gốc **không tương thích về định dạng** với output COLMAP dense-stereo mà pipeline này tạo ra:

1. `train.py`/`scene/dataset_readers.py` khi có `--depths <dir>` sẽ **bắt buộc** đọc `<base_dir>/sparse/0/depth_params.json` (không tồn tại nếu chưa chạy bước tạo riêng) — đây là lỗi chặn ngay lập tức.
2. File đó được tạo bởi `utils/make_depth_scale.py` (script phụ trợ của repo GS, KHÔNG có trong pipeline này) — và script này tự nó cũng expect input là **PNG single-channel 16-bit, inverse-depth, đặt tên `{ten_anh_khong_duoi}.png`** (đọc bằng `cv2.imread` rồi `/2**16`).
3. Nhưng `04_run_colmap_dense.sh` (COLMAP `patch_match_stereo` + `stereo_fusion`) tạo ra `stereo/depth_maps/*.geometric.bin` — **binary format riêng của COLMAP** (metric depth, không phải PNG, không phải inverse-depth, khác hoàn toàn convention).
4. `depths_arg = f'--depths {depth_maps_dir}'` trong `generate_b2_final_candidate_notebook.py` (và cả `B2_done 1/2/2-safe.ipynb` cũ trước đó — WORKLOG trước ghi nhầm là "đã chạy thành công", thực ra chưa từng verify tới bước train thật) trỏ thẳng `depth_maps_dir` (chứa `.bin`) vào `--depths` của GS gốc → **sai định dạng ngay từ gốc**, không chỉ là thiếu 1 file JSON.

Kết luận: **"true depth supervision" như đang wire hiện tại chưa từng chạy được** — không phải do quên 1 bước, mà do 2 định dạng dữ liệu không khớp nhau. Cần 1 script convert riêng (đọc COLMAP `.bin` depth map → PNG 16-bit inverse-depth + tự viết `depth_params.json` với `scale=1, offset=0` mỗi ảnh, vì depth COLMAP dense đã cùng scale metric với sparse reconstruction, không cần rescale kiểu monodepth) thì `--depths` mới dùng được đúng. Việc này **chưa được viết**, và là công việc mới (không nhỏ, cần test kỹ định dạng binary của COLMAP), rủi ro cao nếu làm vội sát deadline.

### Quyết định (người dùng chọn)
Bỏ nhánh depth (cần code mới, rủi ro cao sát deadline), chạy ngay `downloads/B2_isolation_2c_exposure_off.ipynb` (đã có sẵn, chỉ tắt `EXPOSURE_COMP`, không depth) để lấy điểm ứng viên nộp bài an toàn. Đã xác nhận notebook sẵn sàng chạy đúng như thiết kế (không cần sửa gì): `EXPOSURE_COMP=0`, `LOW_VRAM_PROFILE=0`, `RESOLUTION=-1`, không dùng `--depths`, ghi vào `pipeline/work/hcm0031/trick_runs/b2_isolation_2c_exposure_off/` (không đụng `gs_model` thật), cell cuối tự in kết luận GO/STOP so với baseline `0.6731`.

### Next action (cần GPU, ưu tiên cao nhất hiện tại)
- Chạy `downloads/B2_isolation_2c_exposure_off.ipynb` trên Kaggle/Colab GPU (full run từ đầu).
- Nếu score phục hồi gần `0.6731` → dùng kết quả này (đã cấy sẵn ở `trick_runs/b2_isolation_2c_exposure_off/`) làm ứng viên nộp bài an toàn; copy sang `gs_model`/vị trí nộp bài chính thức nếu cần.
- Sau khi có ứng viên an toàn, nếu còn thời gian mới quay lại nhánh depth (cần viết converter COLMAP `.bin` → PNG inverse-depth + `depth_params.json`, xem mục trên) — KHÔNG chặn việc nộp bài vào việc này.

## 2026-07-26 (tiếp) — `2c` chạy OOM lúc iter 11200/30000, sinh notebook submission-candidate mới (exposure_off + safe-densify)

### Kết quả `B2_isolation_2c_exposure_off.ipynb` (Colab, T4 14.56GiB)
`torch.OutOfMemoryError` lúc `densify_and_prune`, iteration 11200/30000 (~37%, 46 phút). Nguyên nhân: `LOW_VRAM_PROFILE=0` ép tắt hoàn toàn (để khớp đúng baseline) → không giới hạn tốc độ densify → số gaussian tăng không kiểm soát → tràn VRAM. Đây là lỗi hạ tầng khác (GPU/VRAM), không liên quan gì tới `EXPOSURE_COMP`/`--depths`.

Vì `LOW_VRAM_PROFILE` (bật/tắt) đã bị loại là nguyên nhân sập điểm thật (bằng chứng 2a/2b trước đó), bật lại **giới hạn densify riêng lẻ** (không bật cả `LOW_VRAM_PROFILE=1` vì cái đó ép `RESOLUTION` xuống `4`, lệch baseline) là an toàn.

### Đã làm (không cần GPU)
Viết `pipeline/scripts/generate_b2_submission_candidate_notebook.py`, sinh `downloads/B2_submission_candidate_exposure_off_safe_densify.ipynb` từ `downloads/B2_done.ipynb`. Cấu hình: `prepared`, `EXPOSURE_COMP=0`, `RESOLUTION=-1`, `LOW_VRAM_PROFILE=0`, thêm `DENSIFY_UNTIL_ITER=15000/DENSIFY_FROM_ITER=500/DENSIFICATION_INTERVAL=100/PERCENT_DENSE=0.01/OPACITY_RESET_INTERVAL=3000` (chặn OOM) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, không dùng `--depths`. Khác `2c`: ghi thẳng vào `gs_model`/`eval_metrics_b2_pilot.csv` thật (không redirect `trick_runs`) vì đây là ứng viên nộp bài, không phải run cách ly. Đã verify cell config + cell train/render/eval patch đúng.

### Next action (cần GPU, ưu tiên cao nhất — "một phát ăn ngay", không debug thêm)
- Chạy `downloads/B2_submission_candidate_exposure_off_safe_densify.ipynb` trên Colab/Kaggle GPU.
- Nếu score ≥ baseline (`0.6731`) hoặc xấp xỉ → dùng ngay làm bài nộp.
- Nếu vẫn OOM → giảm thêm nữa: `DENSIFY_UNTIL_ITER` xuống ~`10000` hoặc thử GPU VRAM lớn hơn (A100/L4 nếu có trên Colab Pro).
- Nếu train xong nhưng score vẫn thấp bất thường → còn nghi phạm khác chưa lộ diện ngoài exposure/densify, cần xem log + ảnh render trước khi thử thêm.

## 2026-07-27 — `B2_submission_candidate_exposure_off_safe_densify.ipynb` chạy xong: score xấp xỉ baseline, xác nhận `EXPOSURE_COMP`; suýt báo cáo nhầm điểm do bug `psnr_max` cũ tái xuất hiện

### Kết quả thật (train full 30000 iter, không OOM)
`eval_metrics_b2_pilot.csv` (dùng đúng `eval_round1_metrics.py` mặc định `psnr_max=50.0`, giống hệt cách chấm baseline):
```
psnr  = 21.6847
ssim  = 0.6822
lpips = 0.1585
score = 0.6714   (baseline 0.6731, delta = -0.0017 -> xap xi bang, trong sai so)
```
**Xác nhận `EXPOSURE_COMP=1` đúng là nguyên nhân gây sập điểm** (mọi run `prepared` trước đó ~0.40-0.49). Densify throttle cũng đã chặn được OOM (train xong đủ 30000 iter, không lặp lại lỗi VRAM của `2c`). Có ứng viên nộp bài hợp lệ (`gs_model` thật, `pipeline/work/hcm0031/`).

### Suýt báo cáo sai điểm: bug `psnr_max` cũ (đã ghi nhận từ trước ở `.ai-debate/02_claude_review.md`) tái xuất hiện
Notebook (kế thừa từ `downloads/B2_done.ipynb`) có sẵn 1 cell `reeval_latest` (từ `pipeline/scripts/manage_b2_artifacts.py`) chạy re-eval riêng với `PSNR_MAX = 30.0` — khác chuẩn `psnr_max=50.0` cả team đã thống nhất (`trao đổi.md` dòng 331). Cùng 1 bộ ảnh render, cùng PSNR/SSIM/LPIPS, chỉ đổi hằng số chuẩn hoá này thôi mà Score nhảy từ `0.6714` lên `0.7581` — **chênh 0.087 điểm ảo, không do model tốt hơn**. User ban đầu tưởng `0.7581` là thật (thắng đậm baseline). Đã phát hiện và giải thích kịp thời trước khi ghi nhầm vào kết quả.

**Root cause**: `pipeline/scripts/manage_b2_artifacts.py` dòng 394 (`--psnr_max` default) và cell tương ứng đã bake sẵn trong `downloads/B2_done.ipynb` (dòng ~14179, `PSNR_MAX = 30.0`) đều default sai `30.0` thay vì `50.0` — dù bug này đã được `.ai-debate/02_claude_review.md`/`03/04/05` ghi nhận và khuyến nghị chuẩn hoá từ trước, code chưa bao giờ được sửa thật.

### Đã sửa (không cần GPU)
- `pipeline/scripts/manage_b2_artifacts.py` dòng 394: default `--psnr_max` từ `30.0` → `50.0`.
- `downloads/B2_done.ipynb` (template gốc dùng cho MỌI notebook sinh ra sau này qua `generate_b2_*.py`): sửa `PSNR_MAX = 30.0` → `PSNR_MAX = 50.0` trong cell `reeval_latest`.
- Các notebook đã sinh ra TRƯỚC fix này (`B2_isolation_2c...`, `B2_final_candidate...`, `B2_submission_candidate...` trong `downloads/`) vẫn còn mang bug cũ trong cell `reeval_latest` — nếu chạy lại cell đó, nhớ tự sửa `PSNR_MAX` thành `50.0` trước, hoặc bỏ qua cell đó (không cần thiết, số điểm chính đã có ở cell report riêng dùng đúng `psnr_max=50.0`).

### Next action
- **Điểm dùng để quyết định nộp bài / so sánh baseline: LUÔN LÀ `psnr_max=50.0`** (số `0.6714` ở trên), không bao giờ dùng số từ cell `reeval_latest` cũ chưa sửa.
- Ứng viên hiện tại (`0.6714`, xấp xỉ baseline) đã an toàn để nộp nếu hết thời gian. Nếu còn thời gian, có thể thử thêm variant khác (vd. tinh chỉnh densify threshold để không giới hạn sớm quá, hoặc quay lại nhánh depth nếu viết xong converter).

## 2026-07-27 (tiếp) — Antenna-focus 2-stage (30k nền + 30k anten, tổng 60k): hạ tầng mới, CHƯA chạy GPU

User hỏi ý tưởng "train 2 lần: 1 lần nền, 1 lần riêng vùng anten, tổng 60k" và chọn phương án đầu tư viết mask thật (còn nhiều giờ). Trước khi làm, đối chiếu số liệu `M0` thật đã có (`trao đổi.md`): `full-image=0.6731`, **`tower-crop=0.7064` (đang là vùng TỐT NHẤT)**, `skyline-crop=0.6384` (vùng YẾU nhất). Anten **không phải** điểm yếu hiện tại của `hcm0031` — đây là thử nghiệm thăm dò, không phải vá lỗi đã biết. Thực nghiệm cũ trên scene khác (`HCM0421` Round2) cho `antenna-focus` gần như trung lập (`0.6611` vs baseline `0.6616`) — không có tín hiệu mạnh là sẽ thắng.

### Thiết kế kỹ thuật đã chốt
- **Cơ chế**: Stage 1 = train full-scene 30k (đúng config đã xác nhận: `prepared + EXPOSURE_COMP=0 + safe-densify`, xem mục trước). Stage 2 = resume checkpoint `chkpnt30000.pth`, train tiếp tới 60k, **CỘNG THÊM** (không thay thế) 1 loss `L1` có trọng số cao trên vùng mask anten — an toàn hơn hẳn cách zero-out loss ngoài mask (tránh catastrophic forgetting nền).
- **Cố tình KHÔNG** nâng `position_lr_max_steps` lên 60000: để nguyên mặc định 30000 (LR vị trí đã gần bằng phẳng cuối stage 1) → stage 2 chủ yếu tinh chỉnh màu/opacity/scale qua extra loss, tránh xáo trộn hình học đã tốt của stage 1.
- **Mask vùng anten cho ảnh TRAIN**: chiếu `tower_bbox3d.json` (đã có sẵn, world-frame) qua pose thật đọc trực tiếp từ `colmap/dense/sparse/0/{cameras,images}.bin` (đúng nguồn `03_train_3dgs.sh` dùng để train) — convex hull + dilate 12px, cùng phương pháp/tham số với `trick/scripts/bootstrap_tower_masks.py` (đã dùng cho M0 tower-crop, có bằng chứng hoạt động tốt: `0.7064`).
- **Patch `train.py`** (GS_REPO clone tươi từ `graphdeco-inria/gaussian-splatting@main` mỗi lần chạy Colab, không sửa được trước): chèn code đọc mask theo `viewpoint_cam.image_name` + cộng thêm `antenna_loss_weight * Ll1_antenna` vào `loss` ngay trước `loss.backward()`. Đã verify anchor text khớp chính xác 100% với source thật (tải trực tiếp từ GitHub, không qua tool tóm tắt) và patch xong vẫn compile được.
- Không cần patch `03_train_3dgs.sh` thêm CLI arg nào — `ANTENNA_MASK_DIR`/`ANTENNA_LOSS_WEIGHT` đọc thẳng qua `os.environ` trong `train.py`, tự động kế thừa qua chuỗi subprocess `env=` từ notebook.

### Bug phát hiện & sửa khi làm (không liên quan trực tiếp nhưng chặn đường nếu không sửa)
1. **`pipeline/work/hcm0031/tower_bbox3d.json` bị gitignore** (`pipeline/work/` bị ignore toàn bộ) → sẽ KHÔNG có mặt sau khi `git clone` trên Colab, làm bước build mask fail chắc chắn. Đã sửa `.gitignore`: đổi `pipeline/work/` → `pipeline/work/*` + chain `!` (dùng `pipeline/work/` trực tiếp làm git không cho re-include file bên trong dù có negate — đã test thực nghiệm xác nhận, xem lịch sử patch). Đã verify: `tower_bbox3d.json` giờ track được, mọi file khác trong `pipeline/work/` vẫn ignore như cũ. **Cần `git add pipeline/work/hcm0031/tower_bbox3d.json .gitignore` và commit/push trước khi chạy notebook mới trên Colab**, nếu không file vẫn sẽ thiếu sau khi clone.
2. **`pipeline/scripts/eval_round1_metrics.py` có `--tower_bbox3d_json`/`--skyline_top_frac` là stub chưa dùng** — đây là lý do cell "Compare full-image/tower/skyline M0" trong MỌI notebook (kể cả các notebook đã chạy trước đó) luôn in `status: missing` cho 2 vùng crop. Đã viết lại để thực sự tính `psnr/ssim/lpips/score` trên vùng crop (tower: bounding box của convex hull chiếu từ `tower_bbox3d.json` qua `test_poses.csv`; skyline: top `skyline_top_frac` chiều cao ảnh), ghi ra `<out_csv>_tower_crop.csv`/`<out_csv>_skyline_crop.csv` đúng schema cột đã có (`image,psnr,ssim,lpips,score,x0,y0,x1,y1`, khớp `eval_metrics_m0_tower_crop.csv` cũ). **Tác dụng phụ tốt**: từ giờ MỌI lần chạy `05_run_b2_pilot.sh` (không chỉ notebook antenna) tự động có số liệu tower-crop/skyline-crop thật, không còn "missing".
3. Thêm `RUN_STEREO` (mặc định `1`, giữ nguyên hành vi cũ) vào `pipeline/scripts/04_run_colmap_dense.sh`: cho phép bỏ qua `patch_match_stereo`/`stereo_fusion` khi không dùng `--depths` (xác nhận: `render_round1_test_poses.py`/`eval_round1_metrics.py` không phụ thuộc `fused.ply`/`stereo/*`) → bỏ được luôn bước build COLMAP CUDA từ source (~15-30 phút/lần, vì bản `apt install colmap` chỉ CPU và chỉ `patch_match_stereo` mới cần CUDA). Đã áp dụng vào notebook antenna mới (`BUILD_COLMAP_CUDA_IF_NEEDED=False` + `RUN_STEREO=0` ở cả cell dense-only lẫn cell train) — lưu ý: cell build CUDA (cell 6 gốc) có 1 bước verify cuối luôn `raise SystemExit` nếu COLMAP thiếu CUDA, đã phải bọc thêm `if BUILD_COLMAP_CUDA_IF_NEEDED:` để không tự phá chính việc bỏ qua CUDA.

### File mới/sửa (không cần GPU, đã xong)
- `pipeline/scripts/colmap_read_model.py` (mới): vendor rút gọn từ COLMAP `read_write_model.py` (đọc `cameras.bin`/`images.bin`), tránh phụ thuộc version `pycolmap` hay đổi giữa Colab/Kaggle.
- `pipeline/scripts/build_tower_train_masks.py` (mới): sinh mask anten cho ảnh train từ `colmap/dense/sparse/0` + `tower_bbox3d.json`. Có safety check: nếu TOÀN BỘ mask rỗng (nghi lệch hệ toạ độ) thì dừng lại thay vì âm thầm train với mask vô nghĩa.
- `pipeline/scripts/eval_round1_metrics.py`: thêm region crop eval thật (mục 2 ở trên). Đã sửa `psnr_max` default đúng `50.0` từ trước đó (không đổi lại).
- `pipeline/scripts/04_run_colmap_dense.sh`: thêm `RUN_STEREO` gate.
- `.gitignore`: track `pipeline/work/hcm0031/tower_bbox3d.json`.
- `pipeline/scripts/generate_b2_antenna_2stage_notebook.py` (mới) → sinh `downloads/B2_antenna_2stage_30k_30k.ipynb` (30 cell, đã verify cấu trúc + syntax từng cell đã chèn, đã test riêng logic patch `train.py` khớp anchor thật 100% và compile được sau patch).

### Next action (cần GPU — CHƯA chạy lần nào, đây là thử nghiệm thật, có thể không thắng)
1. **Bắt buộc trước tiên**: `git add pipeline/work/hcm0031/tower_bbox3d.json .gitignore pipeline/scripts/ && git commit ... && git push` (chưa làm — chờ user xác nhận commit trước, không tự ý commit) — nếu không notebook Colab clone về sẽ thiếu `tower_bbox3d.json` và fail ở bước build mask.
2. Chạy `downloads/B2_antenna_2stage_30k_30k.ipynb` trên Colab/Kaggle GPU. Ước lượng thời gian: ~ tương đương 1 lần chạy 30k đã có (không tính CUDA build vì đã bỏ) + thêm ~1 lần 30k nữa (stage 2) + bước build mask (CPU, vài giây) → tổng có thể ~1.5-2x thời gian lần chạy an toàn vừa xong.
3. Cell báo cáo cuối so sánh 3 vùng (full/tower/skyline) giữa stage1 vs stage2 vs baseline. Quyết định theo đúng logic đã ghi trong cell: nếu `tower-crop` tăng mà `full-image` không giảm rõ → dùng stage2 nộp bài; nếu `tower-crop` tăng nhưng `full-image` giảm → đánh đổi, cân nhắc; nếu không đổi/giảm cả hai → antenna loss không giúp gì cho `hcm0031` (khớp tiền lệ `HCM0421`), quay lại dùng ứng viên `0.6714` (stage 1) đã có sẵn.
4. Nếu hết giờ giữa chừng: ứng viên an toàn `0.6714` (từ `B2_submission_candidate_exposure_off_safe_densify.ipynb`, mục trước) vẫn dùng được ngay, không phụ thuộc kết quả thử nghiệm này.
