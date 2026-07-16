# Nhật ký làm việc tự động (multi-agent) — BTS Digital Twin Round 2

> File này ghi lại lịch sử thực hiện và mục tiêu từng bước cho tới kết quả cuối cùng,
> để nếu phiên làm việc bị ngắt ngang (mất kết nối, hết context, crash...) thì có thể
> đọc lại và tiếp tục đúng chỗ dở dang, không mất dấu.
>
> Quy tắc cập nhật: mỗi agent/bước làm việc PHẢI thêm 1 mục vào "Lịch sử" (thứ tự thời
> gian, mới nhất ở CUỐI) khi bắt đầu và khi kết thúc (kể cả thất bại), và cập nhật
> "Trạng thái hiện tại" + "Bước tiếp theo".

## Mục tiêu cuối cùng

Hoàn thành pipeline sinh ảnh Novel View Synthesis cho 7 scene của
`Dataset/VAI_NVS_DATA_ROUND2/` (5 scene BTS + `bonsai` + `chair`), đạt Score cao nhất
có thể trong thời gian còn lại, tuân thủ đúng luật thi ở `plan.md` mục 1 và 9, và nộp
bài đúng định dạng (checklist `plan.md` mục 8).

Kế hoạch chi tiết đầy đủ: xem `plan.md` (roadmap Phase 0 → E).

## Trạng thái hiện tại

- **Giai đoạn:** Phase 0 xong hoàn toàn. `kaggle_private.ipynb` đã sẵn sàng cho
  Phase A — đang chờ user tự chạy trên Kaggle (agent không có GPU/Kaggle API để tự
  chạy phần này).
- **Cập nhật lúc:** 2026-07-16
- **Phạm vi đã xác nhận với user:**
  - Agent tự làm toàn bộ roadmap Phase 0 → E (không cần hỏi lại từng bước, tự chọn
    phương án tốt nhất theo số đo, không quyết định bằng cảm quan).
  - Train thật trên Kaggle (Phase A/B/D) do **user tự chạy tay** trên Kaggle UI — agent
    chỉ chuẩn bị sẵn notebook/config cho từng cấu hình (A, B, ...), KHÔNG tự động hoá
    phần chạy Kaggle (không có `kaggle` CLI/API key trên máy local). User sẽ dán kết
    quả/log về sau khi chạy xong để agent đọc tiếp và quyết định bước sau.
  - Local GPU chỉ có NVIDIA GTX 1650 4GB — không đủ VRAM để train 3DGS đầy đủ, chỉ dùng
    cho việc dev/test nhẹ (đọc COLMAP, undistort, holdout split...).
  - 2 nhánh kỹ thuật cũ (`feature/mip-splatting`, `feature/depth-anything-v2`) KHÔNG
    được git-merge — kỹ thuật của chúng đã được port trực tiếp vào nhánh hiện tại
    (`coordination/round1-status`), xem lịch sử bên dưới.
- **Đang chạy nền:** không có gì đang chạy — đang CHỜ USER chạy Phase A trên Kaggle và
  dán Score về (ngoài khả năng của agent: không có GPU đủ mạnh/Kaggle API key ở local).
  Phase C (TRR Tier-1) đã code xong song song, xem lịch sử bên dưới.

## Bước tiếp theo

- [ ] **[CẦN USER]** Mở `pipeline/kaggle_private.ipynb` trên Kaggle, chạy Phase A:
      với mỗi trong 7 scene, vài version `MODE="holdout"` đổi
      `ANTIALIASING`/`DEPTH_PRIOR`/`EXPOSURE_COMP` để so Score, ghi lại/dán kết quả về
      để agent đọc tiếp và chọn cấu hình thắng từng scene (plan.md mục 6.1). Sau đó 1
      version `MODE="final"` với cấu hình thắng để lấy checkpoint nộp bài.
- [ ] Backlog (chưa làm, không chặn): `pipeline/kaggle_public.ipynb` giờ lỗi thời hoàn
      toàn (round-1, sẽ crash trên dataset round-2) — có vài cell trực quan hoá hữu ích
      (biểu đồ Score/ảnh, xem N ảnh tệ nhất theo PSNR) đáng port sang
      `kaggle_private.ipynb` chế độ holdout trước khi xoá hẳn file này.
- [ ] `02_validate_frame.py` đã đổi sang scene `HCM0421` (round 2) nhưng CHƯA chạy thật
      (tự chạy COLMAP đầy đủ, tốn thời gian — script này TUỲ CHỌN, không bắt buộc).
- [ ] `kaggle_submission.ipynb` chưa cần sửa gì thêm (đã dùng đúng 7 scene round-2, chỉ
      đóng gói từ checkpoint có sẵn — không phụ thuộc holdout).
- [ ] Sau khi có render thật từ Kaggle: chạy lại `11_trr_refine.py` trên render thật
      (không phải bản mờ giả lập) để tune `k_neighbors`/`patch_size`/`blend_alpha` và
      đo Score thật có tăng không trước khi bật vào pipeline nộp bài (plan.md mục 6.3).
- [ ] Phase D (antenna-focus, chỉ 5 scene BTS) — code đã port (task #3), chưa đo.
- [ ] Phase E (chốt & nộp) — chưa bắt đầu, phụ thuộc kết quả Phase A/B/C/D.

## Lịch sử

### 2026-07-16 — Khởi tạo worklog
- Tạo file `WORKLOG.md` này theo yêu cầu của user: mọi việc agent làm/sắp làm/đang làm
  phải được ghi lại lịch sử + mục tiêu từng bước, phòng trường hợp ngắt ngang.
- Rà soát trạng thái repo: branch `coordination/round1-status`, `plan.md` đã được cập
  nhật cho Round 2 (uncommitted), không có task/agent nào đang chạy nền.
- Chưa tìm thấy trong bộ nhớ (memory) hay trong phiên hiện tại nội dung yêu cầu gốc
  "bật nhiều agent tự động làm tới khi ra kết quả tốt nhất" mà user nhắc tới (khả năng
  thuộc 1 phiên trước không được lưu lại) → đã hỏi lại phạm vi cụ thể trước khi triển
  khai (agent tự làm hết Phase 0→E, train Kaggle user tự chạy tay).

### 2026-07-16 — Task #2 hoàn tất: colmap_runner test + audit hardcode
- Chạy thật `01_run_colmap.py` (CLI thật, dùng sparse có sẵn) trên 3 scene đại diện —
  không crash sau khi fix (xem bug bên dưới):
  - `HCM0421` (BTS, ngang): 240/240 ảnh đăng ký, 171304 điểm 3D.
  - `chair` (dọc, 720×1280): 205/205 ảnh đăng ký, 80491 điểm 3D — **xác nhận không có
    bug lật/méo tỉ lệ khi undistort**, kích thước sau undistort vẫn đúng (720,1280).
  - `bonsai` (1920×1080): 248/248 ảnh đăng ký, 54422 điểm 3D.
  - Số ảnh "missing" (110/58/28 tương ứng) là hành vi đã biết — sparse BTC dựng từ tập
    ảnh gốc lớn hơn tập train/images/ được cấp, không phải lỗi.
- **Bug thật tìm thấy + đã sửa** (`pipeline/scripts/01_run_colmap.py`): dùng
  `scene.split` (field round-1, đã bị xoá khỏi `scenes.py` round-2) → crash ngay khi
  chạy `--scene <tên>`; cờ `--split public/private` không khớp `domain bts/generic`
  mới nên `--all --split ...` **âm thầm trả về 0 scene** (im lặng không báo lỗi, nguy
  hiểm hơn crash). Đã đổi `scene.split`→`scene.domain`, `--split`→`--domain`
  (choices `bts`/`generic`), cập nhật docstring ví dụ dùng tên scene round 2. Verify
  lại bằng chạy CLI thật trên cả 3 scene — pass.
- Không tìm thêm hardcode resolution/orientation nào khác trong
  `02_validate_frame.py`, `09_diagnose_distance.py`.
- Task #2 đánh dấu `completed`.

### 2026-07-16 — Task #1 hoàn tất: holdout split + eval script
- Tạo `pipeline/scripts/00_make_holdout_split.py` (mới): sinh holdout ~10-15% ảnh
  train/scene, pose lấy trực tiếp từ sparse COLMAP có sẵn qua pycolmap (không phải từ
  test_poses.csv — đúng yêu cầu plan.md mục 4 bước 3).
- Sửa `04_render_test_poses.py`: thêm `--poses_csv` (mặc định vẫn `test_poses.csv`
  thật, không đổi hành vi cũ) để render được cả holdout_poses.csv.
- Sửa `05_eval_metrics.py`: bỏ hết `scene.split`/`scene.gt_test_images_dir`/
  `all_scenes("public")` (nguồn gây crash nêu ở mục trước), chuyển hẳn sang chấm trên
  holdout tự tạo (`work/<scene>/holdout/holdout_gt/` vs
  `work/<scene>/<renders_subdir>/`, mặc định `holdout_renders`).
- **Bug tinh vi tìm thấy + đã sửa khi viết `00_make_holdout_split.py`:**
  `pycolmap.Rigid3d.rotation.quat` trả về thứ tự **[x,y,z,w]**, KHÁC quy ước
  `qw,qx,qy,qz` mà `test_poses.csv`/`common/poses.py` dùng toàn repo. Nếu không đảo
  thứ tự khi ghi CSV, mọi pose holdout sẽ sai hoàn toàn (rotation lệch) mà KHÔNG có
  lỗi rõ ràng nào báo ra — chỉ lộ ra khi so ảnh render bị lệch góc. Đã verify: rotation
  matrix từ CSV ghi ra khớp byte-exact (diff ~1.1e-16) với pose gốc trong sparse.
- Test thật (không mock) đã chạy: `00_make_holdout_split.py` trên `chair` (205→179
  train/26 holdout, 12.7%), `bonsai` (248→217/31, 12.5%), `HCM0421` (240→210/30,
  12.5%), seed=42 — không crash, 0 symlink hỏng, 0 overlap train/holdout. `pipeline/
  work/` đã gitignore nên các holdout split này giữ lại được, tái dùng cho Phase A,
  KHÔNG cần tạo lại. `05_eval_metrics.py` test end-to-end bằng ảnh giả — Score ra đúng
  công thức mục 8.4.
- Task #1 đánh dấu `completed`.

### 2026-07-16 — Chạy holdout split cho 4 scene BTS còn lại
- `00_make_holdout_split.py` trên HCM0539/0540/0644/0674 (seed=42) — mỗi scene 240
  ảnh -> 210 train/30 holdout (12.5%). **Cả 7/7 scene round 2 giờ đã có holdout
  split** ở `pipeline/work/<scene>/holdout/`.

### 2026-07-16 — Sửa 02_validate_frame.py sang scene Round 2
- Đổi `SCENE_NAME` từ `HCM0249` (round-1, không tồn tại ở round-2, sẽ crash) sang
  `HCM0421`. Cập nhật docstring: rủi ro gốc (chỉ 1 scene có sparse tin cậy) không còn
  áp dụng vì cả 7/7 scene round-2 đều có sparse hợp lệ (đã xác nhận ở task #2) —
  script giờ chỉ còn là kiểm định tuỳ chọn, không bắt buộc cho scene nào.
- Chưa chạy thật (tự chạy COLMAP đầy đủ tốn thời gian CPU, không chặn tiến độ).

### 2026-07-16 — Phát hiện quan trọng: nhánh kỹ thuật lệch khỏi Round 2
- `feature/mip-splatting` và `feature/depth-anything-v2` (chứa antialiasing,
  depth-prior, antenna-focus, exposure-comp) vẫn nhánh ra từ code Round 1
  (`a3362e0`) — còn `scenes.py` cũ (public/private split), KHÔNG có
  `00_make_holdout_split.py` — sẽ crash nếu dùng thẳng cho dataset Round 2.
- Đã hỏi user cách xử lý: **chọn port trực tiếp kỹ thuật vào nhánh hiện tại**
  (không git merge/rebase 2 nhánh cũ) — vì phần lớn logic 2 nhánh đó (antialiasing
  flag, depth prior, antenna-focus) KHÔNG phụ thuộc cấu trúc dataset round-1, port
  gần như cơ học. Phát hiện thêm: diff của `05_eval_metrics.py` trên
  `feature/mip-splatting` (psnr_max 30->50, gộp điểm theo scene) **đã có sẵn và đi
  xa hơn** trong bản hiện tại (task #1) — khỏi cần đụng lại file đó.

### 2026-07-16 — Task #3 hoàn tất: port kỹ thuật lên nhánh Round 2
- Sửa: `03_train_3dgs.sh` (cờ ANTIALIASING=1/DEPTH_PRIOR=0/EXPOSURE_COMP=0/
  ANTENNA_FOCUS=0 mặc định, ghi `pipeline_train_flags.json`), `04_render_test_poses.py`
  (tự đọc antialiasing/sh_degree từ cfg_args, merge sạch với `--poses_csv` của task
  #1), `requirements.txt` (opencv-python, DA_REPO, pin commit GS_REPO).
- Mới: `07_build_antenna_weights.py` (thêm chặn sớm nếu domain != bts),
  `apply_antenna_patch.py`, `10_sanity_check_render.py`, `08_generate_depth_priors.py`.
- **2 bug thật tìm thấy + đã sửa khi port:**
  1. `08_generate_depth_priors.py` bản gốc dùng `--split public/private` gọi
     `all_scenes(args.split)` — không khớp chữ ký `all_scenes(domain=...)` của
     `scenes.py` Round 2, sẽ crash. Đổi thành `--domain {bts,generic}`.
  2. `03_train_3dgs.sh` dòng poll tiến độ (`LAST_PROGRESS=$(grep ... | tail -1)`)
     dưới `set -euo pipefail`: `grep` exit 1 khi CHƯA có dòng tiến độ nào trong log
     (rất hay xảy ra ngay đầu lúc train) → cả pipeline exit 1 → `set -e` làm SCRIPT
     CHẾT NGAY GIỮA CHỪNG TRAIN, trông như crash dù thật ra train vẫn đang chạy
     bình thường. Bug PRE-EXISTING (không phải do port gây ra) — phát hiện được nhờ
     chạy end-to-end với train.py giả. Đã thêm `|| true`. Verify: `bash -n` sạch,
     chạy end-to-end giả ra đúng `pipeline_train_flags.json`.
- **Ràng buộc CHƯA kiểm chứng bằng train GPU thật** (local 4GB không đủ, chỉ test
  được cú pháp/`--help`/mô phỏng): `--antialiasing` cần `$GS_REPO` đúng pin commit
  `54c035f7834b564019656c3e3fcc3646292f727d`. Cần verify lại ngay khi chạy Kaggle
  lần đầu, và chạy `10_sanity_check_render.py` sau lần train đầu mỗi config trước
  khi tin số liệu `05_eval_metrics.py` (đúng bài học rút ra từ round 1).
- Task #3 đánh dấu `completed`. **Phase 0 (hạ tầng + kỹ thuật) coi như xong.**

### 2026-07-16 — Nối holdout-eval vào 01_run_colmap.py + fix bug crash rerun
- Thêm `--images_dir` override (tham số nội bộ `process_scene()`) + cờ CLI
  `--holdout` vào `01_run_colmap.py`: khi bật, dùng
  `pipeline/work/<scene>/holdout/train_images` (85-90% ảnh, từ task #1) thay vì
  `scene.train_images_dir` đầy đủ — hoàn thiện nốt mắt xích còn thiếu để thật sự
  train được ở chế độ holdout-eval (trước đó `00_make_holdout_split.py` đã tạo
  sẵn thư mục này nhưng chưa có script nào dùng tới).
- **Bug thật tìm thấy + đã sửa khi tự test bằng dữ liệu thật** (`chair`, so trước/
  sau): chạy `01_run_colmap.py --scene chair --holdout` trên workdir
  `pipeline/work/chair/colmap` ĐÃ CÓ SẴN state từ lần chạy full-data trước đó
  (task #2) → `pycolmap.undistort_images` CRASH cứng ("Uncaught exceptions in
  thread pool destructor: 177 exception(s)") thay vì báo lỗi rõ ràng, do ghi đè
  lên `dense/` cũ ứng với tập ảnh thiếu khác. Nguy hiểm vì đúng kịch bản sẽ xảy
  ra trên Kaggle: chạy holdout-eval trước, rồi chuyển sang final retrain 100%
  ảnh trên CÙNG workdir. Sửa `common/colmap_runner.py::_undistort_and_fix_layout`
  — `shutil.rmtree(dense_dir)` trước khi gọi `undistort_images` nếu đã tồn tại.
  Verify lại bằng dữ liệu thật: rerun `--holdout` chồng lên state cũ (179/179,
  84 ảnh loại) rồi rerun KHÔNG `--holdout` chồng tiếp (205/205, 58 ảnh loại) —
  cả 2 chiều đều chạy sạch, không crash.

### 2026-07-16 — Task #4 hoàn tất + fix thêm: kaggle_private.ipynb sẵn sàng cho Phase A
- Notebook đi từ 22 → 28 cell. Thêm biến `SCENE`/`MODE`(holdout|final)/
  `ANTIALIASING`/`DEPTH_PRIOR`/`EXPOSURE_COMP`/`ANTENNA_FOCUS`, nối đủ chuỗi
  00(holdout split)→01(--holdout nếu MODE=holdout)→08(sinh depth prior nếu
  DEPTH_PRIOR=1, tự cài Depth-Anything-V2)→03(set env flags)→04(--poses_csv holdout
  nếu MODE=holdout)→05(in Score nếu MODE=holdout)→gợi ý 10 (sanity-check). Bước 7
  (lấy checkpoint) làm rõ: CHỈ lấy khi `MODE=final`, quy trình 2 bước mỗi scene
  (nhiều lần holdout so cấu hình → 1 lần final với cấu hình thắng).
- Validate: `json.load` + `nbformat.validate()` sạch (chỉ warning vô hại thiếu
  `id` field, đã có từ notebook gốc, không phải regression).
- **Tự sửa thêm sau khi review** (không qua fork, việc nhỏ): cell train hiện đặt
  cố định `ITERATIONS=30000` cho CẢ 2 chế độ — với `MODE=holdout` (chạy nhiều lần/
  scene chỉ để so Score tương đối giữa cấu hình) dùng 30000 iteration là lãng phí
  quota GPU Kaggle (có hạn, và plan.md nhấn mạnh thời gian gấp). Đổi thành:
  `ITERATIONS=15000` khi `MODE=holdout`, `30000` khi `MODE=final` — vẫn có
  checkpoint ở 7000/15000 (đủ tín hiệu so sánh), giữ nguyên 30000 cho bản nộp thật.
- Task #4 đánh dấu `completed`. **`kaggle_private.ipynb` sẵn sàng để user tự chạy
  Phase A trên Kaggle.**

### 2026-07-16 — Task #5 hoàn tất: TRR Tier-1 (Phase C) — chạy song song, không chặn Kaggle
- Tạo `pipeline/scripts/11_trr_refine.py` (321 dòng) — đúng thuật toán 4 bước
  `plan.md` mục 5: (1) chọn k ảnh train gần nhất theo vị trí+góc nhìn camera; (2)
  chiếu points3D vào pose cần refine (pinhole), occlusion qua gộp lưới ô theo
  `patch_size` (giữ điểm gần camera nhất/ô, vectorized); (3) tra `track` COLMAP lấy
  patch pixel thật quanh toạ độ 2D trong ảnh train; (4) blend nhiều candidate theo
  trọng số góc nhìn×nghịch đảo khoảng cách, rồi blend với render gốc qua
  `--blend_alpha` (mặc định 0.6). Vùng không có correspondence GIỮ NGUYÊN pixel
  render gốc (không hallucinate — đúng tinh thần Tier-1 "an toàn").
- **Bug/quirk dataset NGHIÊM TRỌNG phát hiện khi validate bằng dữ liệu thật:**
  `points2D.xy` trong sparse COLMAP round-2 lưu ở **ĐỘ PHÂN GIẢI GỐC** (trước khi
  BTC hạ xuống ảnh `train/images/` cấp cho thí sinh), lệch với
  `camera.width/height/fx/fy` (đã đúng theo ảnh thật) đúng hệ số cột "Scale" ở
  `plan.md` mục 2 — verify thực nghiệm: `chair` lệch 1.500x, `HCM0421` lệch 4.000x,
  `bonsai` lệch 1.000x (std ~0.01px, không phải nhiễu ngẫu nhiên, là lệch hệ thống).
  Trước khi fix, TRR dùng nhầm toạ độ patch → làm GIẢM điểm (-4 đến -7dB) thay vì
  tăng. **KHÔNG ảnh hưởng train/render 3DGS hiện có** (train.py không đọc
  points2D.xy trực tiếp) — chỉ lộ ra ở TRR vì đây là chỗ ĐẦU TIÊN trong pipeline
  đọc field này trực tiếp. Đã thêm `_detect_points2d_scale()` tự đo hệ số lúc
  runtime (không hardcode số từ plan.md, tự thích nghi nếu BTC đổi scale). Verify
  bằng self-test riêng: project 1 ảnh train vào chính pose của nó, mean abs diff
  vùng covered giảm từ 46.4→13.7 sau fix.
- **Validate thật (không mock)** — làm mờ mạnh ảnh holdout GT thật giả lập "render
  3DGS mờ", so PSNR/SSIM trước/sau TRR (vì CHƯA có render GPU thật, đang chờ
  Kaggle):
  - `chair`: PSNR vùng TRR sửa 20.92→22.36dB (**+1.45dB**), coverage TB ~10.6%.
  - `HCM0421`: PSNR vùng TRR sửa 17.60→19.29dB (**+1.69dB**), coverage TB ~7.0%.
- **Giới hạn đã biết:** coverage thấp (~3-13%, bản chất Tier-1 — vùng còn lại là
  việc của Tier-2 3DGS-Enhancer, CHƯA làm); warp = translate thô (chưa affine);
  tham số `k_neighbors`/`patch_size`/`blend_alpha` CHƯA tune trên render 3DGS thật
  (chỉ có sau khi Kaggle train xong) — mặc định hiện tại dựa trên test tổng hợp
  bằng ảnh mờ giả lập, có thể cần chỉnh lại khi có render thật.
- Task #5 đánh dấu `completed`.
