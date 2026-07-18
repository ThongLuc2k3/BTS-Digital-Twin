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

- [ ] **[CẦN USER — ƯU TIÊN CAO NHẤT]** Phase E (final): với MỖI scene, chạy
      `kaggle_private.ipynb` với `MODE="final"` (100% ảnh train, `ITERATIONS=30000`) +
      ĐÚNG cấu hình đã thắng của scene đó (bảng "Stage 1 — cấu hình thắng CHỐT" ở lịch
      sử bên dưới) để lấy checkpoint nộp bài thật. Đây là việc **ưu tiên hơn** Phase
      C/D bên dưới — theo đúng thứ tự plan.md mục 7 và lời khuyên "nộp sớm, không chờ
      tới hạn chót mới nộp bản đầu tiên". Config cụ thể từng scene:
      - `HCM0421`,`HCM0539`,`HCM0540`,`HCM0644`,`HCM0674`,`bonsai`: cấu hình A
        (`ANTIALIASING=1, DEPTH_PRIOR=0, EXPOSURE_COMP=0, ANTENNA_FOCUS=0`)
      - `chair`: cấu hình B (`ANTIALIASING=1, DEPTH_PRIOR=1`, còn lại `0`)
      Nhớ kiểm tra banner in xác nhận cấu hình trước khi để chạy tiếp (mới thêm, tránh
      lặp lỗi quên đổi tham số).
- [ ] Sau khi có đủ 7 checkpoint final: đóng gói bằng `kaggle_submission.ipynb`, kiểm
      tra checklist plan.md mục 8, nộp thử sớm 1 bản an toàn.
- [ ] (Sau khi có bản nộp an toàn, nếu còn thời gian) Phase C — TRR Tier-1 trên render
      thật: cần tải ảnh render thật (`holdout_renders/*.png`) từ Kaggle về (hiện chưa có
      cục bộ, chỉ có `eval_metrics.txt`/notebook) để tune `k_neighbors`/`patch_size`/
      `blend_alpha` bằng `11_trr_refine.py`, đo Score thật có tăng không trước khi bật
      vào pipeline nộp bài (plan.md mục 6.3).
- [ ] (Sau khi có bản nộp an toàn, nếu còn thời gian) Phase D — antenna-focus (chỉ 5
      scene BTS): **hạ tầng + bug scale đã sửa, đã wiring vào notebook, đã có
      `antenna_weights.json` thật cho `HCM0421`** (xem lịch sử "phát hiện bug nghiêm
      trọng ở antenna-focus" 2026-07-18) — set `ANTENNA_FOCUS=1` là chạy được ngay cho
      `HCM0421`. 4 scene BTS còn lại (`HCM0539/0540/0644/0674`) CHƯA có entry trong
      `_ANTENNA_REF` (cell antenna trong `kaggle_private.ipynb`) — agent tự xem ảnh
      train + chọn khung được (đã chứng minh khả thi), chỉ cần làm khi quyết định mở
      rộng Phase D ra ngoài scene proxy.
- [ ] (Ưu tiên thấp, chỉ nếu rất dư thời gian) `feature/gsplat-mcmc` — pipeline train
      thay thế (rasterizer `gsplat` + MCMC densification), có code sẵn ở nhánh cũ
      round-1 nhưng CHƯA port sang round-2 (cấu trúc dataset khác, giống công port
      mip-splatting/depth-anything-v2 đã làm nhưng tốn công hơn vì là pipeline SONG
      SONG khác hẳn, không phải chỉ thêm cờ). plan.md mục 6.2: chỉ đáng thử cho
      `bonsai` (scene indoor dày Gaussian) — round 1 kết luận "chưa thắng" trên BTS
      không tự động đúng cho domain khác.
- [ ] Backlog (chưa làm, không chặn): `pipeline/kaggle_public.ipynb` giờ lỗi thời hoàn
      toàn (round-1, sẽ crash trên dataset round-2) — có vài cell trực quan hoá hữu ích
      (biểu đồ Score/ảnh, xem N ảnh tệ nhất theo PSNR) đáng port sang
      `kaggle_private.ipynb` chế độ holdout trước khi xoá hẳn file này.
- [ ] `02_validate_frame.py` đã đổi sang scene `HCM0421` (round 2) nhưng CHƯA chạy thật
      (tự chạy COLMAP đầy đủ, tốn thời gian — script này TUỲ CHỌN, không bắt buộc).
- [ ] (Tuỳ chọn, không chặn) Nếu còn dư thời gian sau Phase E an toàn: thử lại cấu hình
      B công bằng (15000 iter, có fix OOM) cho 6 scene còn thiếu dữ liệu công bằng
      (HCM0421/539/540/644/674, bonsai) — xem lịch sử "Chốt Stage 1" bên dưới, user đã
      chọn bỏ qua việc này để không tốn thêm Kaggle GPU quota, ưu tiên nộp bài trước.

## CHỐT — checklist chạy Phase E (2026-07-18, code đã sẵn sàng)

**1. Với MỖI trong 7 scene**, chạy `kaggle_private.ipynb` (đã pull code mới nhất từ
GitHub — nhớ push code đã sửa lên trước khi Kaggle clone), Bước 6 sửa:

| Scene | `SCENE` | `MODE` | `ANTIALIASING` | `DEPTH_PRIOR` | `EXPOSURE_COMP` | `ANTENNA_FOCUS` |
|---|---|---|---|---|---|---|
| 1 | `HCM0421` | `final` | 1 | 0 | 0 | 0 |
| 2 | `HCM0539` | `final` | 1 | 0 | 0 | 0 |
| 3 | `HCM0540` | `final` | 1 | 0 | 0 | 0 |
| 4 | `HCM0644` | `final` | 1 | 0 | 0 | 0 |
| 5 | `HCM0674` | `final` | 1 | 0 | 0 | 0 |
| 6 | `bonsai` | `final` | 1 | 0 | 0 | 0 |
| 7 | `chair` | `final` | 1 | **1** | 0 | 0 |

Đọc banner xác nhận cấu hình ngay sau cell tham số trước khi để chạy tiếp (30000
iteration, ~1-2 tiếng/scene). Xong mỗi scene: theo "Bước 7" trong notebook, tải NGUYÊN
thư mục `pipeline/work/<SCENE>/gs_model/` (không phải chỉ file `.ply`) lên Google Drive,
share "Anyone with the link", lấy link thư mục đó.

**2. Sau khi có đủ 7 link thư mục Drive**: mở `kaggle_submission.ipynb` (đã fix 2 bug ở
trên), điền 7 link vào `CHECKPOINT_LINKS` (Bước 5), `Run All` — notebook tự tải, render,
đóng gói `submission.zip`, tự kiểm tra dung lượng < 350MB.

**3. Trước khi nộp thật**: đối chiếu checklist `plan.md` mục 8 (định dạng, tên file,
đủ scene/ảnh) rồi nộp — đây là **bản nộp an toàn đầu tiên**, nộp sớm, không chờ Phase
C/D xong mới nộp.

**4. Sau khi có bản nộp #1**: quay lại Phase C (TRR Tier-1). LƯU Ý: `MODE=final` render
`test_poses.csv` — KHÔNG có GT (đề bài không cấp GT test) nên không tự đo Score được ở
đó. Muốn TUNE `k_neighbors`/`patch_size`/`blend_alpha` (biết TRR có thật sự tăng Score
không) vẫn cần render trên **holdout** (có GT) để so trước/sau — tải thêm
`holdout_renders/*.png` của 2-3 scene đại diện (1 BTS, 1 non-BTS, theo plan.md mục 6.3)
từ 1 lần chạy `MODE="holdout"` (đã có sẵn eval_metrics từ Phase A/B, nhưng ảnh render
thật thì CHƯA tải về máy). Sau khi chốt tham số tốt trên holdout, mới ÁP vào render
`test_poses.csv` thật của bản final (bước này không cần GT, chỉ cần ảnh train + geometry
COLMAP) rồi đóng gói lại, nộp bản #2.

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

### 2026-07-16 — Commit + chuẩn bị push (user hỏi cách chạy Phase A)
- User hỏi chạy Phase A thế nào — phát hiện: toàn bộ code hôm nay CHƯA commit, notebook
  clone GitHub không chỉ định nhánh → sẽ lấy `main` (chưa có gì hôm nay), Kaggle sẽ chạy
  code cũ và crash. Đã hỏi user cách xử lý — chọn: commit + push nhánh hiện tại, sửa
  notebook clone đúng nhánh đó (không đụng `main`).
- Commit `0f85a05`: toàn bộ Phase 0 + port kỹ thuật + Phase C (21 file, loại trừ
  `chat.txt` và file `.tmp` rác Jupyter khỏi commit).
- Commit `c9d2d1b`: thêm `GIT_BRANCH = "coordination/round1-status"` vào cell clone của
  `kaggle_private.ipynb` (trước đó clone không chỉ định nhánh → lấy nhầm `main`).
- User gửi link Google Drive dataset thật (`178EL7jCSVD59q19SMpeOgnOfOIC66I_t`) — cập
  nhật `GDRIVE_URL` trong `kaggle_private.ipynb` + `kaggle_submission.ipynb` (bỏ qua
  `kaggle_public.ipynb`, đã lỗi thời). Commit `d98e7e9`.
- **Đã push cả 3 commit lên `origin/coordination/round1-status`** (được user xác nhận).
  Nhánh này giờ có đủ code để Kaggle chạy Phase A.

## Trạng thái hiện tại (cập nhật)
- `kaggle_private.ipynb` sẵn sàng 100%: đúng nhánh git, đúng link dataset thật. User có
  thể bắt đầu chạy Phase A trên Kaggle ngay.

## Trạng thái hiện tại (cập nhật 2026-07-17, sau Phase A)
- **Phase A hoàn tất**: đủ 7/7 scene có Score holdout cấu hình A (bảng đầy đủ ở lịch sử
  "Phase A hoàn tất" bên dưới). Đang chờ user chạy Phase B (`DEPTH_PRIOR=1`) trên cả 7
  scene để so sánh — xem "Bước tiếp theo" ở trên.

## Trạng thái hiện tại (cập nhật 2026-07-18, sau bug OOM Phase B)
- Lần chạy Phase B đầu tiên (`DEPTH_PRIOR=1`) FAIL 6/6 scene có notebook (`chair` fail
  nặng hơn, không có notebook) vì CUDA OOM giữa chừng train — xem chi tiết + fix ở lịch
  sử "Bug CUDA OOM Phase B" bên dưới. Đã sửa `pipeline/kaggle_private.ipynb` (chưa test
  lại trên Kaggle thật — cần user chạy lại để verify fix).
- **Bước tiếp theo ngay**: user chạy lại Phase B (`DEPTH_PRIOR=1`) trên Kaggle với
  `kaggle_private.ipynb` đã sửa, cho cả 7 scene (kể cả `chair` — chưa có kết quả nào).

## Trạng thái hiện tại (cập nhật 2026-07-18b, phát hiện lỗi thao tác user)
- User đã chạy lại 1 lần (`Result/Lần 2_ anti=depth=1_15k_inter/HCM0421/`, không OOM,
  chạy hết 15000 iteration) — NHƯNG đọc trực tiếp output cell tham số trong notebook cho
  thấy **`DEPTH_PRIOR=0`** (không phải 1 như tên thư mục ghi) — tức lần chạy này thực
  chất là LẶP LẠI cấu hình A (quên đổi tham số trước khi bấm chạy), KHÔNG PHẢI dữ liệu
  Phase B hợp lệ. Score ra gần như giống hệt bản Lần 1 (0.6618 vs 0.6616 — sai khác nằm
  trong nhiễu ngẫu nhiên của train, không phải hiệu ứng depth prior), củng cố thêm bằng
  chứng đây là chạy trùng cấu hình A, không phải B.
- **Bước tiếp theo ngay**: user cần **kiểm tra kỹ dòng in xác nhận** (`SCENE=... MODE=...
  DEPTH_PRIOR=...`) ngay sau khi chạy cell tham số, TRƯỚC KHI để cả notebook chạy ~1
  tiếng — rồi chạy lại đúng `DEPTH_PRIOR=1` cho `HCM0421` (và 6 scene còn lại, vẫn CHƯA
  có dữ liệu Phase B hợp lệ nào cả) bằng `kaggle_private.ipynb` bản đã fix OOM. Vì lần
  chạy vừa rồi không thật sự bật depth prior, **fix SH_DEGREE=2/DENSIFY_GRAD_THRESHOLD
  vẫn CHƯA được kiểm chứng thật** (code fix chỉ kích hoạt khi DEPTH_PRIOR=1) — vẫn cần
  1 lần chạy đúng cấu hình để biết fix có đủ tránh OOM không.

### 2026-07-17 — Dọn dẹp file/folder thừa
- User yêu cầu rà và xoá file/folder thừa. Xoá ngay (rác rõ ràng, không mất giá trị):
  `chat.txt` (1 byte, rỗng), `pipeline/kaggle_private.ipynb.tmp.1522.02b1ca5a101d` (file
  tạm Jupyter), `pipeline/extra/` (chỉ còn `__pycache__` rỗng, hết source), 3
  `__pycache__/` khác (tự sinh lại).
- Hỏi + được xác nhận xoá thêm 2 thứ (có giá trị thật, không phải rác thuần):
  - `pipeline/kaggle_public.ipynb` (tracked git, đã lỗi thời hoàn toàn cho round 2 — vai
    trò đã thay bằng chế độ holdout trong `kaggle_private.ipynb`). Commit `4cc00a5`
    (recoverable qua git history nếu cần xem lại).
  - `Dataset/VAI_NVS_DATA/` (3.2GB, dữ liệu Round 1, không nằm trong git, vòng thi đã bị
    BTC bỏ) — xoá thẳng đĩa, giải phóng 3.2GB.
- **Giữ lại, KHÔNG xoá** (có giá trị thật, không phải rác): `Kết quả/` (kết quả/notebook
  thật của Round 1, phục vụ khoá luận), `checkpoints/checkpoint_GGdrive.txt` (link Drive
  tới checkpoint đã nộp thật Round 1, Score 58.67320), `pipeline/work/` (holdout split đã
  tạo cho cả 7 scene, tái dùng được), `REVIEW_REPO_2026-07-16.md` (báo cáo review có nội
  dung thật, không rõ nguồn gốc tạo ra trong phiên nào — CHƯA hỏi user, để nguyên).
- **Chưa push** commit `4cc00a5` (xoá kaggle_public.ipynb) — cần hỏi user (GDRIVE_URL đã
  push từ trước).

### 2026-07-17 — User báo lỗi Kaggle "An exception has occurred" ở bước tải dataset
- User dán thông báo lỗi Jupyter chung chung (không traceback đầy đủ) ở bước tải/giải
  nén dataset. Chưa có traceback thật — nhưng rà lại cell 11 (`kaggle_private.ipynb`,
  tự dò thư mục `VAI_NVS_DATA_ROUND2` trong zip vừa giải nén) phát hiện lỗi thiết kế
  thật: bắt buộc phải có 1 thư mục con tên ĐÚNG CHỮ "VAI_NVS_DATA_ROUND2" mới nhận
  diện được — nếu file zip user upload giải nén ra KHÔNG có đúng lớp thư mục bọc
  ngoài đó (vd giải nén thẳng scene ra gốc, hoặc thư mục bọc ngoài đặt tên khác) sẽ
  báo `SystemExit` dù dữ liệu vẫn đầy đủ.
- Sửa: bỏ yêu cầu tên thư mục phải khớp chữ — chấp nhận bất kỳ thư mục nào (kể cả
  gốc giải nén `_dataset_raw` nếu zip không có lớp bọc) miễn chứa đủ >= 4/7 scene
  mong đợi trực tiếp bên trong (`_MIN_MATCH = 4`, tránh khớp nhầm thư mục rác). Test
  bằng dữ liệu thật (symlink 3 scene thật vào 2 cấu trúc thư mục giả lập — có/không
  lớp bọc ngoài) — cả 2 đều detect đúng. Cũng sửa 1 dòng print còn sót chữ "phase1"
  (tàn dư round 1) trong cell tải dataset. Commit `f044ffa`.
- Vẫn CHƯA xác nhận đây có phải NGUYÊN NHÂN THẬT của lỗi user gặp hay không (chưa có
  traceback) — đã hỏi lại user traceback đầy đủ/tên cell lỗi để xác nhận.
- User xác nhận — đã push cả 2 commit lên `origin/coordination/round1-status`.

### 2026-07-17 — Nhận kết quả Kaggle thật đầu tiên: 2/7 scene, Phase A cấu hình A
- User đã tự chạy Kaggle xong, dán về `Result/HCM0421/` và `Result/HCM0539/`
  (notebook đã chạy + `eval_metrics.txt` + `holdout_poses.txt` + `manifest.txt`).
- Cả 2 đều chạy `MODE="holdout"`, cấu hình `A` (`ANTIALIASING=1, DEPTH_PRIOR=0,
  EXPOSURE_COMP=0, ANTENNA_FOCUS=0`), `ITERATIONS=15000` — đúng quy trình Phase A.
- Đọc `eval_metrics.txt`, tính trung bình 30 ảnh holdout/scene:
  - `HCM0421`: PSNR=21.248dB, SSIM=0.6692, LPIPS=0.1666, **Score=0.6616**
  - `HCM0539`: PSNR=21.307dB, SSIM=0.6877, LPIPS=0.1623, **Score=0.6692**
  - Pipeline (COLMAP holdout → train → render → eval) chạy sạch trên GPU Kaggle thật,
    xác nhận hạ tầng Phase 0 hoạt động đúng như thiết kế, không phát sinh bug mới.
  - Chưa có mốc so sánh (chưa chạy cấu hình B/TRR trên 2 scene này) nên chưa kết luận
    được gì về việc cấu hình A có phải lựa chọn tốt nhất không — chỉ là baseline.
- Theo plan.md mục 7, Phase A yêu cầu chạy cấu hình A trên **toàn bộ 7 scene** trước
  khi sang Phase B — còn thiếu 5 scene (`HCM0540`, `HCM0644`, `HCM0674`, `bonsai`,
  `chair`). Đã cập nhật "Bước tiếp theo" ở trên, chưa tự chạy được (không có Kaggle
  API key/GPU ở local).

### 2026-07-17 — Phase A hoàn tất: đủ 7/7 scene, cấu hình A (baseline)
- User đã tự chạy nốt 5 scene còn lại trên Kaggle, dán kết quả về `Result/<scene>/`.
  Đã xác nhận (đọc trực tiếp cell tham số trong từng notebook) cả 7/7 scene đều dùng
  đúng cấu hình A (`ANTIALIASING=1, DEPTH_PRIOR=0, EXPOSURE_COMP=0, ANTENNA_FOCUS=0`,
  `MODE="holdout"`, `ITERATIONS=15000`) — kết quả so sánh được với nhau, không lệch
  cấu hình.
- Bảng Score trung bình holdout (đọc `eval_metrics.txt`, N = số ảnh holdout/scene):

  | Scene    | N  | PSNR (dB) | SSIM   | LPIPS  | Score  |
  |----------|----|-----------|--------|--------|--------|
  | HCM0421  | 30 | 21.248    | 0.6692 | 0.1666 | 0.6616 |
  | HCM0539  | 30 | 21.307    | 0.6877 | 0.1623 | 0.6692 |
  | HCM0540  | 30 | 21.668    | 0.6754 | 0.1695 | 0.6648 |
  | HCM0644  | 30 | 20.279    | 0.6719 | 0.1732 | 0.6540 |
  | HCM0674  | 30 | 21.146    | 0.6941 | 0.1631 | 0.6699 |
  | bonsai   | 31 | 25.742    | 0.8318 | 0.2427 | 0.7069 |
  | chair    | 26 | 23.544    | 0.7550 | 0.2823 | 0.6549 |

  - 5 scene BTS khá đồng đều (Score 0.654–0.670), `bonsai` cao nhất (indoor, dense —
    dễ dựng 3DGS hơn), `chair` thấp nhất trong nhóm dù PSNR/SSIM cao hơn BTS (LPIPS
    cao 0.28 kéo Score xuống — nghi khung cảnh có chi tiết/texture phức tạp hơn hoặc
    khẩu độ view thưa hơn ở vài góc, CHƯA điều tra sâu).
  - Không có bug/crash nào khi chạy 5 scene còn lại — hạ tầng Phase 0 ổn định trên cả
    domain BTS lẫn generic (bonsai/chair).
  - **Phase A coi như hoàn tất theo plan.md mục 7** (đủ 7/7 scene có baseline cấu hình
    A). Đã cập nhật "Bước tiếp theo" chuyển sang Phase B (`DEPTH_PRIOR=1`, cả 7 scene).

### 2026-07-18 — Bug CUDA OOM Phase B (DEPTH_PRIOR=1) + fix trong kaggle_private.ipynb
- User dán kết quả Phase B (thư mục `Result/Lần 2_ anti=depth=1_lỗi 7k_inter/`, cấu hình
  `ANTIALIASING=1, DEPTH_PRIOR=1`) — đọc log lỗi trong 6 notebook (`HCM0421/539/540/644/
  674`, `bonsai`) phát hiện **FAIL 6/6**: `torch.OutOfMemoryError: CUDA out of memory`
  của `diff_gaussian_rasterization` giữa chừng train, ở các mốc rất khác nhau (59%-97%
  của 15000 iteration: 8900/15000 tới 14500/15000) — GPU Kaggle báo capacity ~14.56GiB,
  gần cạn sát nút lúc crash (13.8-14.5GiB đã dùng) ở TẤT CẢ 6 scene, không phải lỗi
  riêng scene nào. `chair` fail nặng hơn nữa — không có cả notebook lẫn nội dung trong
  `manifest.txt`/`eval_metrics.txt` (0 dòng), tức crash sớm hơn/nặng hơn 6 scene kia,
  không đọc được log để biết chi tiết.
  - Checkpoint `iteration_7000` vẫn sống sót ở cả 6 scene fail (đúng thiết kế
    `SAVE_ITERATIONS` của `03_train_3dgs.sh`) — dùng tạm được nhưng KHÔNG so sánh công
    bằng với Score cấu hình A (đã đo ở iteration 15000 đủ).
- **Nguyên nhân suy luận** (không có log memory chi tiết hơn để xác nhận 100%, nhưng
  khớp với việc cấu hình A cùng `ITERATIONS=15000` chạy trót lọt 7/7 trước đó): thêm
  depth-prior loss đẩy densify sinh thêm Gaussian/dùng thêm bộ nhớ đủ để vượt ngưỡng gần
  cạn sẵn có của cấu hình A — không phải do 1 scene cụ thể nặng hơn.
- **Fix đã áp trong `pipeline/kaggle_private.ipynb`** (cell "Bước 5" — set env train):
  khi `DEPTH_PRIOR=1`, tự set thêm `SH_DEGREE=2` và `DENSIFY_GRAD_THRESHOLD=0.0004` —
  đúng tổ hợp biện pháp giảm bộ nhớ mà `03_train_3dgs.sh` đã tự viết sẵn trong comment
  đầu file (ví dụ dùng đúng cặp này). Cấu hình A không đổi gì (đã chạy tốt ở mặc định).
  Validate: `json.load` + `nbformat.validate()` sạch (chỉ warning thiếu `id`, không mới).
- **CHƯA verify trên Kaggle thật** (chưa có GPU để tự chạy) — cần user chạy lại Phase B
  cho cả 7 scene (kể cả `chair`, hiện chưa có kết quả nào) để xác nhận fix đủ.
- **Đánh đổi CẦN BIẾT**: từ giờ Score cấu hình B không còn tách bạch thuần "hiệu ứng
  depth prior" — mà là depth_prior + SH_DEGREE=2 + densify_grad_threshold cao hơn cộng
  lại (SH_DEGREE thấp hơn giảm chi tiết màu/view-dependent hiệu ứng ánh sáng; densify
  threshold cao hơn giảm số Gaussian ở vùng chi tiết mảnh). Nếu sau này muốn so sánh
  tinh khiết hơn, phải train lại cấu hình B trên GPU lớn hơn không cần giảm 2 tham số
  này — nhưng theo mục tiêu thực dụng của plan.md (chọn cấu hình thắng theo Score đo
  được, không phải theo lý thuyết), Score đo được với fix này vẫn dùng chốt được cấu
  hình A/B thắng cho từng scene, chỉ cần biết rõ nó là "B đã điều chỉnh cho vừa GPU".

### 2026-07-18 — Chốt Stage 1 (A/B) + phát hiện thêm: `chair` thật ra chạy trót lọt
- User làm rõ cách xử lý 2 batch kết quả gây nhầm lẫn ở lượt trước:
  - `Result/Lần 2_ anti=depth=1_lỗi 7k_inter/`: user CHỦ ĐỘNG chấp nhận dùng nguyên kết
    quả này làm dữ liệu Phase B (không rerun 6 scene OOM để tiết kiệm quota GPU) — dù
    biết checkpoint chỉ dừng ở 7000 iteration cho 6/7 scene.
  - `Result/Lần 2_ anti=depth=1_15k_inter/HCM0421/`: user xác nhận ĐÚNG là vẫn lỗi
    (thiếu bấm đổi `DEPTH_PRIOR` trước khi chạy, dù notebook đã là bản mới nhất có sẵn
    fix OOM) — không dùng dữ liệu này.
- **Đọc kỹ lại notebook `chair` trong batch "lỗi 7k_inter"**: phát hiện `chair` THỰC RA
  chạy trót lọt HẾT 15000/15000 iteration, KHÔNG OOM (khác 6 scene kia) — dùng
  `sh_degree=3, densify_grad_threshold=0.0002` mặc định (notebook batch này chưa có fix
  SH_DEGREE, vì chạy trước khi agent thêm fix). Tức Score B của `chair` là dữ liệu
  **công bằng 100%** (cùng 15000 iteration với A), không như 6 scene BTS+bonsai kia.
  Khả năng: ảnh `chair` (720×1280, portrait, ít Gaussian hơn scene BTS 4K rộng) không
  đủ tải để chạm ngưỡng OOM dù cùng bật depth-prior.
- **Đã hỏi user cách xử lý 6 scene có Score B không công bằng (7k vs A ở 15k)** — user
  chọn: **chốt A thắng cho 6 scene này** (không tốn thêm Kaggle GPU quota để chạy lại B
  công bằng ngay bây giờ, có thể quay lại sau nếu dư thời gian).
- **Bảng Stage 1 — cấu hình thắng CHỐT** (dùng cho Phase E — final):

  | Scene    | Cấu hình thắng | Score dùng để chọn | Ghi chú |
  |----------|----------------|---------------------|---------|
  | HCM0421  | A              | 0.6616 (15k) vs B 0.6151 (7k, không công bằng) | |
  | HCM0539  | A              | 0.6692 (15k) vs B 0.6251 (7k, không công bằng) | |
  | HCM0540  | A              | 0.6648 (15k) vs B 0.6085 (7k, không công bằng) | |
  | HCM0644  | A              | 0.6540 (15k) vs B 0.6107 (7k, không công bằng) | |
  | HCM0674  | A              | 0.6699 (15k) vs B 0.6234 (7k, không công bằng) | |
  | bonsai   | A              | 0.7069 (15k) vs B 0.7053 (7k, không công bằng) | rất sát, có thể đáng thử lại B công bằng nếu dư thời gian |
  | chair    | **B**          | 0.6616 (15k) vs A 0.6549 (15k) — **công bằng, B thắng thật** | |

### 2026-07-18 — Pre-flight antenna patch (Phase D) — kiểm chứng cục bộ, không cần GPU
- Trước khi khuyến nghị Phase D (antenna-focus), tự kiểm tra rủi ro đã ghi trong comment
  `03_train_3dgs.sh` ("apply_antenna_patch.py viết cho bản train.py CŨ hơn commit đã
  pin, CÓ THỂ không áp sạch") — clone thật `graphdeco-inria/gaussian-splatting`,
  checkout đúng commit pin `54c035f7834b564019656c3e3fcc3646292f727d`, chạy
  `apply_antenna_patch.py --gs_repo` thật. Kết quả: **áp sạch, 8/8 chỗ vá thành công**,
  `python -m py_compile train.py` sau vá không lỗi cú pháp — rủi ro tương thích nêu
  trong comment KHÔNG xảy ra trên commit hiện tại. Hạ tầng code Phase D coi như sẵn
  sàng, chỉ còn thiếu: (1) chọn ảnh + bbox ăn-ten thủ công/scene (việc của người, cần
  xem ảnh), (2) thêm cell gọi 2 script này vào `kaggle_private.ipynb` trước train (chưa
  làm — notebook hiện có biến `ANTENNA_FOCUS` nhưng KHÔNG có cell nào tạo
  `antenna_weights.json`/vá `GS_REPO`, nên nếu bật `ANTENNA_FOCUS=1` ngay bây giờ sẽ báo
  lỗi rõ ràng và dừng, không train nhầm).
- Quyết định thứ tự ưu tiên: theo plan.md mục 7, Phase D chỉ nên làm "nếu còn thời
  gian" — SAU khi có 1 bản nộp an toàn (Phase E). Đã cập nhật "Bước tiếp theo" đổi ưu
  tiên cao nhất sang Phase E (final train + submit) thay vì đi tiếp Phase C/D ngay.

### 2026-07-18 — Fix `10_sanity_check_render.py`: sanity check chạy "thành công" nhưng không kiểm tra được gì
- User chỉ ra file `Result/Lần 2_ anti=depth=1_15k_inter/HCM0421/bts-digital-twin-round2.ipynb`
  có vẻ lỗi render, yêu cầu kiểm tra code. Đọc log cell "Gợi ý sanity-check" (cell 24,
  chạy `10_sanity_check_render.py`) thấy toàn bộ 8/8 ảnh mẫu bị `[BỎ QUA]` vì
  `kích thước khác nhau GT=(989, 1320) render=(981, 1309)` → kết thúc bằng
  "Không có ảnh nào kiểm tra được".
- **Nguyên nhân:** `03_train_3dgs.sh` mặc định (`CLEANUP_DENSE_IMAGES=1`) LUÔN tự xoá
  `colmap/dense/images/` (ảnh đã undistort) ngay sau khi train xong — nên tới lúc cell
  sanity-check chạy, thư mục đó **luôn** đã bị xoá trong thực tế (không phải ca hiếm).
  Code có fallback đọc ảnh gốc ở `Dataset/.../train/images/` — nhưng ảnh đó CHƯA
  undistort (989×1320) trong khi model thật train trên ảnh ĐÃ undistort (981×1309,
  COLMAP tự crop viền vài px khi khử méo) — lệch kích thước nên bị so sánh trượt 100%.
  Hệ quả: script chạy "thành công", không báo lỗi cứng, nhưng hoàn toàn KHÔNG kiểm tra
  được gì — mất tác dụng của chính lưới an toàn nó được viết ra để làm (bắt lỗi
  antialiasing train/render lệch nhau, đúng bug thật đã xảy ra ở round 1).
- **Đã sửa** `pipeline/scripts/10_sanity_check_render.py`: `find_train_image()` giờ trả
  thêm cờ `is_undistorted` (biết ảnh lấy từ nhánh nào); nếu phải dùng ảnh fallback CHƯA
  undistort, `load_img01()` tự resize (PIL BICUBIC) về đúng kích thước render trước khi
  so PSNR/SSIM/LPIPS — không pixel-exact 100% nhưng đủ nhạy để bắt lệch cấu hình lớn
  (antialiasing/sh_degree sai lệch nhiều dB, không thể nhầm với sai số resize ~1%). In
  rõ cảnh báo "[LƯU Ý] N/M ảnh dùng GT XẤP XỈ" ở phần tổng kết. Trường hợp cả 2 ảnh đã
  undistort mà vẫn lệch size thì VẪN giữ nguyên báo lỗi cứng (đó mới là bug thật). Verify:
  `python -m py_compile` sạch (chưa chạy lại thật trên Kaggle — cần GPU).

### 2026-07-18 — Fix BUG NGHIÊM TRỌNG trong `kaggle_submission.ipynb`: thiếu cfg_args/pipeline_train_flags.json khi tải checkpoint
- User yêu cầu chốt lại toàn bộ code trước khi chạy Phase E thật cho 7 scene. Rà lại
  `kaggle_submission.ipynb` (notebook đóng gói submission cuối, dùng SAU khi có đủ 7
  checkpoint final) để đảm bảo an toàn trước khi dùng cho bài nộp thật — phát hiện
  **bug nghiêm trọng, đúng loại lỗi round-1 dự án đã nhiều lần cảnh giác**:
  - Cell "Bước 5" (tải checkpoint) chỉ tải đúng 1 file `point_cloud.ply` qua
    `gdown --fuzzy "{link}" -O <đường dẫn cố định iteration_30000/point_cloud.ply>` —
    KHÔNG tải `cfg_args` và `pipeline_train_flags.json` (2 file này nằm cùng cấp trong
    `gs_model/`, do `03_train_3dgs.sh`/`train.py` tự ghi).
  - Cell "Bước 6" gọi `04_render_test_poses.py --scene {scene}` KHÔNG truyền
    `--antialiasing` thủ công (mặc định `"auto"` — tự đọc từ 2 file trên). Thiếu 2 file
    đó, script chỉ IN CẢNH BÁO rồi **âm thầm mặc định `antialiasing=False`** — trong khi
    **CẢ 7 SCENE đều train với `ANTIALIASING=1`** (cấu hình A hoặc B đều bật). Nếu chạy
    đúng như thiết kế cũ, TOÀN BỘ ảnh render nộp bài (cả 7 scene) sẽ SAI cấu hình
    antialiasing so với lúc train — không có lỗi cứng nào chặn lại, sẽ âm thầm mất điểm
    lớn trên bảng xếp hạng thật (chính là bug mà `10_sanity_check_render.py` được viết
    ra để bắt, xem docstring của nó — nhưng bug này nằm ở notebook ĐÓNG GÓI, ngoài phạm
    vi sanity-check vì không có bước train ngay trước để chạy sanity-check).
  - Markdown hướng dẫn cũ ("Bước 5") ghi rõ SAI: "mỗi scene 1 link file point_cloud.ply
    đơn, KHÔNG phải thư mục gs_model" — ngược hẳn với hướng dẫn ở `kaggle_private.ipynb`
    Bước 7 ("tải thẳng cả thư mục... upload nguyên vậy lên Drive") — 2 notebook lệch
    nhau về format link Drive kỳ vọng, thêm 1 lớp rủi ro nhầm lẫn khi user thao tác.
- **Đã sửa**: cell "Bước 5" giờ tải NGUYÊN thư mục `gs_model` bằng `gdown --fuzzy
  --folder`, tự dò lớp thư mục chứa `cfg_args` (đề phòng `gdown --folder` tự thêm 1 lớp
  thư mục con), rồi ASSERT cứng có đủ `cfg_args` + `pipeline_train_flags.json` + ít nhất
  1 `point_cloud.ply` trước khi copy vào đúng vị trí — báo lỗi rõ ràng ngay nếu thiếu
  thay vì âm thầm render sai. Markdown hướng dẫn cập nhật khớp lại với
  `kaggle_private.ipynb` (link Drive phải là thư mục `gs_model`, không phải file đơn).
- **Bug phụ tìm thấy + sửa luôn (cùng file)**: cell dò thư mục dataset trong zip (Bước
  4) vẫn dùng logic CŨ bắt buộc đúng tên "VAI_NVS_DATA_ROUND2" — đúng bug đã sửa ở
  `kaggle_private.ipynb` ngày 2026-07-17 (commit `f044ffa`) nhưng **chưa được port sang
  `kaggle_submission.ipynb`** (2 notebook tải cùng 1 dataset zip nên cùng rủi ro cấu
  trúc thư mục). Đã port y hệt logic linh hoạt (`_MIN_MATCH=4`, không bắt buộc tên).
  Tiện thể sửa luôn 1 dòng print sót chữ "phase1" (tàn dư round 1, giống lỗi đã sửa ở
  `kaggle_private.ipynb`).
- Verify: `nbformat.validate()` sạch (chỉ warning thiếu `id`, không mới), `ast.parse()`
  từng cell code sạch cú pháp. **CHƯA chạy thật trên Kaggle** (cần có checkpoint thật +
  GPU) — sẽ lộ ra khi user chạy `kaggle_submission.ipynb` lần đầu sau khi có đủ 7
  checkpoint final.
- **Sửa lại nhận định SAI trước đó trong worklog** ("kaggle_submission.ipynb chưa cần
  sửa gì thêm", ghi ngày 2026-07-16) — nhận định đó SAI, chưa từng rà kỹ file này tới
  giờ mới phát hiện 2 bug thật ở trên.

### 2026-07-18 — Đổi phương pháp test: chốt `HCM0421` làm scene proxy xuyên suốt
- User đề xuất: từ giờ không bắt buộc test đủ 7 scene mỗi lần thử ý tưởng mới — chỉ
  chạy 1 scene cố định, coi Score scene đó là tín hiệu đại diện cho hướng đi, tiết
  kiệm GPU quota Kaggle. Đã đồng ý với 1 lưu ý: hợp lý cho nhóm 5 scene BTS (Score
  Phase A đồng đều 0.654-0.670) nhưng `chair`/`bonsai` (domain khác) đã cho thấy phản
  ứng khác hẳn với depth-prior — ý tưởng thắng trên proxy chỉ áp chắc cho nhóm BTS,
  còn `chair`/`bonsai` cần spot-check riêng trước khi áp.
- **Chốt `HCM0421` làm scene test xuyên suốt** — đã có sẵn lịch sử baseline (A/B) để
  so sánh ngay không cần chạy lại từ đầu.
- Hàng đợi test trên `HCM0421`: (1) DEPTH_PRIOR=1 + fix OOM hiện tại — ĐANG CHỜ user
  chạy; (2) nếu vẫn OOM: dùng script backup coarse-to-fine (xem mục dưới); (3) nếu (1)
  ổn: thử cấu hình C = B + antenna-focus; (4) Phase C (TRR) tune trên render thật.

### 2026-07-18 — Xây + test xong script backup: train coarse-to-fine (progressive resolution)
- User đề xuất ý tưởng: nếu vẫn OOM, thử train giai đoạn đầu ở độ phân giải Gaussian
  THẤP hơn (ít Gaussian sinh ra, nhẹ VRAM), sau đó "upsample" tiếp tục train ở độ phân
  giải cao dần tới đúng yêu cầu, thay vì sinh Gaussian trực tiếp ở kích thước đầy đủ.
- **Đã tra trực tiếp source thật của `gaussian-splatting`** (bản đã pin, KHÔNG suy
  đoán) để xác nhận tính khả thi trước khi code: `--resolution` nạp lại mỗi lần chạy
  process (không đông cứng trong checkpoint); `--start_checkpoint <path>.pth` khôi
  phục ĐẦY ĐỦ Gaussian + optimizer + số iteration đã chạy (`first_iter`, không phải
  chỉ load `.ply` tĩnh); `spatial_lr_scale` tính từ world-space, không phụ thuộc
  resolution ảnh. **Phát hiện quan trọng phụ**: `densify_until_iter` mặc định CỐ ĐỊNH
  15000 (không tự co giãn theo `--iterations`) — giải thích được vì sao OOM hay xảy ra
  ở bản holdout (`ITERATIONS=15000` trùng khớp `densify_until_iter`, densify chạy suốt
  không bao giờ ổn định) mà bản final (`ITERATIONS=30000`) có thể ít rủi ro hơn (nửa
  sau 15000-30000 không sinh thêm Gaussian) — CHƯA kiểm chứng thật trên GPU.
- **Tạo mới** `pipeline/scripts/03_train_3dgs_progressive.sh` (không đụng
  `03_train_3dgs.sh` đang chạy tốt cho Phase E thật — tách file riêng để cô lập rủi
  ro). Nhận 1 scene/lần, chia N giai đoạn theo `PROGRESSIVE_RESOLUTIONS`/
  `PROGRESSIVE_FRACTIONS` (mặc định 3 giai đoạn: resolution 4→2→1, tại 30%/60%/100%
  ITERATIONS), tự nối checkpoint `.pth` giữa các giai đoạn, giữ nguyên các cờ
  ANTIALIASING/DEPTH_PRIOR/EXPOSURE_COMP/SH_DEGREE/DENSIFY_GRAD_THRESHOLD như script
  gốc. ANTENNA_FOCUS CHƯA hỗ trợ (báo lỗi rõ nếu bật nhầm — tổ hợp chưa test).
- **Test end-to-end bằng train.py giả** (mock, vì không có GPU cục bộ) — dựng
  `GS_REPO` giả nhận đúng chữ ký CLI thật, mô phỏng ghi `.ply`/`.pth`/`cfg_args`, chạy
  qua `pipeline/work/_test_progressive/` (dọn sau khi test xong). **1 bug thật tìm
  thấy + đã sửa**: kiểm tra "PROGRESSIVE_FRACTIONS phải tăng dần nghiêm ngặt" dùng
  `python3 -c "..." | while read...` qua **process substitution** (`< <(...)`) —
  `set -e` ở shell cha KHÔNG bắt được lỗi bên trong process substitution, nên khi
  python assert fail (vd fractions "0.5 0.3 1.0", giảm ở giữa), script **ÂM THẦM chạy
  tiếp** với ít giai đoạn hơn dự kiến (chỉ 1/3), in "xong toàn bộ" SAI sự thật thay vì
  dừng lại báo lỗi — đúng lớp bug nguy hiểm nhất dự án này luôn cảnh giác (âm thầm
  sai, không phải crash). Sửa: đổi sang `$(...)` (command substitution, exit code
  check được tường minh bằng `||`). Verify lại: 8 kịch bản test (happy path 3 giai
  đoạn, crash giữa giai đoạn 2 + phục hồi bằng checkpoint giai đoạn 1, 4 kiểu input
  sai khác nhau, ANTENNA_FOCUS chặn đúng) — **tất cả đều đúng hành vi mong đợi** sau
  fix, kể cả kịch bản vừa tìm ra bug.
- **Đã thêm cell tuỳ chọn vào `kaggle_private.ipynb`** (chèn ngay sau cell train
  chính, KHÔNG thay thế/tự động chạy) — user tự thay cell nếu cell train chính OOM,
  không đụng luồng chính đang dùng cho Phase E thật.
- **CHƯA test thật trên GPU** — đây là backup, chỉ dùng nếu fix đơn giản
  (SH_DEGREE=2/DENSIFY_GRAD_THRESHOLD=0.0004) ở `03_train_3dgs.sh` không đủ.

### 2026-07-18 — User hỏi "còn ý tưởng khả thi nào chưa thử" — rà nhánh cũ + phát hiện bug nghiêm trọng ở antenna-focus
- Rà 2 nhánh kỹ thuật round-1 chưa port: `feature/gsplat-mcmc` (pipeline train bằng
  `gsplat` + MCMC densification, có sẵn code — plan.md mục 6.2 gợi ý đáng thử cho
  `bonsai`) và `compact/compact-gaussian` (plan.md đã ghi rõ "rủi ro cao, không đủ
  effort để debug", giữ nguyên quyết định bỏ qua). `gsplat-mcmc` là pipeline train
  SONG SONG hoàn toàn khác (rasterizer khác `graphdeco-inria`), port tốn công hơn
  nhiều so với việc chỉ thêm cờ vào `train.py` gốc (như mip-splatting/depth-prior đã
  làm) — xếp ưu tiên THẤP hơn Phase D/E vì chỉ đáng thử cho 1/7 scene (`bonsai`).
- **Dùng khả năng đọc ảnh trực tiếp xem 2 ảnh train thật của `HCM0421`** để tự xác
  định vị trí ăn-ten (việc trước đây coi là "cần mắt người", té ra agent tự làm được)
  — chuẩn bị cho Phase D (antenna-focus).
- **Phát hiện bug nghiêm trọng khi thử chạy `07_build_antenna_weights.py` thật với
  dữ liệu COLMAP cục bộ của `HCM0421`**: script đọc thẳng `points2D.xy` từ
  `pipeline/work/<scene>/colmap/dense/sparse/0` (sparse do `01_run_colmap.py` tự
  sinh qua COLMAP `image_undistorter`) mà KHÔNG quy đổi scale — đúng loại bug đã tìm
  thấy ở `11_trr_refine.py` trước đây (`_detect_points2d_scale`), nhưng ở 1 sparse
  KHÁC (script TRR đọc `scene.provided_sparse_dir` gốc BTC, còn script này đọc sparse
  tự sinh sau undistort) — **tưởng đã tránh được bug vì dùng nguồn khác, hoá ra
  KHÔNG**: verify bằng dữ liệu thật, `image_undistorter` viết lại ĐÚNG
  `camera.width/height` nhưng KHÔNG viết lại `points2D.xy` — vẫn lệch y hệt hệ số cũ
  (HCM0421 **3.9042x**, khớp cột "Scale" plan.md; chair 1.4999x; bonsai 1.0000x —
  test cả 3 scene bằng sparse thật cục bộ). Hệ quả nếu KHÔNG sửa: người dùng nhập
  khung pixel theo đúng ảnh thật đang xem sẽ **LUÔN lọc ra 0 điểm 3D** (verify bằng
  chính khung đã chọn cho HCM0421 — 0 điểm nếu không chia lại scale, 21 điểm nếu có)
  → `07_build_antenna_weights.py` **CHƯA BAO GIỜ CHẠY ĐƯỢC** cho tới hôm nay, dù đã
  qua "pre-flight" patch-compat check trước đó (check đó chỉ verify `train.py` áp
  patch sạch, không test nhánh đọc `points2D.xy` này).
- **Đã sửa** `pipeline/scripts/07_build_antenna_weights.py`: thêm
  `_detect_points2d_scale()` (giống TRR, đo tại runtime, không hardcode số) + chia lại
  toạ độ `points2D.xy` trước khi lọc theo khung `--box`. Verify bằng dữ liệu COLMAP
  thật cục bộ (3 scene có sẵn sparse từ Phase 0): scale đo được khớp chính xác cột
  "Scale" plan.md cho cả 3.
- **Đã tự chọn khung ăn-ten cho `HCM0421`** (ảnh `DJI_20241230093301_0003_V.JPG`, box
  `600 370 770 930`, verify bằng crop + vẽ khung đè lên ảnh thật trước khi chốt) — chạy
  thật script đã sửa: 21 điểm 3D lọt khung, bbox 3D hợp lý, chiếu được vào 240/240 ảnh
  train, coverage trung bình 0.65 (khá cao — hợp lý vì đây là drone survey xoay quanh
  sát chính toà nhà có ăn-ten này, không phải lỗi). Đã ghi thật
  `pipeline/work/HCM0421/antenna_weights.json` — dùng được ngay.
- **Đã wiring vào `kaggle_private.ipynb`**: thêm cell mới (giữa cell cài Depth-Anything-V2
  và cell train) — nếu `ANTENNA_FOCUS=1`: tự vá `GS_REPO` (gọi `apply_antenna_patch.py`
  nếu chưa vá, không vá lại nếu đã vá), tự sinh `antenna_weights.json` cho scene hiện
  tại nếu có trong bảng tra `_ANTENNA_REF` (mới có `HCM0421`) — scene BTS khác chưa có
  entry sẽ in cảnh báo rõ + train bình thường (không crash). Verify: `nbformat.validate()`
  sạch; multi-line `!python ... \` bên trong khối `if:` đúng pattern đã dùng thật (và
  chạy thành công) ở cell render `04_render_test_poses.py` cùng notebook — không phải
  cú pháp mới chưa kiểm chứng.
- **Việc còn thiếu để dùng Phase D cho 4 scene BTS còn lại** (`HCM0539/0540/0644/0674`):
  cần agent tự xem 1 ảnh train/scene (đã chứng minh làm được, xem trên) rồi thêm entry
  vào `_ANTENNA_REF` — CHƯA làm (ưu tiên thấp hơn Phase E đang chạy).
